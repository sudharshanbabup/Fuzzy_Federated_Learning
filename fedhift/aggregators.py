"""Server-side aggregation rules.

Every aggregator implements the same contract::

    aggregate(updates, sizes, state) -> (delta, info)

where ``updates`` is a (m, p) float tensor of client model deltas, ``sizes`` the
matching sample counts, and ``state`` a mutable dict that lets stateful rules
(FedHIFT) carry reputation across rounds.  ``info`` returns per-client weights
and any diagnostics the experiment driver wants to log.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from .fuzzy import FuzzyTrustEngine, IT2Config, entropic_weights


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def geometric_median(X: torch.Tensor, iters: int = 12, tol: float = 1e-7,
                     w: torch.Tensor | None = None) -> torch.Tensor:
    """Weiszfeld iteration for the (weighted) geometric median of the rows of X."""
    if w is None:
        w = torch.ones(X.shape[0], dtype=X.dtype)
    w = w / w.sum()
    z = (w[:, None] * X).sum(0)
    for _ in range(iters):
        d = torch.linalg.norm(X - z, dim=1).clamp_min(1e-9)
        nw = w / d
        nw = nw / nw.sum()
        z_new = (nw[:, None] * X).sum(0)
        if torch.linalg.norm(z_new - z) <= tol * (torch.linalg.norm(z) + 1e-12):
            z = z_new
            break
        z = z_new
    return z


def cosine_matrix(X: torch.Tensor) -> np.ndarray:
    Xn = X / torch.linalg.norm(X, dim=1, keepdim=True).clamp_min(1e-12)
    return (Xn @ Xn.T).numpy()


def sketch_indices(p: int, dim: int, gen: torch.Generator) -> torch.Tensor | None:
    """Draw a *fixed* random coordinate subset used for all rounds.

    Server-side trust statistics are evaluated on this sketch, so their cost is
    independent of the model dimension p.  The subset must be fixed across
    rounds, otherwise inner products between updates of different rounds (used
    by the temporal-stability statistic) are not comparable.
    """
    if dim <= 0 or dim >= p:
        return None
    return torch.randperm(p, generator=gen)[:dim]


def sketch(X: torch.Tensor, idx: torch.Tensor | None) -> torch.Tensor:
    return X if idx is None else X[:, idx]


def mad(v: np.ndarray) -> float:
    """Median absolute deviation, scaled to be consistent for a normal sample."""
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)))


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def agg_fedavg(U: torch.Tensor, sizes: np.ndarray, st: Dict) -> Tuple[torch.Tensor, Dict]:
    w = torch.tensor(sizes / sizes.sum(), dtype=U.dtype)
    return (w[:, None] * U).sum(0), {"w": w.numpy()}


def agg_fedavg_clip(U: torch.Tensor, sizes: np.ndarray, st: Dict, nu: float = 1.0
                    ) -> Tuple[torch.Tensor, Dict]:
    """FedAvg with every update clipped to nu times the cohort median norm.

    This isolates the contribution of bounded per-client leverage from the
    contribution of trust weighting: it is the size prior of FedAvg with the
    clipping step of FedEFT and nothing else.
    """
    w = sizes / sizes.sum()
    norms = torch.linalg.norm(U, dim=1)
    med = float(norms.median())
    wt = torch.tensor(w, dtype=U.dtype)
    if med < 1e-12:
        return (wt[:, None] * U).sum(0), {"w": w}
    scale = torch.clamp(nu * med / norms.clamp_min(1e-12), max=1.0)
    return (wt[:, None] * (U * scale[:, None])).sum(0), {"w": w}


def agg_median(U: torch.Tensor, sizes: np.ndarray, st: Dict) -> Tuple[torch.Tensor, Dict]:
    return U.median(dim=0).values, {"w": np.full(len(U), 1.0 / len(U))}


def agg_trimmed_mean(U: torch.Tensor, sizes: np.ndarray, st: Dict,
                     beta: float = 0.2) -> Tuple[torch.Tensor, Dict]:
    m = U.shape[0]
    k = int(np.floor(beta * m))
    if 2 * k >= m:
        k = max(0, (m - 1) // 2)
    S, _ = torch.sort(U, dim=0)
    keep = S[k:m - k] if k > 0 else S
    return keep.mean(0), {"w": np.full(m, 1.0 / m)}


def agg_multikrum(U: torch.Tensor, sizes: np.ndarray, st: Dict,
                  f: int | None = None) -> Tuple[torch.Tensor, Dict]:
    m = U.shape[0]
    f = int(st.get("n_byz_est", max(1, m // 5))) if f is None else f
    f = min(f, max(0, (m - 3) // 2))
    d2 = torch.cdist(U, U) ** 2
    nb = max(1, m - f - 2)
    scores = []
    for i in range(m):
        row = torch.cat([d2[i, :i], d2[i, i + 1:]])
        scores.append(torch.sort(row).values[:nb].sum().item())
    order = np.argsort(scores)
    sel = order[: max(1, m - f)]
    w = np.zeros(m)
    w[sel] = 1.0 / len(sel)
    return U[sel].mean(0), {"w": w, "selected": sel.tolist()}


def agg_rfa(U: torch.Tensor, sizes: np.ndarray, st: Dict) -> Tuple[torch.Tensor, Dict]:
    w = torch.tensor(sizes / sizes.sum(), dtype=U.dtype)
    return geometric_median(U, w=w), {"w": w.numpy()}


def agg_fltrust(U: torch.Tensor, sizes: np.ndarray, st: Dict) -> Tuple[torch.Tensor, Dict]:
    """FLTrust: cosine-ReLU trust scores against a server update from root data."""
    g0 = st.get("root_update")
    if g0 is None:
        return agg_fedavg(U, sizes, st)
    n0 = torch.linalg.norm(g0).clamp_min(1e-12)
    Un = torch.linalg.norm(U, dim=1).clamp_min(1e-12)
    cs = (U @ g0) / (Un * n0)
    ts = torch.relu(cs)
    Unorm = U * (n0 / Un)[:, None]              # normalise to the server's magnitude
    s = ts.sum().clamp_min(1e-12)
    w = (ts / s)
    return (w[:, None] * Unorm).sum(0), {"w": w.numpy()}


# --------------------------------------------------------------------------- #
# proposed method
# --------------------------------------------------------------------------- #
class FedHIFT:
    """Heterogeneity-aware Interval Fuzzy Trust aggregation.

    Parameters
    ----------
    temperature : entropic regularisation weight T of the weight allocation.
    nu          : norm-clipping factor (multiples of the median update norm).
    beta_rep    : reputation smoothing factor; 1.0 disables memory.
    sketch_dim  : dimension of the server-side statistic sketch (0 = exact).
    type1       : collapse the FOU and run a type-1 engine (ablation).
    use_size_prior : anchor the weights on the FedAvg size prior.
    """

    name = "fedhift"

    def __init__(self, temperature: float = 0.20, nu: float = 1.0,
                 beta_rep: float = 0.5, sketch_dim: int = 16384,
                 type1: bool = False, use_size_prior: bool = True,
                 cfg: IT2Config | None = None, criteria: Sequence[str] | None = None,
                 seed: int = 0, scorer: str = "fuzzy"):
        self.scorer = scorer
        self.T = temperature
        self.nu = nu
        self.beta_rep = beta_rep
        self.sketch_dim = sketch_dim
        self.use_size_prior = use_size_prior
        self.engine = FuzzyTrustEngine(cfg, type1=type1)
        self.criteria = tuple(criteria) if criteria else ("align", "peer", "norm", "stab")
        self.gen = torch.Generator().manual_seed(seed)
        self._sk_idx: torch.Tensor | None = None
        self._sk_init = False

    # -- statistics ------------------------------------------------------- #
    def _statistics(self, U: torch.Tensor, cids: Sequence[int], st: Dict
                    ) -> Tuple[np.ndarray, float]:
        m = U.shape[0]
        if not self._sk_init:
            self._sk_idx = sketch_indices(U.shape[1], self.sketch_dim, self.gen)
            self._sk_init = True
        Z = sketch(U, self._sk_idx)
        norms = torch.linalg.norm(U, dim=1).numpy()
        med_norm = float(np.median(norms)) + 1e-12

        # Directional (spherical) geometric median: the reference is computed on
        # unit-normalised updates so that a magnitude-inflating adversary cannot
        # drag the consensus direction.
        Zn = torch.linalg.norm(Z, dim=1).clamp_min(1e-12)
        Zdir = Z / Zn[:, None]
        ref = geometric_median(Zdir)
        refn = float(torch.linalg.norm(ref))
        if refn < 1e-12:
            # the unit directions cancel exactly: no reference direction exists,
            # so the alignment statistic carries no information this round
            align = np.zeros(m)
        else:
            align = ((Zdir @ ref) / refn).numpy()                    # in [-1,1]

        C = cosine_matrix(Z)
        np.fill_diagonal(C, -np.inf)
        q = max(1, int(np.ceil(m / 2)) - 1)
        peer = np.sort(C, axis=1)[:, -q:].mean(axis=1)               # in [-1,1]

        ratio = np.log(np.clip(norms / med_norm, 1e-6, 1e6))
        norm_reg = np.exp(-np.abs(ratio) / 0.7)                      # in (0,1]

        hist = st.setdefault("hist_dir", {})
        stab = np.empty(m)
        for i, k in enumerate(cids):
            h = hist.get(k)
            if h is None:
                stab[i] = 0.0            # neutral after the [-1,1] -> [0,1] map
            else:
                stab[i] = float(torch.dot(Zdir[i], h) /
                                torch.linalg.norm(h).clamp_min(1e-12))
            # the memory stores unit directions, so a magnitude-inflating client
            # cannot dominate its own history
            hist[k] = Zdir[i].clone() if h is None else 0.6 * h + 0.4 * Zdir[i]

        u = np.stack([(align + 1) / 2, (peer + 1) / 2, norm_reg, (stab + 1) / 2], axis=1)
        # criterion ablation: a disabled criterion is pinned to the neutral value
        for j, cname in enumerate(("align", "peer", "norm", "stab")):
            if cname not in self.criteria:
                u[:, j] = 0.5
        hetero = self.engine.heterogeneity(u)
        return u, hetero

    # -- aggregation ------------------------------------------------------ #
    def __call__(self, U: torch.Tensor, sizes: np.ndarray, st: Dict
                 ) -> Tuple[torch.Tensor, Dict]:
        m = U.shape[0]
        cids = st.get("client_ids", list(range(m)))
        u, hetero = self._statistics(U, cids, st)
        if self.scorer == "cosine":
            # Non-fuzzy control: the trust degree is the rescaled cosine to the
            # spherical median, fed through the same allocation and clipping.
            tau_inst, span = u[:, 0].copy(), np.zeros(m)
        elif self.scorer == "linear":
            # Non-fuzzy control: an equal-weight linear pool of the same four
            # statistics, again through the same allocation and clipping.
            tau_inst, span = u.mean(axis=1), np.zeros(m)
        else:
            tau_inst, span = self.engine(u, hetero)

        rep = st.setdefault("reputation", {})
        tau = np.empty(m)
        for i, k in enumerate(cids):
            prev = rep.get(k)
            tau[i] = tau_inst[i] if prev is None else \
                (1 - self.beta_rep) * prev + self.beta_rep * tau_inst[i]
            rep[k] = tau[i]

        prior = sizes / sizes.sum() if self.use_size_prior else np.full(m, 1.0 / m)
        w = entropic_weights(tau, prior, self.T)

        norms = torch.linalg.norm(U, dim=1)
        med = float(norms.median())
        if med < 1e-12:                      # at least half the cohort is silent
            Uc = U
        else:
            scale = torch.clamp(self.nu * med / norms.clamp_min(1e-12), max=1.0)
            Uc = U * scale[:, None]

        wt = torch.tensor(w, dtype=U.dtype)
        return (wt[:, None] * Uc).sum(0), {
            "w": w, "tau": tau, "tau_inst": tau_inst, "span": span,
            "hetero": hetero, "u": u,
        }


AGGREGATORS = {
    "fedavg": agg_fedavg,
    "fedavg_clip": agg_fedavg_clip,
    "fedprox": agg_fedavg,          # FedProx differs only in the client objective
    "median": agg_median,
    "trimmed_mean": agg_trimmed_mean,
    "multikrum": agg_multikrum,
    "rfa": agg_rfa,
    "fltrust": agg_fltrust,
}


def build_aggregator(name: str, **kw):
    if name.startswith("fedhift"):
        return FedHIFT(**kw)
    if name == "klcos":
        return FedHIFT(scorer="cosine", **kw)
    if name == "kllin":
        return FedHIFT(scorer="linear", **kw)
    return AGGREGATORS[name]
