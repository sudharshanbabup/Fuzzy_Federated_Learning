"""Interval type-2 fuzzy trust inference engine (the FedHIFT reasoning core).

The engine maps a four-dimensional vector of server-observable update statistics
to a crisp trust degree in [0,1].  Antecedent sets are Gaussian interval type-2
sets with an *uncertain standard deviation*; the width of the footprint of
uncertainty (FOU) is driven at run time by an estimate of the ambient
statistical heterogeneity of the federation.  Consequents are interval
singletons (a zero-order Takagi-Sugeno rule base), so center-of-sets type
reduction admits an exact prefix-sum solution and defuzzification is the
midpoint of the type-reduced interval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# Linguistic label indices used in the rule table.
LOW, HIGH = 0, 1
VARNAMES = ("alignment", "peer agreement", "magnitude regularity", "temporal stability")

# --------------------------------------------------------------------------- #
# Rule base:  (A_align, A_peer, A_norm, A_stab) -> [y_lower, y_upper]
# A_* in {LOW, HIGH}.  Sixteen rules exhaust the antecedent grid.
# --------------------------------------------------------------------------- #
RULE_BASE: List[Tuple[Tuple[int, int, int, int], Tuple[float, float]]] = [
    ((LOW,  LOW,  LOW,  LOW),  (0.00, 0.04)),  # R1  incoherent on every axis
    ((LOW,  LOW,  LOW,  HIGH), (0.02, 0.10)),  # R2  persistent misaligned attacker
    ((LOW,  LOW,  HIGH, LOW),  (0.05, 0.15)),  # R3  erratic, isolated
    ((LOW,  LOW,  HIGH, HIGH), (0.18, 0.32)),  # R4  isolated but self-consistent
    ((LOW,  HIGH, LOW,  LOW),  (0.10, 0.22)),  # R5  cluster member, bad norm, erratic
    ((LOW,  HIGH, LOW,  HIGH), (0.25, 0.40)),  # R6  cluster member, inflated update
    ((LOW,  HIGH, HIGH, LOW),  (0.35, 0.52)),  # R7  cluster member, unstable
    ((LOW,  HIGH, HIGH, HIGH), (0.62, 0.80)),  # R8  healthy MINORITY cluster -> protect
    ((HIGH, LOW,  LOW,  LOW),  (0.12, 0.25)),  # R9  direction-mimicking scaled attack
    ((HIGH, LOW,  LOW,  HIGH), (0.30, 0.45)),  # R10 aligned, abnormal magnitude
    ((HIGH, LOW,  HIGH, LOW),  (0.45, 0.60)),  # R11 aligned, healthy, unstable
    ((HIGH, LOW,  HIGH, HIGH), (0.72, 0.88)),  # R12 aligned and healthy, few peers
    ((HIGH, HIGH, LOW,  LOW),  (0.35, 0.50)),  # R13
    ((HIGH, HIGH, LOW,  HIGH), (0.55, 0.70)),  # R14
    ((HIGH, HIGH, HIGH, LOW),  (0.70, 0.85)),  # R15
    ((HIGH, HIGH, HIGH, HIGH), (0.94, 1.00)),  # R16 fully coherent honest client
]

CENTERS = (0.0, 1.0)  # Gaussian centers of the LOW / HIGH sets on [0,1]

# Only the two *directional* criteria are confounded by statistical
# heterogeneity; the magnitude and temporal criteria are not, so their sets keep
# a degenerate (type-1) footprint of uncertainty.
HETERO_SENSITIVE = (True, True, False, False)


@dataclass
class IT2Config:
    sigma0: float = 0.25      # nominal antecedent width
    kappa: float = 1.20       # FOU sensitivity to measured heterogeneity
    sigma_ref: float = 0.18   # heterogeneity scale at which the FOU saturates
    phi_max: float = 2.2      # cap on the FOU expansion factor
    rules: List[Tuple[Tuple[int, int, int, int], Tuple[float, float]]] = \
        field(default_factory=lambda: list(RULE_BASE))


def gaussian(x: np.ndarray, c: float, s: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def fou_factor(hetero: float, cfg: IT2Config) -> float:
    """Map a robust heterogeneity estimate to the FOU expansion factor phi >= 1."""
    h = float(np.clip(hetero / cfg.sigma_ref, 0.0, 1.0))
    return float(min(1.0 + cfg.kappa * h, cfg.phi_max))


def membership_bounds(u: np.ndarray, phi: float, cfg: IT2Config
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Lower / upper memberships of every input in both linguistic sets.

    Returns two arrays of shape (m, V, 2): [client, variable, {LOW, HIGH}].
    A Gaussian set with uncertain standard deviation sigma in [s_lo, s_hi] has
    its upper MF given by the *wide* Gaussian and its lower MF by the *narrow*
    one.  The expansion factor phi is applied only to the heterogeneity-sensitive
    variables flagged in HETERO_SENSITIVE.
    """
    V = u.shape[-1]
    lo = np.empty(u.shape + (2,))
    up = np.empty(u.shape + (2,))
    for v in range(V):
        pv = phi if (v < len(HETERO_SENSITIVE) and HETERO_SENSITIVE[v]) else 1.0
        s_hi = cfg.sigma0 * pv
        s_lo = cfg.sigma0 / pv
        for j, c in enumerate(CENTERS):
            lo[..., v, j] = gaussian(u[..., v], c, s_lo)
            up[..., v, j] = gaussian(u[..., v], c, s_hi)
    return lo, up


def ekm(y: np.ndarray, f_lo: np.ndarray, f_hi: np.ndarray, side: str) -> float:
    """Exact center-of-sets type reduction for interval singleton consequents.

    Computes min (side='l') or max (side='r') of the weighted average
    sum(f_r y_r) / sum(f_r) over f_r in [f_lo_r, f_hi_r].  The optimum is always
    attained at a *switch point*: the minimiser uses the upper firing strength
    for every rule whose consequent lies below the optimum and the lower firing
    strength above it (Karnik and Mendel).  Enumerating the R+1 switch points
    with prefix sums therefore gives the exact answer in O(R) time after the
    consequents have been sorted once at construction, without the iterative
    fixed point of the classical KM/EKM recursion.
    """
    R = len(y)
    if R == 0:
        return 0.0
    a = f_hi if side == "l" else f_lo          # used on the prefix  [0, k)
    b = f_lo if side == "l" else f_hi          # used on the suffix  [k, R)
    ay = np.concatenate(([0.0], np.cumsum(a * y)))
    an = np.concatenate(([0.0], np.cumsum(a)))
    by = np.concatenate(([0.0], np.cumsum(b * y)))
    bn = np.concatenate(([0.0], np.cumsum(b)))
    num = ay + (by[R] - by)                    # length R+1, indexed by k
    den = an + (bn[R] - bn)
    ok = den > 1e-300
    if not ok.any():
        return float(y.mean())
    c = np.where(ok, num / np.where(ok, den, 1.0), np.nan)
    return float(np.nanmin(c) if side == "l" else np.nanmax(c))


class FuzzyTrustEngine:
    """Interval type-2 TSK engine producing a crisp trust degree per client."""

    def __init__(self, cfg: IT2Config | None = None, type1: bool = False):
        self.cfg = cfg or IT2Config()
        self.type1 = type1
        ante = np.array([r[0] for r in self.cfg.rules], dtype=np.int64)  # (R,4)
        cons = np.array([r[1] for r in self.cfg.rules], dtype=np.float64)  # (R,2)
        order = np.argsort(cons.mean(axis=1), kind="stable")
        self.ante = ante[order]
        self.cons = cons[order]
        self.y_lo = self.cons[:, 0]
        self.y_hi = self.cons[:, 1]
        self.y_mid = self.cons.mean(axis=1)
        self.last_phi = 1.0
        # The switch-point enumeration in ekm() requires both consequent
        # endpoints to be sorted; the design table is monotone in the midpoint,
        # which is asserted here rather than assumed.
        if not (np.all(np.diff(self.y_lo) >= -1e-12) and
                np.all(np.diff(self.y_hi) >= -1e-12)):
            raise ValueError("consequent intervals must be jointly monotone")

    # ------------------------------------------------------------------ #
    def firing(self, u: np.ndarray, phi: float) -> Tuple[np.ndarray, np.ndarray]:
        """Lower / upper firing strengths, shape (m, R). Product t-norm."""
        lo, up = membership_bounds(u, phi, self.cfg)          # (m,4,2)
        m = u.shape[0]
        R, V = self.ante.shape
        f_lo = np.ones((m, R))
        f_hi = np.ones((m, R))
        for v in range(V):
            sel = self.ante[:, v]                              # (R,)
            f_lo *= lo[:, v, :][:, sel]
            f_hi *= up[:, v, :][:, sel]
        return f_lo, f_hi

    def type1_pass(self, u: np.ndarray) -> np.ndarray:
        """Provisional type-1 trust used to identify the compliant majority."""
        f_lo, f_hi = self.firing(u, 1.0)
        f = 0.5 * (f_lo + f_hi)
        denom = f.sum(axis=1)
        denom[denom <= 1e-12] = 1.0
        return np.clip((f * self.y_mid).sum(axis=1) / denom, 0.0, 1.0)

    def heterogeneity(self, u: np.ndarray) -> float:
        """Two-stage estimate of the ambient heterogeneity of the federation.

        A type-1 pass ranks the clients; the robust dispersion of the directional
        statistic is then measured *within the more trusted half only*, so that
        Byzantine outliers cannot inflate the estimate and thereby widen the FOU
        to their own advantage.
        """
        m = u.shape[0]
        if m < 4:
            return 0.0
        tau0 = self.type1_pass(u)
        keep = np.argsort(-tau0)[: max(2, int(np.ceil(m / 2)))]
        a = u[keep, 0]
        med = np.median(a)
        return float(1.4826 * np.median(np.abs(a - med)))

    def __call__(self, u: np.ndarray, hetero: float | None = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Infer trust.

        Parameters
        ----------
        u : (m, 4) array of antecedent inputs in [0,1]
        hetero : optional externally supplied heterogeneity estimate; when None
                 it is computed internally by the two-stage procedure.

        Returns
        -------
        tau : (m,) crisp trust degrees in [0,1]
        span : (m,) width of the type-reduced interval (the engine's own
               uncertainty about the call)
        """
        u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
        if hetero is None:
            hetero = self.heterogeneity(u)
        phi = 1.0 if self.type1 else fou_factor(hetero, self.cfg)
        f_lo, f_hi = self.firing(u, phi)
        self.last_phi = phi
        if self.type1:
            f = 0.5 * (f_lo + f_hi)
            denom = f.sum(axis=1, keepdims=True)
            denom[denom <= 1e-12] = 1.0
            tau = (f * self.y_mid).sum(axis=1) / denom[:, 0]
            return np.clip(tau, 0.0, 1.0), np.zeros(len(u))
        tau = np.empty(len(u))
        span = np.empty(len(u))
        for i in range(len(u)):
            yl = ekm(self.y_lo, f_lo[i], f_hi[i], "l")
            yr = ekm(self.y_hi, f_lo[i], f_hi[i], "r")
            if yr < yl:
                yl, yr = yr, yl
            tau[i] = 0.5 * (yl + yr)
            span[i] = yr - yl
        return np.clip(tau, 0.0, 1.0), span


def entropic_weights(tau: np.ndarray, prior: np.ndarray, temperature: float) -> np.ndarray:
    """Closed-form maximiser of  <w, tau> - T * KL(w || prior)  over the simplex.

    T -> infinity recovers the prior (FedAvg); T -> 0 concentrates all mass on
    the most trusted client (a Krum-like hard selection).
    """
    prior = np.asarray(prior, dtype=np.float64)
    prior = prior / prior.sum()
    z = tau / max(temperature, 1e-8)
    z = z - z.max()
    w = prior * np.exp(z)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        return prior.copy()
    return w / s
