"""Byzantine threat models.

Two families are simulated:
  * data-poisoning  -- the client trains honestly on corrupted data;
  * model-poisoning -- the client returns an update crafted from the honest ones.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

DATA_POISON = {"label_flip", "noisy_label"}
MODEL_POISON = {"sign_flip", "gauss", "scaling", "alie", "ipm", "adaptive"}
ALL_ATTACKS = ["none"] + sorted(DATA_POISON | MODEL_POISON)


def poison_labels(y: torch.Tensor, attack: str, num_classes: int,
                  gen: torch.Generator) -> torch.Tensor:
    if attack == "label_flip":                       # deterministic full flip
        return (num_classes - 1 - y).long()
    if attack == "noisy_label":                      # 60% uniform random relabelling
        y = y.clone()
        mask = torch.rand(len(y), generator=gen) < 0.6
        y[mask] = torch.randint(0, num_classes, (int(mask.sum()),), generator=gen)
        return y
    return y


def apply_model_poison(updates: Dict[int, torch.Tensor], malicious: List[int],
                       attack: str, gen: torch.Generator,
                       scale: float = 8.0, z: float = 1.5,
                       eps: float = 1.2, rho: float = 0.7) -> Dict[int, torch.Tensor]:
    """Rewrite the malicious clients' updates in place-safe fashion.

    Colluding attackers are assumed to see one another's honest updates, which is
    the standard omniscient-within-coalition assumption used to construct the
    ALIE and IPM attacks.
    """
    if attack not in MODEL_POISON or not malicious:
        return updates
    out = dict(updates)
    mal = [k for k in malicious if k in updates]
    if not mal:
        return out
    honest_stack = torch.stack([updates[k] for k in updates])
    ref_norm = torch.linalg.norm(honest_stack, dim=1).median()

    if attack == "sign_flip":
        for k in mal:
            out[k] = -3.0 * updates[k]
    elif attack == "gauss":
        for k in mal:
            g = torch.randn(updates[k].shape, generator=gen)
            out[k] = g / torch.linalg.norm(g) * ref_norm * 3.0
    elif attack == "scaling":
        # model-replacement: push the mean of the coalition, amplified
        base = torch.stack([updates[k] for k in mal]).mean(0)
        base = -base / (torch.linalg.norm(base) + 1e-12) * ref_norm
        for k in mal:
            out[k] = base * scale
    elif attack == "alie":
        # "a little is enough": stay inside the honest population's variance.
        # The coalition estimates the mean and the (population) standard
        # deviation of its own honest updates; a singleton coalition has no
        # variance estimate and therefore submits its honest update unchanged.
        S = torch.stack([updates[k] for k in mal])
        mu = S.mean(0)
        sd = S.std(0, unbiased=False) if S.shape[0] > 1 else torch.zeros_like(mu)
        pert = mu - z * sd
        for k in mal:
            out[k] = pert
    elif attack == "ipm":
        # inner-product manipulation: negatively correlated with the honest mean
        mu = torch.stack([updates[k] for k in mal]).mean(0)
        for k in mal:
            out[k] = -eps * mu

    elif attack == "adaptive":
        # White-box attack constructed against the four trust statistics of
        # Section IV. The coalition submits
        #
        #     Delta_k^B = M_t ( rho * mu_hat + sqrt(1 - rho^2) * b_hat ),
        #
        # where mu_hat is the unit coalition-mean direction, b_hat a unit vector
        # orthogonal to it that is FIXED for the whole run, and M_t the cohort
        # median norm. By construction the update has
        #   * directional alignment (1 + rho * cos(mu_hat, r)) / 2, high because
        #     the coalition mean tracks the consensus;
        #   * peer agreement 1 inside the coalition and rho outside it;
        #   * magnitude regularity exactly 1, since the norm is the median;
        #   * temporal stability close to 1, since b_hat never changes.
        # The damage is carried by the orthogonal component, which pushes the
        # global model along a fixed direction unrelated to any local optimum.
        mu = torch.stack([updates[k] for k in mal]).mean(0)
        mu = mu / (torch.linalg.norm(mu) + 1e-12)
        b = _fixed_direction(mu.numel(), mu.dtype)
        b = b - torch.dot(b, mu) * mu                  # orthogonalise
        b = b / (torch.linalg.norm(b) + 1e-12)
        d = rho * mu + float(np.sqrt(max(0.0, 1.0 - rho ** 2))) * b
        d = d / (torch.linalg.norm(d) + 1e-12) * ref_norm
        for k in mal:
            out[k] = d
    return out


_FIXED_DIR_CACHE: Dict[int, torch.Tensor] = {}


def _fixed_direction(p: int, dtype) -> torch.Tensor:
    """A direction drawn once and reused for the whole run.

    Holding it fixed is what gives the adaptive attack its temporal
    self-consistency; redrawing it every round would be detected by the fourth
    statistic.
    """
    if p not in _FIXED_DIR_CACHE:
        g = torch.Generator().manual_seed(20260829)
        v = torch.randn(p, generator=g)
        _FIXED_DIR_CACHE[p] = v / torch.linalg.norm(v)
    return _FIXED_DIR_CACHE[p].to(dtype)
