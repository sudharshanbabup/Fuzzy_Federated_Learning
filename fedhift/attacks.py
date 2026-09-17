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
MODEL_POISON = {"sign_flip", "gauss", "scaling", "alie", "ipm"}
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
                       eps: float = 1.2) -> Dict[int, torch.Tensor]:
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
    return out
