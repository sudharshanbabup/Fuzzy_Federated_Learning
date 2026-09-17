"""Diagnostics behind the numbers quoted in Section VII of the paper.

Two things are checked here that the result JSONs do not record directly:

* `antecedents` prints the mean value of the four fuzzy antecedents, separately
  for benign and Byzantine clients, under each attack. This is what shows that
  magnitude regularity carries detection under sign flipping while directional
  alignment carries it under inner-product manipulation.
* `heterogeneity` prints the measured heterogeneity and the resulting FOU factor
  against the Dirichlet concentration, with and without the two-stage estimator,
  which is the source of the "0.253 against 0.079" figure in Section VII-D.

Usage:  python diagnostics.py [antecedents|heterogeneity|both]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_num_threads(2)

from fedhift.aggregators import FedHIFT, mad          # noqa: E402
from fedhift.data import load_federated               # noqa: E402
from fedhift.fl import FLConfig, run                  # noqa: E402


def antecedents(rounds: int = 10, alpha: float = 0.5):
    fd = load_federated("fmnist", 30, alpha, 0, 20000, 100)
    orig = FedHIFT.__call__
    store: list = []

    def patched(self, U, sizes, st):
        out = orig(self, U, sizes, st)
        store.append((out[1]["u"], out[1]["hetero"], list(st["client_ids"]),
                      np.asarray(out[1]["tau"])))
        return out

    FedHIFT.__call__ = patched
    try:
        for atk in ["sign_flip", "label_flip", "alie", "ipm", "gauss", "scaling"]:
            store.clear()
            cfg = FLConfig(rounds=rounds, eval_every=rounds, aggregator="fedhift",
                           attack=atk, byz_frac=0.2, alpha=alpha, train_subsample=20000)
            r = run(cfg, fd)
            mal = set(r["malicious"])
            B, M, TB, TM = [], [], [], []
            for u, h, cids, tau in store[3:]:          # skip the warm-up rounds
                for i, k in enumerate(cids):
                    (M if k in mal else B).append(u[i])
                    (TM if k in mal else TB).append(tau[i])
            B, M = np.array(B), np.array(M)
            print("%-11s acc=%.3f het=%.3f  W_B=%.3f" %
                  (atk, r["acc_last5"], h, r["mal_weight_mass"]), flush=True)
            print("   benign    align=%.3f peer=%.3f norm=%.3f stab=%.3f | tau=%.3f"
                  % (*B.mean(0), np.mean(TB)))
            print("   Byzantine align=%.3f peer=%.3f norm=%.3f stab=%.3f | tau=%.3f"
                  % (*M.mean(0), np.mean(TM)))
    finally:
        FedHIFT.__call__ = orig


def heterogeneity(rounds: int = 20):
    """Two-stage estimate vs. a naive whole-cohort MAD, across alpha and attack."""
    orig = FedHIFT.__call__
    rows: dict = {}
    tag = ""

    def patched(self, U, sizes, st):
        out = orig(self, U, sizes, st)
        a = 2 * out[1]["u"][:, 0] - 1                  # back to raw cosine
        rows.setdefault(tag, []).append(
            (out[1]["hetero"], mad(a), self.engine.last_phi))
        return out

    FedHIFT.__call__ = patched
    try:
        for alpha in [0.05, 0.1, 0.3, 0.5, 1.0]:
            fd = load_federated("fmnist", 30, alpha, 0, 20000, 100)
            for atk in ["none", "sign_flip", "label_flip"]:
                tag = "a%.2f_%s" % (alpha, atk)
                rows[tag] = []
                cfg = FLConfig(rounds=rounds, eval_every=rounds, aggregator="fedhift",
                               attack=atk, byz_frac=0.2, alpha=alpha,
                               train_subsample=20000)
                r = run(cfg, fd)
                v = np.array(rows[tag])
                print("alpha=%-5g %-11s acc=%.3f  two-stage=%.3f  naive MAD=%.3f  phi=%.2f"
                      % (alpha, atk, r["acc_last5"], v[:, 0].mean(), v[:, 1].mean(),
                         v[:, 2].mean()), flush=True)
    finally:
        FedHIFT.__call__ = orig


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("antecedents", "both"):
        antecedents()
    if which in ("heterogeneity", "both"):
        heterogeneity()
