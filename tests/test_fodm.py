"""Unit tests for the components added in the FODM revision.

Run with:  python tests/test_fodm.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedhift.aggregators import (FedHIFT, agg_bulyan, agg_dnc, agg_flame,
                                 learned_trust)
from fedhift.data import add_trigger, quantity_partition
from fedhift.fl import _theory_record, perturbed_rules
from fedhift.fuzzy import FuzzyTrustEngine, IT2Config, ekm, entropic_weights


def _cohort(m=10, p=400, nbad=2, seed=0, scale=-5.0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(p, generator=g, dtype=torch.float64)
    U = base + 0.3 * torch.randn(m, p, generator=g, dtype=torch.float64)
    U[:nbad] = scale * base
    return U


def test_bulyan_excludes_outliers():
    # distinct (non-identical) outliers; identical colluders can pass Krum's
    # late, small-set rounds, which is a property of Bulyan, not a bug
    U = _cohort(m=11, nbad=2)
    U[:2] += 3.0 * torch.randn(2, U.shape[1], generator=torch.Generator().manual_seed(9), dtype=U.dtype)
    out, info = agg_bulyan(U, np.ones(11), {"n_byz_est": 2})
    assert info["w"][:2].sum() == 0, "Bulyan selected an obvious outlier"
    assert torch.linalg.norm(out - U[2:].mean(0)) < torch.linalg.norm(out - U[0])


def test_bulyan_respects_4f3():
    U = _cohort(m=10, nbad=2)
    _, info = agg_bulyan(U, np.ones(10), {"n_byz_est": 2})
    # f is capped at floor((10-3)/4) = 1, so theta = m - 2f = 8 clients are selected
    assert int((np.asarray(info["w"]) > 0).sum()) == 8


def test_dnc_removes_projection_outliers():
    U = _cohort(m=10, nbad=2, scale=-8.0)
    _, info = agg_dnc(U, np.ones(10), {"n_byz_est": 2, "seed": 0})
    assert np.asarray(info["w"])[:2].sum() == 0


def test_flame_admits_majority_and_clips():
    U = _cohort(m=10, nbad=2, scale=-20.0)
    out, info = agg_flame(U, np.ones(10), {"seed": 0}, lam=0.0)
    w = np.asarray(info["w"])
    assert w[:2].sum() == 0 and abs(w.sum() - 1) < 1e-12
    med = float(torch.linalg.norm(U, dim=1).median())
    assert float(torch.linalg.norm(out)) <= med + 1e-9, "clipped average exceeds the median norm"


def test_learned_trust_shapes_and_range():
    rng = np.random.default_rng(0)
    u = rng.random((7, 4))
    P = {"linopt": {"coef": [1, -1, 1, 0], "intercept": 0.2},
         "logistic": {"coef": [3, 0, 4, 0], "intercept": -3},
         "mlp": {"coefs": [rng.normal(size=(4, 16)).tolist(), rng.normal(size=(16, 16)).tolist(),
                           rng.normal(size=(16, 1)).tolist()],
                 "intercepts": [np.zeros(16).tolist(), np.zeros(16).tolist(), [0.0]]}}
    for k in ("linopt", "logistic", "mlp"):
        t = learned_trust(k, u, P)
        assert t.shape == (7,) and (t >= 0).all() and (t <= 1).all()


def test_calibrated_scorers_file_reproduces_sklearn():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calib", "scorers.json")
    if not os.path.exists(path):
        return
    P = json.load(open(path))
    for k in ("linopt", "logistic", "mlp"):
        assert k in P
    assert min(P["train_auc"].values()) > 0.9


def _brute_tr(y, flo, fhi, side):
    best = None
    for pick in itertools.product([0, 1], repeat=len(y)):
        f = np.where(np.array(pick) == 1, fhi, flo)
        v = (f * y).sum() / f.sum()
        best = v if best is None else (min(best, v) if side == "l" else max(best, v))
    return best


def test_type_reduction_exact_on_unsorted_perturbed_rules():
    """Per-side sorting keeps the prefix-sum reduction exact even when the two
    end-points are not jointly monotone (perturbed rule bases)."""
    rng = np.random.default_rng(3)
    for _ in range(30):
        R = 8
        y_lo = rng.random(R); y_hi = np.maximum(y_lo, rng.random(R))
        flo = rng.random(R) * 0.5; fhi = flo + rng.random(R) * 0.5
        ol, orr = np.argsort(y_lo), np.argsort(y_hi)
        yl = ekm(y_lo[ol], flo[ol], fhi[ol], "l")
        yr = ekm(y_hi[orr], flo[orr], fhi[orr], "r")
        assert abs(yl - _brute_tr(y_lo, flo, fhi, "l")) < 1e-10
        assert abs(yr - _brute_tr(y_hi, flo, fhi, "r")) < 1e-10


def test_perturbed_rules_valid_and_engine_runs():
    for eps in (0.05, 0.2):
        rules = perturbed_rules(eps, 1)
        assert all(0 <= lo <= hi <= 1 for _, (lo, hi) in rules)
        e = FuzzyTrustEngine(IT2Config(rules=rules))
        t, s = e(np.random.default_rng(0).random((5, 4)), 0.1)
        assert ((t >= 0) & (t <= 1)).all() and (s >= -1e-12).all()
    assert perturbed_rules(0.1, 2) != perturbed_rules(0.1, 3)


def test_type1_engine_monotone():
    e = FuzzyTrustEngine(type1=True)
    U = np.random.default_rng(1).random((4000, 4))
    for j in range(4):
        V = U.copy(); V[:, j] = np.minimum(1, V[:, j] + 0.05)
        assert (e(V, 0)[0] - e(U, 0)[0]).min() > -1e-12


def test_trigger_and_quantity_partition():
    x = torch.zeros(3, 1, 28, 28)
    x[:, :, 0, 0] = 2.0
    xt = add_trigger(x)
    assert torch.all(xt[:, :, 23:27, 23:27] == 2.0) and torch.all(x[:, :, 23:27, 23:27] == 0)
    parts = quantity_partition(5000, 30, 1.5, np.random.default_rng(0))
    allix = np.concatenate(parts)
    assert len(np.unique(allix)) == len(allix) and min(len(p) for p in parts) >= 20


def test_theory_record_bias_within_theorem2():
    torch.manual_seed(0)
    U = torch.randn(10, 64, dtype=torch.float64)
    U[:2] *= 30.0
    sizes = np.linspace(100, 300, 10)
    agg = FedHIFT(sketch_dim=0, seed=0)
    st = {"client_ids": list(range(10))}
    delta, info = agg(U, sizes, st)
    th = _theory_record(info, U, delta, [0, 0] + [1] * 8, 0.2)
    assert th["bias"] <= 2 * th["M"] * th["W_B"] + 1e-9
    assert abs(th["W_B"] - np.asarray(info["w"])[:2].sum()) < 1e-12


def test_entropic_weights_gibbs_value():
    """The optimal value equals T log sum pi e^{r/T} (Theorem 1)."""
    rng = np.random.default_rng(2)
    r = rng.random(6); pi = rng.random(6); pi /= pi.sum(); T = 0.3
    w = entropic_weights(r, pi, T)
    val = w @ r - T * np.sum(w * np.log(w / pi))
    assert abs(val - T * np.log(np.sum(pi * np.exp(r / T)))) < 1e-10


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fns:
        try:
            f()
            print(f"  PASS  {f.__name__}")
        except Exception as e:
            bad += 1
            print(f"  FAIL  {f.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
