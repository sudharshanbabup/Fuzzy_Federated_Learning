"""Unit tests for the FedHIFT components.

Run with:  python -m pytest tests -q      (or)   python tests/test_units.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedhift.aggregators import (FedHIFT, agg_fedavg, agg_median, agg_multikrum,
                                 agg_rfa, agg_trimmed_mean, cosine_matrix,
                                 geometric_median, mad, sketch, sketch_indices)
from fedhift.attacks import apply_model_poison, poison_labels
from fedhift.fuzzy import (FuzzyTrustEngine, IT2Config, ekm, entropic_weights,
                           fou_factor, membership_bounds)
from fedhift.metrics import auc_binary, jain_index, worst_frac


# --------------------------------------------------------------------- fuzzy
def test_membership_bounds_ordering():
    """The lower MF must never exceed the upper MF anywhere in the FOU."""
    u = np.random.default_rng(0).random((50, 4))
    for phi in (1.0, 1.3, 2.0):
        lo, up = membership_bounds(u, phi, IT2Config())
        assert np.all(lo <= up + 1e-12)
    lo1, up1 = membership_bounds(u, 1.0, IT2Config())
    assert np.allclose(lo1, up1), "phi = 1 must collapse the FOU to a type-1 set"


def test_fou_monotone():
    cfg = IT2Config()
    v = [fou_factor(h, cfg) for h in (0.0, 0.05, 0.15, 0.4, 5.0)]
    assert v[0] == 1.0
    assert all(a <= b + 1e-12 for a, b in zip(v, v[1:]))
    assert v[-1] <= cfg.phi_max + 1e-12


def test_ekm_matches_brute_force():
    """EKM must reproduce an exhaustive switch-point search."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        R = rng.integers(4, 12)
        y = np.sort(rng.random(R))
        f_lo = rng.random(R) * 0.5
        f_hi = f_lo + rng.random(R) * 0.5
        best_l, best_r = np.inf, -np.inf
        for k in range(R + 1):                       # exhaustive switch point
            fl = np.concatenate([f_hi[:k], f_lo[k:]])
            fr = np.concatenate([f_lo[:k], f_hi[k:]])
            best_l = min(best_l, (fl * y).sum() / fl.sum())
            best_r = max(best_r, (fr * y).sum() / fr.sum())
        assert abs(ekm(y, f_lo, f_hi, "l") - best_l) < 1e-9
        assert abs(ekm(y, f_lo, f_hi, "r") - best_r) < 1e-9


def test_engine_monotone_in_evidence():
    """Trust must be non-decreasing in every antecedent (the rule base is monotone)."""
    eng = FuzzyTrustEngine()
    rng = np.random.default_rng(11)
    base = rng.random((40, 4)) * 0.8 + 0.1
    tau0, _ = eng(base, 0.02)
    for j in range(4):
        up = base.copy()
        up[:, j] = np.minimum(1.0, up[:, j] + 0.15)
        tau1, _ = eng(up, 0.02)
        assert np.mean(tau1 - tau0) > 0, f"variable {j} is not evidence-monotone"


def test_fou_regularises_towards_neutral():
    """Widening the FOU must shrink the spread of the inferred trust values."""
    eng = FuzzyTrustEngine()
    u = np.array([[0.9, 0.9, 0.9, 0.9], [0.15, 0.2, 0.15, 0.2]])
    spreads = []
    for h in (0.0, 0.1, 0.3):
        tau, span = eng(u, h)
        spreads.append(tau[0] - tau[1])
    assert spreads[0] > spreads[1] > spreads[2] > 0


def test_type1_is_special_case():
    eng2 = FuzzyTrustEngine(type1=True)
    u = np.random.default_rng(5).random((10, 4))
    tau, span = eng2(u, 0.4)
    assert np.all(span == 0) and np.all((tau >= 0) & (tau <= 1))


# ------------------------------------------------------------ weight allocation
def test_entropic_weights_limits():
    tau = np.array([0.1, 0.5, 0.9])
    prior = np.array([0.2, 0.3, 0.5])
    w_hi = entropic_weights(tau, prior, 1e6)
    assert np.allclose(w_hi, prior, atol=1e-4), "T -> inf must recover the prior"
    w_lo = entropic_weights(tau, prior, 1e-3)
    assert w_lo.argmax() == tau.argmax() and w_lo.max() > 0.999


def test_entropic_weights_optimality():
    """The closed form must beat random simplex points on the KL-regularised objective."""
    rng = np.random.default_rng(2)
    tau = rng.random(6)
    prior = rng.dirichlet(np.ones(6))
    T = 0.3
    w = entropic_weights(tau, prior, T)

    def obj(v):
        v = np.clip(v, 1e-12, None)
        return v @ tau - T * np.sum(v * np.log(v / prior))

    best = obj(w)
    for _ in range(2000):
        assert obj(rng.dirichlet(np.ones(6))) <= best + 1e-9
    assert abs(w.sum() - 1) < 1e-12


def test_fairness_ratio_bound():
    """Proposition 2: relative weight distortion is bounded by exp(spread / T)."""
    rng = np.random.default_rng(9)
    for _ in range(200):
        T = float(rng.uniform(0.05, 1.0))
        tau = rng.random(8)
        prior = rng.dirichlet(np.ones(8))
        w = entropic_weights(tau, prior, T)
        r = w / prior
        assert r.max() / r.min() <= np.exp((tau.max() - tau.min()) / T) * (1 + 1e-8)


# --------------------------------------------------------------- aggregators
def test_geometric_median_recovers_centre():
    rng = torch.Generator().manual_seed(0)
    c = torch.randn(64, generator=rng)
    X = c + 0.01 * torch.randn(21, 64, generator=rng)
    X[:8] = 50 * torch.randn(8, 64, generator=rng)      # 8 / 21 gross outliers
    z = geometric_median(X, iters=60)
    assert torch.linalg.norm(z - c) < 0.5 * torch.linalg.norm(c)


def test_aggregators_are_convex_combinations():
    X = torch.randn(9, 40)
    s = np.full(9, 100.0)
    for fn in (agg_fedavg, agg_median, agg_trimmed_mean, agg_multikrum, agg_rfa):
        d, info = fn(X, s, {})
        assert d.shape == (40,)
        assert np.all(np.asarray(info["w"]) >= -1e-12)
        assert abs(np.asarray(info["w"]).sum() - 1) < 1e-6


def test_trimmed_mean_removes_extremes():
    X = torch.zeros(10, 5)
    X[0] = 100.0
    X[1] = -100.0
    d, _ = agg_trimmed_mean(X, np.ones(10), {}, beta=0.2)
    assert torch.allclose(d, torch.zeros(5), atol=1e-6)


def test_sketch_preserves_cosine():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(12, 50000, generator=g)
    X[:9] += 0.6 * torch.randn(1, 50000, generator=g)
    exact = cosine_matrix(X)
    idx = sketch_indices(50000, 16384, g)
    approx = cosine_matrix(sketch(X, idx))
    assert np.abs(exact - approx).max() < 0.05


def test_mad_robust():
    v = np.array([1.0, 1.1, 0.9, 1.05, 900.0])
    assert mad(v) < 0.5


# --------------------------------------------------------------------- attacks
def test_label_flip_is_involution():
    y = torch.arange(10)
    f = poison_labels(y, "label_flip", 10, torch.Generator())
    assert torch.equal(poison_labels(f, "label_flip", 10, torch.Generator()), y)


def test_model_poison_only_touches_malicious():
    g = torch.Generator().manual_seed(0)
    U = {k: torch.randn(200, generator=g) for k in range(8)}
    for atk in ("sign_flip", "gauss", "scaling", "alie", "ipm"):
        out = apply_model_poison(U, [6, 7], atk, torch.Generator().manual_seed(1))
        for k in range(6):
            assert torch.equal(out[k], U[k]), f"{atk} modified an honest update"
        assert not torch.equal(out[6], U[6])


def test_sign_flip_reverses_direction():
    g = torch.Generator().manual_seed(0)
    U = {k: torch.randn(200, generator=g) for k in range(6)}
    out = apply_model_poison(U, [5], "sign_flip", g)
    cs = torch.nn.functional.cosine_similarity(out[5], U[5], dim=0)
    assert cs < -0.999


# --------------------------------------------------------------------- metrics
def test_jain_index_bounds():
    assert abs(jain_index([0.5] * 7) - 1.0) < 1e-12
    assert jain_index([1.0] + [0.0] * 9) - 0.1 < 1e-12
    assert 0 < jain_index([0.3, 0.6, 0.9]) < 1


def test_auc_matches_reference():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = 30
        y = rng.integers(0, 2, n)
        if y.sum() in (0, n):
            continue
        s = rng.random(n)
        pos, neg = s[y == 1], s[y == 0]
        brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
        assert abs(auc_binary(s, y) - brute) < 1e-9


def test_worst_frac():
    assert abs(worst_frac([0.1, 0.2, 0.3, 0.9], 0.25) - 0.1) < 1e-12


# ------------------------------------------------------- end-to-end aggregation
def test_fedhift_downweights_gross_outliers():
    torch.manual_seed(0)
    U = torch.randn(10, 4000) * 0.1
    U[:8] += torch.randn(1, 4000) * 0.3
    U[8:] = torch.randn(2, 4000) * 2.0
    st = {"client_ids": list(range(10))}
    d, info = FedHIFT(sketch_dim=0)(U, np.full(10, 100.0), st)
    w = info["w"]
    assert w[8:].sum() < 0.25 * w[:8].sum()
    assert abs(w.sum() - 1) < 1e-9
    assert torch.isfinite(d).all()


def test_fedhift_clipping_bounds_output():
    torch.manual_seed(1)
    U = torch.randn(9, 2000) * 0.05
    U[0] *= 1000.0                                   # magnitude bomb
    st = {"client_ids": list(range(9))}
    d, _ = FedHIFT(sketch_dim=0, nu=1.0)(U, np.full(9, 50.0), st)
    med = torch.linalg.norm(U, dim=1).median()
    assert torch.linalg.norm(d) <= med * 1.0 + 1e-5


def test_fedhift_is_permutation_equivariant():
    torch.manual_seed(2)
    U = torch.randn(8, 1500) * 0.1
    s = np.random.default_rng(0).uniform(50, 200, 8)
    a = FedHIFT(sketch_dim=0, seed=3)(U, s, {"client_ids": list(range(8))})[0]
    p = np.array([3, 1, 7, 0, 5, 2, 6, 4])
    b = FedHIFT(sketch_dim=0, seed=3)(U[p], s[p], {"client_ids": p.tolist()})[0]
    assert torch.allclose(a, b, atol=1e-5)


def test_fedhift_reduces_to_fedavg_at_high_temperature():
    torch.manual_seed(4)
    U = torch.randn(7, 1200) * 0.05
    s = np.random.default_rng(1).uniform(50, 300, 7)
    d, info = FedHIFT(sketch_dim=0, temperature=1e6, nu=1e9)(
        U, s, {"client_ids": list(range(7))})
    assert np.allclose(info["w"], s / s.sum(), atol=1e-4)
    assert torch.allclose(d, agg_fedavg(U, s, {})[0], atol=1e-5)


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
