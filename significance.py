"""Paired statistics behind every comparative claim in the paper.

Every aggregation rule sees the identical federation, client-sampling sequence
and adversarial set for a given seed, so two rules are compared cell by cell
over the matched (attack, seed) grid rather than on unpaired means. For each
comparison we report

  * the paired mean difference and a 95% confidence interval for it,
  * Cohen's d_z, the standardised paired effect size,
  * an exact two-sided p value from a sign-flip permutation test on the paired
    differences, which assumes only that the sign of a difference is
    exchangeable under the null and therefore does not rely on normality,
  * the number of cells won,
  * and the Holm-Bonferroni adjusted p value within each family of comparisons,
    a family being one metric on one data set against all baselines.

The permutation test is exact when n <= 20 (all 2^n sign assignments are
enumerated) and Monte Carlo with 200000 draws above that.

Usage:  python significance.py
"""
from __future__ import annotations

import glob
import itertools
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
COHERENT = ["scaling", "sign_flip"]           # adversary injects one direction
OTHER = ["none", "gauss", "alie", "label_flip", "ipm", "adaptive"]
EXACT_MAX = 20
RNG = np.random.default_rng(20260829)


# --------------------------------------------------------------- statistics --
def perm_p(d: np.ndarray) -> float:
    """Two-sided p of a sign-flip permutation test on paired differences."""
    n = len(d)
    obs = abs(d.mean())
    if n == 0:
        return float("nan")
    if n <= EXACT_MAX:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    else:
        # Beyond EXACT_MAX the enumeration is sampled. The generator is
        # re-seeded per call so that regenerating a table reproduces the same
        # p-value; a Monte-Carlo p that drifts between runs would make the
        # tables and the prose disagree.
        signs = np.random.default_rng(20260906).choice(
            [1.0, -1.0], size=(200000, n))
    means = np.abs(signs @ d) / n
    return float((means >= obs - 1e-12).mean())


def t_ci(d: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for the paired mean, from the t distribution."""
    n = len(d)
    if n < 2:
        return (float("nan"), float("nan"))
    se = d.std(ddof=1) / np.sqrt(n)
    return (d.mean() - _tcrit(n - 1, level) * se,
            d.mean() + _tcrit(n - 1, level) * se)


_TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
          7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
          13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
          19: 2.093, 20: 2.086, 24: 2.064, 29: 2.045, 39: 2.023, 59: 2.001}


def _tcrit(df: int, level: float) -> float:
    if df in _TCRIT:
        return _TCRIT[df]
    ks = sorted(_TCRIT)
    return _TCRIT[min(ks, key=lambda k: abs(k - df))]


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p values, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------- loading --
def _cells(suite, keyfn, metric="acc_last5", scale=100.0):
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(RES, suite, "*.json")):
        r = json.load(open(f))
        c = r["config"]
        if c["aggregator"].startswith("fedhift") and c.get("tag", "") not in ("", "full"):
            continue
        if c.get("type1"):
            continue
        if metric not in r:
            continue
        out[keyfn(c)][c["aggregator"]] = r[metric] * scale
    return out


def compare(cells, a, opponents, keep=None, label=""):
    """Print one family of comparisons with Holm-adjusted permutation p values."""
    rows = []
    for b in opponents:
        ks = [k for k in cells
              if a in cells[k] and b in cells[k]
              and (keep is None or (k[0] if isinstance(k, tuple) else k) in keep)]
        if len(ks) < 2:
            continue
        d = np.array([cells[k][a] - cells[k][b] for k in sorted(ks, key=str)])
        lo, hi = t_ci(d)
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("inf")
        rows.append([b, len(d), d.mean(), lo, hi, dz, perm_p(d), int((d > 0).sum())])
    if not rows:
        return
    adj = holm([r[6] for r in rows])
    print(f"\n{label}")
    print(f"  {'vs':<13}{'n':>3}{'diff':>9}{'95% CI':>18}{'d_z':>7}"
          f"{'p':>9}{'p_holm':>9}{'wins':>8}")
    for r, pa in zip(rows, adj):
        star = " *" if pa < 0.05 else ""
        print(f"  {r[0]:<13}{r[1]:>3}{r[2]:>+9.2f}"
              f"  [{r[3]:+6.2f},{r[4]:+6.2f}]{r[5]:>7.2f}"
              f"{r[6]:>9.4f}{pa:>9.4f}{f'{r[7]}/{r[1]}':>8}{star}")


# ----------------------------------------------------------------------- main --
def main():
    base = ["fedavg", "fedavg_clip", "median", "trimmed_mean", "multikrum",
            "rfa", "fltrust", "klcos"]
    suite = "main_v6" if glob.glob(os.path.join(RES, "main_v6", "*.json")) else "main_fmnist"

    for metric, scale, nm in [("acc_last5", 100.0, "accuracy (pp)"),
                              ("benign_jain", 1.0, "Jain over per-client accuracy"),
                              ("benign_jain_bacc", 1.0, "Jain over balanced accuracy"),
                              ("benign_worst10", 100.0, "worst-decile accuracy (pp)"),
                              ("benign_mean_f1", 1.0, "mean per-client macro F1")]:
        fm = _cells(suite, lambda c: (c["attack"], c["seed"]), metric, scale)
        fm = {k: v for k, v in fm.items() if "fedhift" in v}
        if not fm:
            continue
        compare(fm, "fedhift", base, None, f"### Fashion-MNIST, all conditions: {nm}")
        if metric == "acc_last5":
            compare(fm, "fedhift", base, COHERENT,
                    "### Fashion-MNIST, coherent-direction attacks: accuracy (pp)")
            compare(fm, "fedhift", base, OTHER,
                    "### Fashion-MNIST, all other conditions: accuracy (pp)")
            for atk in sorted({k[0] for k in fm}):
                compare(fm, "fedhift", ["rfa", "fedavg_clip", "klcos"], [atk],
                        f"### Fashion-MNIST, {atk}: accuracy (pp)")

    ni = _cells("noniid", lambda c: (c["alpha"], c["seed"]))
    ni = {k: v for k, v in ni.items() if "fedhift" in v}
    for al in sorted({k[0] for k in ni}):
        compare(ni, "fedhift", ["fedavg", "median", "multikrum", "rfa", "fltrust"],
                [al], f"### Dirichlet sweep, label flipping, alpha={al:g}: accuracy (pp)")

    cf = _cells("main_cifar", lambda c: (c["attack"], c["seed"]))
    for k, v in _cells("cifar_v6", lambda c: (c["attack"], c["seed"])).items():
        cf[k].update(v)
    cf = {k: v for k, v in cf.items() if "fedhift" in v}
    if cf:
        compare(cf, "fedhift", base, None, "### CIFAR-10, all conditions: accuracy (pp)")
        compare(cf, "fedhift", base, COHERENT,
                "### CIFAR-10, coherent-direction attacks: accuracy (pp)")

    # kappa: is any setting distinguishable from the default?
    ks = defaultdict(dict)
    for f in glob.glob(os.path.join(RES, "sensitivity", "*.json")):
        r = json.load(open(f))
        c = r["config"]
        if c["tag"].startswith("K"):
            ks[c["seed"]][c["kappa"]] = r["acc_last5"] * 100
    if ks:
        print("\n### FOU gain kappa against the default 1.2 (alpha=0.1, label flip)")
        for k in sorted({x for d in ks.values() for x in d}):
            if k == 1.2:
                continue
            d = np.array([ks[s][k] - ks[s][1.2] for s in ks if k in ks[s] and 1.2 in ks[s]])
            lo, hi = t_ci(d)
            print(f"  kappa={k:<5g} n={len(d)}  diff={d.mean():+6.2f}  "
                  f"[{lo:+6.2f},{hi:+6.2f}]  p={perm_p(d):.4f}")

    # interval type-2 against its type-1 reduction, keyed by configuration so
    # that the overlap between the two grids is not counted twice
    it2 = defaultdict(dict)
    for suite2 in ("fou", "fou_temp"):
        for f in glob.glob(os.path.join(RES, suite2, "*.json")):
            r = json.load(open(f))
            c = r["config"]
            it2[(c["alpha"], c["temperature"], c["seed"])][bool(c["type1"])] = \
                r["acc_last5"] * 100
    d = np.array([v[False] - v[True] for v in it2.values() if len(v) == 2])
    if len(d) > 1:
        lo, hi = t_ci(d)
        print(f"\n### interval type-2 minus type-1, distinct paired configurations\n"
              f"  n={len(d)}  diff={d.mean():+.3f}  [{lo:+.3f},{hi:+.3f}]  "
              f"d_z={d.mean()/d.std(ddof=1):+.3f}  p={perm_p(d):.4f}  "
              f"wins={(d > 0).sum()}/{len(d)}")

    print("\np is a two-sided sign-flip permutation test on the paired "
          "differences; p_holm is Holm-Bonferroni adjusted within the family "
          "printed in the same block. * marks p_holm < 0.05.")


if __name__ == "__main__":
    main()
