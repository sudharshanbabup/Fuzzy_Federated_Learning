"""Paired significance tests behind every comparative claim in the paper.

Because every aggregation rule sees the identical federation, client-sampling
sequence and adversarial set for a given seed, two rules are compared cell by
cell rather than on unpaired means. This script prints the paired differences,
their t statistics and the win counts for each claim the paper makes.

Usage:  python significance.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
BASELINES = ["fedavg", "median", "trimmed_mean", "multikrum", "rfa", "fltrust"]
MAGNITUDE = ["scaling", "sign_flip"]          # attacks that inflate ||Delta||
STEALTH = ["none", "gauss", "alie"]           # benign or variance-bounded


def _cells(suite, keyfn):
    """dict[cell][aggregator] -> (acc, jain, worst10, W_B)."""
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(RES, suite, "*.json")):
        r = json.load(open(f))
        c = r["config"]
        if c["aggregator"] == "fedhift" and c.get("tag", "") not in ("", "full"):
            continue
        if c.get("type1"):
            continue
        out[keyfn(c)][c["aggregator"]] = (r["acc_last5"] * 100, r["benign_jain"],
                                          r["benign_worst10"] * 100,
                                          r["mal_weight_mass"])
    return out


def paired(cells, a, b, idx):
    ks = [k for k in cells if a in cells[k] and b in cells[k]]
    d = np.array([cells[k][a][idx] - cells[k][b][idx] for k in ks])
    if len(d) < 2:
        return len(d), d.mean() if len(d) else float("nan"), float("nan"), 0
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else float("inf")
    return len(d), d.mean(), t, int((d > 0).sum())


def block(title, cells, metric_idx, metric_name, opponents=("rfa",), subsets=None):
    print(f"\n### {title} [{metric_name}]")
    subsets = subsets or {"all": None}
    for sname, keep in subsets.items():
        sub = {k: v for k, v in cells.items()
               if keep is None or (k[0] if isinstance(k, tuple) else k) in keep}
        for opp in opponents:
            n, m, t, w = paired(sub, "fedhift", opp, metric_idx)
            flag = "" if abs(t) < 2.11 else ("  *significant*" if t == t else "")
            print(f"  {sname:<18s} vs {opp:<12s} n={n:2d}  d={m:+7.3f}  "
                  f"t={t:6.2f}  wins={w}/{n}{flag}")


def main():
    fm = _cells("main_fmnist", lambda c: (c["attack"], c["seed"]))
    ipm = _cells("ablation_ipm", lambda c: ("ipm", c["seed"]))
    fm.update(ipm)
    fm = {k: v for k, v in fm.items() if "fedhift" in v}

    subsets = {"all conditions": None,
               "magnitude attacks": MAGNITUDE,
               "benign / stealth": STEALTH,
               "label flip": ["label_flip"],
               "ipm": ["ipm"]}
    block("Fashion-MNIST accuracy", fm, 0, "accuracy pp", ("rfa", "multikrum", "fedavg"),
          subsets)
    block("Fashion-MNIST fairness", fm, 1, "Jain", ("rfa",), subsets)
    block("Fashion-MNIST worst decile", fm, 2, "worst-10% pp", ("rfa",), subsets)

    ni = _cells("noniid", lambda c: (c["alpha"], c["seed"]))
    ni = {k: v for k, v in ni.items() if "fedhift" in v}
    print("\n### non-IID sweep, label flipping")
    for al in sorted({k[0] for k in ni}):
        sub = {k: v for k, v in ni.items() if k[0] == al}
        for opp in ("rfa", "multikrum", "fedavg"):
            n, m, t, w = paired(sub, "fedhift", opp, 0)
            _, mj, _, _ = paired(sub, "fedhift", opp, 1)
            print(f"  alpha={al:<6g} vs {opp:<11s} n={n}  dacc={m:+6.2f}  "
                  f"wins={w}/{n}  dJain={mj:+.4f}")

    cf = _cells("main_cifar", lambda c: (c["attack"], c["seed"]))
    cf = {k: v for k, v in cf.items() if "fedhift" in v}
    if cf:
        block("CIFAR-10 accuracy", cf, 0, "accuracy pp",
              ("fltrust", "multikrum", "rfa", "fedavg"),
              {"all conditions": None, "magnitude attacks": MAGNITUDE})
        block("CIFAR-10 fairness", cf, 1, "Jain", ("fltrust", "rfa"))

    # kappa: is any setting distinguishable from any other?
    ks = defaultdict(dict)
    for f in glob.glob(os.path.join(RES, "sensitivity", "*.json")):
        r = json.load(open(f))
        c = r["config"]
        if c["tag"].startswith("K"):
            ks[c["seed"]][c["kappa"]] = r["acc_last5"] * 100
    if ks:
        vals = sorted({k for d in ks.values() for k in d})
        print("\n### FOU gain kappa, paired over seeds (alpha=0.1, label flip)")
        base = 1.1
        for k in vals:
            if k == base:
                continue
            d = np.array([ks[s][k] - ks[s][base] for s in ks if k in ks[s] and base in ks[s]])
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            print(f"  kappa={k:<5g} vs {base}:  d={d.mean():+6.2f}  t={t:5.2f}  n={len(d)}")

    # interval type-2 vs its type-1 reduction
    it2 = defaultdict(dict)
    for suite, kf in (("fou", lambda c: ("a", c["alpha"], c["seed"])),
                      ("fou_temp", lambda c: ("t", c["temperature"], c["alpha"], c["seed"]))):
        for f in glob.glob(os.path.join(RES, suite, "*.json")):
            r = json.load(open(f))
            c = r["config"]
            it2[kf(c)][bool(c["type1"])] = r["acc_last5"] * 100
    d = np.array([v[False] - v[True] for v in it2.values() if len(v) == 2])
    if len(d) > 1:
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        print(f"\n### interval type-2 minus type-1, all paired runs\n"
              f"  n={len(d)}  d={d.mean():+.3f}  sd={d.std(ddof=1):.3f}  "
              f"t={t:.2f}  wins={(d > 0).sum()}/{len(d)}")

    print("\n(|t| > 2.11 is the two-tailed 5% threshold at n = 18; the exact "
          "threshold depends on n.)")


if __name__ == "__main__":
    main()
