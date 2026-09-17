"""Server aggregation cost against model dimension.

The complexity claim of the paper is specific: the *additional* work that trust
estimation introduces is independent of the model dimension p, while the norm,
clipping and weighted-sum work is O(mp) exactly as it is for FedAvg. That
predicts two things about a log-log plot of aggregation time against p. FedEFT
should run parallel to FedAvg with a constant vertical offset that shrinks in
relative terms as p grows, and the order-statistic rules should have a strictly
larger cost at every p because they touch all p coordinates repeatedly.

This script measures that directly on synthetic cohorts, with no training in the
loop, so the timing is the aggregation call and nothing else. Times are the
median over repeats of the per-round cost at a cohort of m clients.

    python bench_scaling.py                 # the grid used in the paper
    python bench_scaling.py --quick         # a fast smoke run
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "results", "scaling.json")

import sys
sys.path.insert(0, HERE)
from fedhift.aggregators import (FedHIFT, agg_fedavg, agg_fedavg_clip,
                                 agg_median, agg_multikrum, agg_rfa,
                                 agg_trimmed_mean, sketch_indices)

RULES = ["fedavg", "fedavg_clip", "median", "trimmed_mean", "multikrum",
         "rfa", "fedhift"]


def _make(rule: str, p: int, d: int):
    """Return a callable with the aggregator signature."""
    if rule == "fedhift":
        return FedHIFT(temperature=0.2, nu=1.0, sketch_dim=min(d, p))
    return {"fedavg": agg_fedavg, "fedavg_clip": agg_fedavg_clip,
            "median": agg_median, "trimmed_mean": agg_trimmed_mean,
            "multikrum": agg_multikrum, "rfa": agg_rfa}[rule]


def time_rule(rule: str, p: int, m: int, d: int, repeats: int,
              seed: int = 0) -> float:
    """Median wall-clock seconds of one aggregation call at dimension p."""
    g = torch.Generator().manual_seed(seed)
    sizes = np.linspace(400.0, 1400.0, m)
    times = []
    agg = _make(rule, p, d)
    warmup = 2                                   # the engine allocates on its
    for r in range(repeats + warmup):            # first two calls
        U = torch.randn(m, p, generator=g, dtype=torch.float32)
        U /= torch.linalg.norm(U, dim=1, keepdim=True)
        st = {"client_ids": list(range(m)), "round": r + 4}
        t0 = time.perf_counter()
        agg(U, sizes, st)
        dt = time.perf_counter() - t0
        if r >= warmup:
            times.append(dt)
        del U
    gc.collect()
    return float(np.median(times))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=10, help="cohort size")
    ap.add_argument("--d", type=int, default=1 << 14, help="sketch dimension")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    torch.set_num_threads(1)
    # The sweep stops at 3e6. At 1e7 a cohort of ten float32 vectors is 400 MB
    # and the sorting rules hold several copies, so every rule becomes
    # memory-bandwidth bound and the ratios collapse towards one; that measures
    # the machine rather than the algorithms.
    ps = [10 ** 4, 10 ** 5] if a.quick else [10 ** 4, 3 * 10 ** 4, 10 ** 5,
                                             3 * 10 ** 5, 10 ** 6, 3 * 10 ** 6]
    repeats = 2 if a.quick else a.repeats

    rows = {}
    for rule in RULES:
        rows[rule] = []
        for p in ps:
            t = time_rule(rule, p, a.m, a.d, repeats)
            rows[rule].append(t * 1e3)           # milliseconds
            print(f"  {rule:<13} p={p:>9,}  {t * 1e3:8.2f} ms", flush=True)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump({"p": ps, "m": a.m, "sketch_dim": a.d,
                   "repeats": repeats, "ms": rows}, fh, indent=1)
    print("wrote", OUT_JSON)

    # the slope of log(time) against log(p) is the quantity the claim is about
    lp = np.log10(np.asarray(ps, dtype=float))
    print("\nlog-log slope over the top decade (1 = linear in p):")
    for rule in RULES:
        lt = np.log10(np.asarray(rows[rule]))
        k = np.polyfit(lp[-3:], lt[-3:], 1)[0]
        print(f"  {rule:<13} {k:5.2f}")


if __name__ == "__main__":
    main()
