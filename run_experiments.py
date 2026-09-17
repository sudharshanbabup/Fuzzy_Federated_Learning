"""Experiment driver for the FedHIFT study.

Usage
-----
    python run_experiments.py --suite main_fmnist --workers 2
    python run_experiments.py --suite all --workers 2

Every run is written as a single JSON file under ``results/<suite>/``; the driver
skips runs whose output already exists, so a suite can be resumed after an
interruption.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results")

METHODS = ["fedavg", "median", "trimmed_mean", "multikrum", "rfa", "fltrust", "fedhift"]
ATTACKS_F = ["none", "label_flip", "sign_flip", "gauss", "scaling", "alie"]
ATTACKS_C = ["none", "label_flip", "sign_flip", "scaling", "alie"]

FM = dict(dataset="fmnist", num_clients=30, clients_per_round=10, rounds=30,
          local_epochs=1, batch_size=32, lr=0.05, momentum=0.9,
          train_subsample=20000, eval_every=3)
# CIFAR-10 needs a smaller step than Fashion-MNIST: at lr = 0.05 with momentum
# 0.9 the GroupNorm network collapses to a constant-logit state from which the
# shrunken updates produced by the order-statistic rules cannot recover.
CF = dict(dataset="cifar10", num_clients=30, clients_per_round=10, rounds=40,
          local_epochs=1, batch_size=32, lr=0.02, momentum=0.9,
          train_subsample=20000, eval_every=4)


def key(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def build_suite(name: str) -> list[dict]:
    jobs: list[dict] = []

    if name == "main_fmnist":
        for m, a, s in itertools.product(METHODS, ATTACKS_F, [0, 1, 2]):
            jobs.append({**FM, "aggregator": m, "attack": a, "seed": s,
                         "alpha": 0.5, "byz_frac": 0.2})
        # FedProx reference point (clean + label flip)
        for a, s in itertools.product(["none", "label_flip"], [0, 1, 2]):
            jobs.append({**FM, "aggregator": "fedprox", "attack": a, "seed": s,
                         "alpha": 0.5, "byz_frac": 0.2, "prox_mu": 0.01})

    elif name == "main_cifar":
        for m, a, s in itertools.product(METHODS, ATTACKS_C, [0, 1, 2]):
            jobs.append({**CF, "aggregator": m, "attack": a, "seed": s,
                         "alpha": 0.5, "byz_frac": 0.2})

    elif name == "noniid":
        # The two non-fuzzy controls are included here as well, because the
        # question the sweep has to answer is whether the low-concentration
        # advantage survives once clipping alone is available to the baseline.
        for m, al, s in itertools.product(
                ["fedavg", "fedavg_clip", "median", "multikrum", "rfa",
                 "fltrust", "klcos", "fedhift"],
                [0.05, 0.1, 0.3, 1.0], [0, 1, 2, 3, 4, 5]):
            jobs.append({**FM, "aggregator": m, "attack": "label_flip", "seed": s,
                         "alpha": al, "byz_frac": 0.2})

    elif name == "byzfrac":
        for m, b, s in itertools.product(
                ["fedavg", "median", "multikrum", "rfa", "fltrust", "fedhift"],
                [0.0, 0.1, 0.3, 0.4], [0]):
            jobs.append({**FM, "aggregator": m, "attack": "sign_flip", "seed": s,
                         "alpha": 0.5, "byz_frac": b})

    elif name == "ablation":
        variants = {
            "full": {},
            "type1": {"type1": True},
            "no_rep": {"beta_rep": 1.0},
            "no_prior": {"use_size_prior": False},
            "no_clip": {"nu": 1e9},
            "wo_align": {"criteria": ["peer", "norm", "stab"]},
            "wo_peer": {"criteria": ["align", "norm", "stab"]},
            "wo_norm": {"criteria": ["align", "peer", "stab"]},
            "wo_stab": {"criteria": ["align", "peer", "norm"]},
        }
        for (vn, kw), a, s in itertools.product(variants.items(),
                                                ["label_flip", "sign_flip"], [0, 1]):
            jobs.append({**FM, "aggregator": "fedhift", "attack": a, "seed": s,
                         "alpha": 0.5, "byz_frac": 0.2, "tag": vn, **kw})
        # ablation under strong heterogeneity, where the FOU should matter most
        for (vn, kw), s in itertools.product(
                [("full", {}), ("type1", {"type1": True})], [0, 1]):
            jobs.append({**FM, "aggregator": "fedhift", "attack": "label_flip", "seed": s,
                         "alpha": 0.05, "byz_frac": 0.2, "tag": vn + "_a005", **kw})

    elif name == "fou":
        # does the interval type-2 footprint pay off, and where?
        for tp, al, sd in itertools.product([False, True], [0.05, 0.1, 0.3, 0.5, 1.0],
                                            [0, 1, 2, 3]):
            jobs.append({**FM, "aggregator": "fedhift", "attack": "label_flip",
                         "seed": sd, "alpha": al, "byz_frac": 0.2,
                         "type1": tp, "tag": ("type1" if tp else "full") + "_fou"})

    elif name == "fou_temp":
        # The FOU rescales the effective trust margin delta_H / T, so its effect
        # should appear as a change in how sharply performance depends on T.
        for tp, T, al, sd in itertools.product([False, True], [0.05, 0.1, 0.2, 0.4],
                                               [0.05, 0.5], [0, 1, 2]):
            jobs.append({**FM, "aggregator": "fedhift", "attack": "label_flip",
                         "seed": sd, "alpha": al, "byz_frac": 0.2, "temperature": T,
                         "type1": tp, "tag": ("type1" if tp else "full") + "_T"})

    elif name == "ablation_ipm":
        variants = {
            "full": {}, "type1": {"type1": True}, "no_rep": {"beta_rep": 1.0},
            "no_prior": {"use_size_prior": False}, "no_clip": {"nu": 1e9},
            "wo_align": {"criteria": ["peer", "norm", "stab"]},
            "wo_peer": {"criteria": ["align", "norm", "stab"]},
            "wo_norm": {"criteria": ["align", "peer", "stab"]},
            "wo_stab": {"criteria": ["align", "peer", "norm"]},
        }
        for (vn, kw), sd in itertools.product(variants.items(), [0, 1]):
            jobs.append({**FM, "aggregator": "fedhift", "attack": "ipm", "seed": sd,
                         "alpha": 0.5, "byz_frac": 0.2, "tag": vn, **kw})
        for m, sd in itertools.product(METHODS, [0, 1]):
            jobs.append({**FM, "aggregator": m, "attack": "ipm", "seed": sd,
                         "alpha": 0.5, "byz_frac": 0.2})

    elif name == "main_v6":
        # The Fashion-MNIST study of the revision: nine aggregation rules, eight
        # attacks including the white-box adaptive one, five seeds. Ordered
        # seed-major so that an interrupted campaign still leaves a balanced
        # grid over whichever seeds completed.
        rules = ["fedavg", "fedavg_clip", "median", "trimmed_mean", "multikrum",
                 "rfa", "fltrust", "klcos", "fedhift"]
        atks = ["none", "label_flip", "sign_flip", "gauss", "scaling", "alie",
                "ipm", "adaptive"]
        for s in [0, 1, 2, 3, 4]:
            for m, a in itertools.product(rules, atks):
                jobs.append({**FM, "aggregator": m, "attack": a, "seed": s,
                             "alpha": 0.5, "byz_frac": 0.2})
            for a in ["none", "label_flip"]:
                jobs.append({**FM, "aggregator": "fedprox", "attack": a, "seed": s,
                             "alpha": 0.5, "byz_frac": 0.2, "prox_mu": 0.01})

    elif name == "adaptive_full":
        # The three adaptive variants the manuscript tabulates, run against
        # every rule rather than only against ours. rho = 0.9 is the
        # trust-aware adversary, which spends its budget on conforming to the
        # four statistics; rho = 0 is the clip-aware adversary, which sits
        # exactly at the median-norm ball and spends everything on damage;
        # rho = 0.7 is the combined attack and is already in main_v6.
        for s in [0, 1, 2]:
            for m in ["fedavg", "fedavg_clip", "median", "multikrum", "rfa",
                      "fltrust", "klcos", "fedhift"]:
                for rho in [0.9, 0.0]:
                    jobs.append({**FM, "aggregator": m, "attack": "adaptive",
                                 "seed": s, "alpha": 0.5, "byz_frac": 0.2,
                                 "attack_rho": rho, "tag": f"R{rho}"})

    elif name == "adaptive_rho":
        # How much damage a white-box adversary can do as a function of the
        # alignment budget it keeps. rho = 1 is a perfectly conforming update
        # that does nothing; rho = 0 spends the whole norm on the orthogonal
        # direction and is easy to see.
        for s in [0, 1, 2]:
            for m in ["fedavg", "rfa", "fedhift"]:
                for rho in [0.9, 0.7, 0.5, 0.3, 0.0]:
                    jobs.append({**FM, "aggregator": m, "attack": "adaptive",
                                 "seed": s, "alpha": 0.5, "byz_frac": 0.2,
                                 "attack_rho": rho, "tag": f"R{rho}"})

    elif name == "cifar_v6":
        rules = ["fedavg_clip", "klcos"]
        atks = ["none", "label_flip", "sign_flip", "scaling", "alie"]
        for s in [0, 1, 2]:
            for m, a in itertools.product(rules, atks):
                jobs.append({**CF, "aggregator": m, "attack": a, "seed": s,
                             "alpha": 0.5, "byz_frac": 0.2})


    elif name == "clip_a005":
        # Isolates the contribution of median-norm clipping at the most skewed
        # concentration, where the trust engine has almost stopped separating.
        for tag, kw in [("full", {}), ("noclip", {"nu": 1e9})]:
            for s in [0, 1, 2, 3]:
                jobs.append({**FM, "aggregator": "fedhift", "attack": "label_flip",
                             "seed": s, "alpha": 0.05, "byz_frac": 0.2,
                             "tag": tag, **kw})

    elif name == "sensitivity":
        for T in [0.05, 0.1, 0.2, 0.4, 0.8, 4.0]:
            for s in [0, 1]:
                jobs.append({**FM, "aggregator": "fedhift", "attack": "sign_flip",
                             "seed": s, "alpha": 0.5, "byz_frac": 0.2,
                             "temperature": T, "tag": f"T{T}"})
        for kp in [0.0, 0.5, 1.1, 1.2, 2.0]:
            for s in [0, 1, 2, 3]:
                jobs.append({**FM, "aggregator": "fedhift", "attack": "label_flip",
                             "seed": s, "alpha": 0.1, "byz_frac": 0.2,
                             "kappa": kp, "tag": f"K{kp}"})
        for sd in [1024, 4096, 16384, 0]:
            for s in [0, 1]:
                jobs.append({**FM, "aggregator": "fedhift", "attack": "sign_flip",
                             "seed": s, "alpha": 0.5, "byz_frac": 0.2,
                             "sketch_dim": sd, "tag": f"S{sd}"})
    else:
        raise ValueError(name)
    return jobs


def _one(args):
    suite, cfg = args
    import torch
    torch.set_num_threads(1)
    from fedhift.fl import FLConfig, run
    out_dir = os.path.join(RESULTS, suite)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, key(cfg) + ".json")
    if os.path.exists(path):
        return path, "cached", 0.0
    t0 = time.time()
    try:
        res = run(FLConfig(**cfg))
    except Exception:
        return path, "FAIL:" + traceback.format_exc(limit=3), time.time() - t0
    with open(path, "w") as f:
        json.dump(res, f)
    return path, "ok", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    suites = (["main_fmnist", "ablation", "ablation_ipm", "noniid", "byzfrac",
               "fou", "fou_temp", "sensitivity", "clip_a005", "main_cifar"]
              if args.suite == "all" else [args.suite])
    for suite in suites:
        jobs = build_suite(suite)
        print(f"[{suite}] {len(jobs)} runs", flush=True)
        t0 = time.time()
        with Pool(args.workers) as pool:
            for i, (path, status, dt) in enumerate(
                    pool.imap_unordered(_one, [(suite, j) for j in jobs]), 1):
                if status.startswith("FAIL"):
                    print(f"  !! {os.path.basename(path)} {status}", flush=True)
                elif status == "ok":
                    print(f"  [{i}/{len(jobs)}] {os.path.basename(path)} {dt:.0f}s "
                          f"elapsed {(time.time()-t0)/60:.1f}m", flush=True)
        print(f"[{suite}] done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
