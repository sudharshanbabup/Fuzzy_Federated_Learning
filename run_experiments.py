"""Experiment driver for the FedEFT study.

Usage
-----
    python run_experiments.py --suite main_v6 --workers 2
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


# Federated EMNIST: one client per writer (natural heterogeneity), 62 classes.
FE = dict(dataset="femnist", num_clients=100, clients_per_round=20, rounds=50,
          local_epochs=1, batch_size=32, lr=0.05, momentum=0.9,
          train_subsample=None, eval_every=5)

TRUST_FNS = ["tr_cosine", "tr_linear", "tr_linopt", "tr_logistic", "tr_mlp",
             "tr_type1", "fedhift"]
RULES12 = ["fedavg", "fedavg_clip", "median", "trimmed_mean", "multikrum", "rfa",
           "fltrust", "flame", "bulyan", "dnc", "klcos", "fedhift"]
ATK8 = ["none", "label_flip", "sign_flip", "gauss", "scaling", "alie", "ipm", "adaptive"]


def key(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]


LITE = {  # reduced subsets used when only the two-core instance was available
    "byzsweep": lambda c: c["seed"] == 0,
    "temp": lambda c: c["seed"] == 0,
    "ruleperturb": lambda c: c.get("rule_seed", 0) in (0, 1),
    "qskew": lambda c: c["seed"] == 0,
    "churn": lambda c: c["seed"] == 0,
}


def build_suite(name: str) -> list[dict]:
    if name.endswith("_lite"):
        base = name[:-5]
        return [c for c in build_suite(base) if LITE[base](c)]
    jobs: list[dict] = []

    if name == "main_cifar":
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
    # ------------------------------------------------------------------ #
    # FODM revision suites
    # ------------------------------------------------------------------ #
    elif name == "trustfn":
        # Same statistics, same reputation, same entropic allocation, same clip:
        # only the map u -> tau changes.
        for s in [0, 1, 2]:
            for m, a in itertools.product(TRUST_FNS, ["label_flip", "sign_flip", "scaling",
                                                      "alie", "ipm", "backdoor"]):
                jobs.append({**FM, "aggregator": m, "attack": a, "seed": s,
                             "alpha": 0.5, "byz_frac": 0.2})
            for m, a in itertools.product(TRUST_FNS, ["label_flip", "sign_flip"]):
                jobs.append({**FM, "aggregator": m, "attack": a, "seed": s,
                             "alpha": 0.1, "byz_frac": 0.2})

    elif name == "modern":
        # Modern robust baselines on the main Fashion-MNIST grid, with FedEFT
        # re-run under the same software stack for the paired tests.
        for s in [0, 1, 2]:
            for m, a in itertools.product(["flame", "bulyan", "dnc", "fedhift"], ATK8):
                jobs.append({**FM, "aggregator": m, "attack": a, "seed": s,
                             "alpha": 0.5, "byz_frac": 0.2})

    elif name == "backdoor":
        for s in [0, 1, 2]:
            for m, al in itertools.product(RULES12, [1.0, 0.5, 0.1]):
                jobs.append({**FM, "aggregator": m, "attack": "backdoor", "seed": s,
                             "alpha": al, "byz_frac": 0.2})
            for m in RULES12 if s < 2 else []:
                jobs.append({**FM, "aggregator": m, "attack": "backdoor_boost", "seed": s,
                             "alpha": 0.5, "byz_frac": 0.2})

    elif name == "femnist":
        for s in [0, 1]:
            for m, a in itertools.product(RULES12, ["none", "label_flip", "sign_flip",
                                                    "scaling", "backdoor"]):
                jobs.append({**FE, "aggregator": m, "attack": a, "seed": s, "byz_frac": 0.2})

    elif name == "byzsweep":
        for s in [0, 1]:
            for m, b in itertools.product(["fedavg_clip", "flame", "klcos", "fedhift"],
                                          [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]):
                jobs.append({**FM, "aggregator": m, "attack": "sign_flip", "seed": s,
                             "alpha": 0.5, "byz_frac": b, "keep_log": m == "fedhift"})

    elif name == "temp":
        for s in [0, 1]:
            for T in [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 4.0]:
                for a, al in [("sign_flip", 0.5), ("sign_flip", 0.1)]:
                    jobs.append({**FM, "aggregator": "fedhift", "attack": a, "seed": s,
                                 "alpha": al, "byz_frac": 0.2, "temperature": T,
                                 "keep_log": True})

    elif name == "theory":
        for a, al in itertools.product(["sign_flip", "label_flip", "scaling"],
                                       [0.05, 0.1, 0.3, 0.5, 1.0]):
            jobs.append({**FM, "aggregator": "fedhift", "attack": a, "seed": 0,
                         "alpha": al, "byz_frac": 0.2, "keep_log": True})

    elif name == "ruleperturb":
        for a in ["label_flip", "sign_flip", "scaling"]:
            jobs.append({**FM, "aggregator": "fedhift", "attack": a, "seed": 0,
                         "alpha": 0.5, "byz_frac": 0.2})
            for eps, rs in itertools.product([0.05, 0.10, 0.20], [0, 1, 2]):
                jobs.append({**FM, "aggregator": "fedhift", "attack": a, "seed": 0,
                             "alpha": 0.5, "byz_frac": 0.2, "rule_perturb": eps,
                             "rule_seed": rs})

    elif name == "longhorizon":
        LH = {**FM, "rounds": 200, "train_subsample": 10000, "eval_every": 10,
              "track_dir": True}
        for m in ["fedavg", "fedavg_clip", "rfa", "fedhift"]:
            jobs.append({**LH, "aggregator": m, "attack": "none", "seed": 0,
                         "alpha": 0.5, "byz_frac": 0.2})
            for rho in [0.7, 0.0]:
                jobs.append({**LH, "aggregator": m, "attack": "adaptive", "seed": 0,
                             "alpha": 0.5, "byz_frac": 0.2, "attack_rho": rho,
                             "keep_log": m == "fedhift"})

    elif name == "qskew":
        for s in [0, 1]:
            for m, sq, a in itertools.product(
                    ["fedavg", "fedavg_clip", "rfa", "flame", "fedhift"],
                    [0.5, 1.5], ["none", "sign_flip"]):
                jobs.append({**FM, "aggregator": m, "attack": a, "seed": s,
                             "partition": "quantity", "sigma_q": sq, "byz_frac": 0.2,
                             "keep_log": m == "fedhift"})

    elif name == "churn":
        # On-off adversary: honest for 10 rounds (building reputation), then
        # sign flipping, under Markov client availability.
        for s in [0, 1]:
            for dr in [0.0, 0.3, 0.6]:
                for m, extra in [("fedavg_clip", {}), ("flame", {}), ("klcos", {}),
                                 ("fedhift", {}), ("fedhift", {"beta_rep": 1.0, "tag": "norep"})]:
                    jobs.append({**FM, "aggregator": m, "attack": "sign_flip", "seed": s,
                                 "alpha": 0.5, "byz_frac": 0.2, "dropout": dr,
                                 "attack_start": 10, **extra})
    else:
        raise ValueError(name)
    return jobs


def _one(args):
    suite, cfg = args
    import torch
    torch.set_num_threads(1)
    from fedhift.fl import FLConfig, run
    out_dir = os.path.join(RESULTS, suite.replace("_lite", ""))
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
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job from i")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    si, sn = (int(v) for v in args.shard.split("/"))

    suites = (["main_v6", "modern", "main_cifar", "cifar_v6", "femnist", "trustfn",
               "backdoor", "noniid", "qskew", "churn", "theory", "temp", "byzsweep",
               "ablation", "ablation_ipm", "fou", "fou_temp", "sensitivity",
               "ruleperturb", "adaptive_rho", "adaptive_full", "longhorizon"]
              if args.suite == "all" else args.suite.split(","))
    for suite in suites:
        jobs = build_suite(suite)[si::sn]
        print(f"[{suite}] {len(jobs)} runs (shard {args.shard})", flush=True)
        if args.list:
            continue
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
