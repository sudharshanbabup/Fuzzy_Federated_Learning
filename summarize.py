"""Print every number quoted in the paper, straight from the result JSONs."""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ORDER = ["fedavg", "fedprox", "median", "trimmed_mean", "multikrum", "rfa",
         "fltrust", "fedhift"]


def load(s):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, s, "*.json")))]


def tab(recs, kf, vf, fmt="%.1f"):
    d = defaultdict(list)
    for r in recs:
        d[kf(r)].append(vf(r))
    return {k: (np.mean(v), np.std(v), len(v)) for k, v in d.items()}


def show(title, recs, metric, scale=100.0, fmt="%6.2f"):
    print(f"\n### {title} [{metric}]")
    atks = sorted({r["config"]["attack"] for r in recs})
    t = tab(recs, lambda r: (r["config"]["aggregator"], r["config"]["attack"]),
            lambda r: r[metric] * scale)
    print("%-14s" % "" + "".join("%10s" % a[:9] for a in atks) + "%10s" % "MEAN")
    for m in ORDER:
        row = [t.get((m, a)) for a in atks]
        if all(x is None for x in row):
            continue
        cells = "".join("   ---   " if x is None else (fmt % x[0]).rjust(10) for x in row)
        mu = np.mean([x[0] for x in row if x is not None])
        print("%-14s%s%s" % (m, cells, (fmt % mu).rjust(10)))


def main():
    fm = [r for r in load("main_fmnist")]
    if fm:
        core = [r for r in fm if r["config"]["aggregator"] != "fedprox"]
        show("Fashion-MNIST accuracy", fm, "acc_last5")
        show("Fashion-MNIST Jain (benign)", core, "benign_jain", 1.0, "%6.4f")
        show("Fashion-MNIST worst-10%", core, "benign_worst10")
        show("Fashion-MNIST Byz weight mass", core, "mal_weight_mass", 1.0, "%6.4f")
        show("Fashion-MNIST detection AUROC", core, "det_auc", 1.0, "%6.3f")
        t = tab(core, lambda r: r["config"]["aggregator"],
                lambda r: 1000 * r["agg_time_s"] / r["config"]["rounds"])
        print("\n### server ms/round")
        for m in ORDER:
            if m in t:
                print("  %-14s %7.2f  (x%.1f vs FedAvg)" % (m, t[m][0], t[m][0] / t["fedavg"][0]))
        print("  params:", core[0]["num_params"])

    cf = load("main_cifar")
    if cf:
        show("CIFAR-10 accuracy", cf, "acc_last5")
        show("CIFAR-10 Jain (benign)", cf, "benign_jain", 1.0, "%6.4f")
        show("CIFAR-10 worst-10%", cf, "benign_worst10")

    ni = load("noniid")
    if ni:
        print("\n### non-IID sweep (label flip), accuracy by alpha")
        t = tab(ni, lambda r: (r["config"]["aggregator"], r["config"]["alpha"]),
                lambda r: r["acc_last5"] * 100)
        al = sorted({r["config"]["alpha"] for r in ni})
        print("%-14s" % "" + "".join("%9s" % a for a in al))
        for m in ORDER:
            row = [t.get((m, a)) for a in al]
            if all(x is None for x in row):
                continue
            print("%-14s%s" % (m, "".join("  ---  " if x is None else "%9.2f" % x[0] for x in row)))
        print("  Jain:")
        t2 = tab(ni, lambda r: (r["config"]["aggregator"], r["config"]["alpha"]),
                 lambda r: r["benign_jain"])
        for m in ORDER:
            row = [t2.get((m, a)) for a in al]
            if all(x is None for x in row):
                continue
            print("%-14s%s" % (m, "".join("  ---  " if x is None else "%9.4f" % x[0] for x in row)))

    bz = load("byzfrac")
    if bz:
        print("\n### Byzantine fraction sweep (sign flip)")
        t = tab(bz, lambda r: (r["config"]["aggregator"], r["config"]["byz_frac"]),
                lambda r: r["acc_last5"] * 100)
        bs = sorted({r["config"]["byz_frac"] for r in bz})
        print("%-14s" % "" + "".join("%9s" % b for b in bs))
        for m in ORDER:
            row = [t.get((m, b)) for b in bs]
            if all(x is None for x in row):
                continue
            print("%-14s%s" % (m, "".join("  ---  " if x is None else "%9.2f" % x[0] for x in row)))

    ab = load("ablation")
    if ab:
        print("\n### ablation (accuracy / Jain / Byz mass)")
        t = tab(ab, lambda r: (r["config"]["tag"], r["config"]["attack"]),
                lambda r: r["acc_last5"] * 100)
        tj = tab(ab, lambda r: (r["config"]["tag"], r["config"]["attack"]),
                 lambda r: r["benign_jain"])
        tw = tab(ab, lambda r: (r["config"]["tag"], r["config"]["attack"]),
                 lambda r: r["mal_weight_mass"])
        tags = sorted({r["config"]["tag"] for r in ab})
        for tg in tags:
            for a in sorted({k[1] for k in t if k[0] == tg}):
                if (tg, a) in t:
                    print("  %-12s %-11s acc=%6.2f jain=%.4f Wb=%.4f"
                          % (tg, a, t[(tg, a)][0], tj[(tg, a)][0], tw[(tg, a)][0]))

    se = load("sensitivity")
    if se:
        print("\n### sensitivity")
        for pre, field in (("T", "temperature"), ("K", "kappa"), ("S", "sketch_dim")):
            rr = [r for r in se if r["config"]["tag"].startswith(pre)]
            if not rr:
                continue
            t = tab(rr, lambda r: r["config"][field], lambda r: r["acc_last5"] * 100)
            tj = tab(rr, lambda r: r["config"][field], lambda r: r["benign_jain"])
            tw = tab(rr, lambda r: r["config"][field], lambda r: r["mal_weight_mass"])
            print(" ", field)
            for k in sorted(t):
                print("    %-10s acc=%6.2f jain=%.4f Wb=%.4f" % (k, t[k][0], tj[k][0], tw[k][0]))


if __name__ == "__main__":
    main()
