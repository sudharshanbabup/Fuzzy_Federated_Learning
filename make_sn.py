"""Every figure and table of the Springer Nature manuscript, from the run records.

    python make_sn.py            # everything
    python make_sn.py figures    # figures only
    python make_sn.py tables     # tables only

Figures are vector PDF authored at the exact width they are included at, so the
type reaches the page at the size set in snstyle.py. Line plots draw a smooth
shape-preserving interpolant through the measurements and put the markers and
error bars on the measurements themselves.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures_sn")
TAB = os.path.join(HERE, "tables_sn")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

sys.path.insert(0, HERE)
from snstyle import (COL, LEGEND, MED, NARROW, WIDE, apply_style, curve,
                     smooth_xy, style_for)
from significance import holm, perm_p, t_ci

apply_style()

PRETTY = {"fedavg": "FedAvg", "fedavg_clip": "FedAvg+clip", "fedprox": "FedProx",
          "median": "Coord. median", "trimmed_mean": "Trimmed mean",
          "multikrum": "Multi-Krum", "rfa": "RFA", "fltrust": "FLTrust",
          "klcos": "KL-cos", "fedhift": "FedEFT (ours)"}
SHORT = dict(PRETTY, fedhift="FedEFT")
ATK = {"none": "No attack", "label_flip": "Label flip", "sign_flip": "Sign flip",
       "gauss": "Gaussian", "scaling": "Scaling", "alie": "ALIE", "ipm": "IPM",
       "adaptive": "Adaptive"}
ORDER = ["fedavg", "fedavg_clip", "median", "trimmed_mean", "multikrum",
         "rfa", "fltrust", "klcos", "fedhift"]
# Colour, marker and dash are assigned together, so no series is distinguished
# by colour alone.
IDX = {k: i for i, k in enumerate(ORDER)}


def load(suite):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, suite, "*.json"))):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def agg(records, keyfn, valfn):
    d = defaultdict(list)
    for r in records:
        try:
            d[keyfn(r)].append(valfn(r))
        except (KeyError, TypeError):
            continue
    return {k: (float(np.mean(v)), float(np.std(v)), len(v))
            for k, v in d.items()}


def fmt(mu, sd=None, prec=1):
    if not np.isfinite(mu):
        return "---"
    if sd is None:
        return f"{mu:.{prec}f}"
    return f"{mu:.{prec}f}\\,$\\pm$\\,{sd:.{prec}f}"


def write(path, lines):
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  wrote", os.path.basename(path))


def holm_map(ps: dict) -> dict:
    ks = list(ps)
    return dict(zip(ks, holm([ps[k] for k in ks])))


# ------------------------------------------------------------------ figures --
def fig_hetero(fname="fig_hetero.pdf"):
    """Accuracy and fairness against the Dirichlet concentration."""
    recs = load("noniid")
    alphas = sorted({r["config"]["alpha"] for r in recs})
    rules = [m for m in ORDER if any(r["config"]["aggregator"] == m for r in recs)]
    acc = agg(recs, lambda r: (r["config"]["aggregator"], r["config"]["alpha"]),
              lambda r: r["acc_last5"] * 100)
    # Jain indices crowd against unity, so the right panel plots the
    # unfairness 1-J on a log axis, where the differences are legible.
    unf = agg(recs, lambda r: (r["config"]["aggregator"], r["config"]["alpha"]),
              lambda r: 1.0 - r["benign_jain"])

    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.55))
    for ax, tab, ylab, logy in [(axes[0], acc, "Accuracy (%)", False),
                                (axes[1], unf, r"Unfairness $1-J$", True)]:
        for m in rules:
            mu = [tab[(m, a)][0] for a in alphas]
            sd = [tab[(m, a)][1] for a in alphas]
            curve(ax, alphas, mu, sd if not logy else None, index=IDX[m],
                  label=PRETTY[m], logx=True, emphasis=(m == "fedhift"))
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(r"Dirichlet concentration $\alpha$")
        ax.set_ylabel(ylab)
        ax.grid(ls=":")
        ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="outside upper center", **LEGEND)
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)
    print("  wrote", fname)


def fig_fairness(fname="fig_fairness.pdf"):
    """Unfairness 1-J per attack, log axis, lower is better."""
    recs = [r for r in load("main_v6") if r["config"]["aggregator"] != "fedprox"]
    atks = ["label_flip", "sign_flip", "scaling", "alie"]
    m = agg(recs, lambda r: (r["config"]["aggregator"], r["config"]["attack"]),
            lambda r: 1.0 - r["benign_jain"])
    # FedAvg+clip and KL-cos sit within a bar width of FedEFT on every group,
    # because outcome fairness here is carried by the clipping step the three
    # share; plotting them adds two indistinguishable bars per group and no
    # information, so they are reported in Tables 3 and 4 instead.
    drop = {"fedavg_clip", "klcos"}
    rules = [x for x in ORDER if any(k[0] == x for k in m) and x not in drop]
    fig, ax = plt.subplots(figsize=(WIDE, 2.2))
    x = np.arange(len(atks))
    w = 0.82 / len(rules)
    hatches = ["", "///", "", "\\\\\\", "", "...", "", "xxx", ""]
    for i, mm in enumerate(rules):
        mu = [m.get((mm, a), (np.nan,) * 3)[0] for a in atks]
        ax.bar(x + (i - (len(rules) - 1) / 2) * w, mu, w,
               color=COL[IDX[mm] % len(COL)], hatch=hatches[i % len(hatches)],
               edgecolor="black", linewidth=0.35, label=PRETTY[mm])
    ax.set_xticks(x)
    ax.set_xticklabels([ATK[a] for a in atks])
    ax.set_ylabel(r"Unfairness $1-J$")
    ax.set_yscale("log")
    ax.grid(axis="y", ls=":")
    ax.set_axisbelow(True)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.30), **LEGEND)
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)
    print("  wrote", fname)


def fig_ablation(fname="fig_ablation.pdf"):
    """One component removed at a time: damage control against detection."""
    recs = load("ablation") + [r for r in load("ablation_ipm")
                               if r["config"]["aggregator"] == "fedhift"]
    recs = [r for r in recs if not r["config"].get("tag", "").endswith("_a005")]
    names = ["wo_align", "wo_peer", "wo_norm", "wo_stab",
             "type1", "no_rep", "no_prior", "no_clip"]
    lbl = {"wo_align": "no alignment", "wo_peer": "no peer agr.",
           "wo_norm": "no magnitude", "wo_stab": "no stability",
           "type1": "type-1 sets", "no_rep": "no reputation",
           "no_prior": "no size prior", "no_clip": "no norm clip"}
    atks = ["label_flip", "sign_flip", "ipm"]
    acc = agg(recs, lambda r: (r["config"]["tag"], r["config"]["attack"]),
              lambda r: r["acc_last5"] * 100)
    wb = agg(recs, lambda r: (r["config"]["tag"], r["config"]["attack"]),
             lambda r: r["mal_weight_mass"])

    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.75))
    y = np.arange(len(names))
    h = 0.78 / len(atks)
    face = [COL[1], COL[2], COL[3]]
    hatch = ["", "///", "..."]
    for i, a in enumerate(atks):
        kw = dict(color=face[i], hatch=hatch[i], edgecolor="black",
                  linewidth=0.35, label=ATK[a])
        b0 = acc.get(("full", a), (np.nan,) * 3)[0]
        axes[0].barh(y + (i - 1) * h,
                     [acc.get((n, a), (np.nan,) * 3)[0] - b0 for n in names],
                     h, **kw)
        b1 = wb.get(("full", a), (np.nan,) * 3)[0]
        axes[1].barh(y + (i - 1) * h,
                     [wb.get((n, a), (np.nan,) * 3)[0] - b1 for n in names],
                     h, **kw)
    for k, ax in enumerate(axes):
        ax.set_yticks(y)
        ax.set_yticklabels([lbl[n] for n in names] if k == 0 else [])
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.7)
        ax.grid(axis="x", ls=":")
        ax.set_axisbelow(True)
    axes[0].set_xlabel("change in accuracy (pp)")
    axes[1].set_xlabel(r"change in $W_{\mathcal{B}}$")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3,
               loc="outside upper center", **LEGEND)
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)
    print("  wrote", fname)


def fig_sensitivity(fname="fig_sensitivity.pdf"):
    """Accuracy and Byzantine weight mass against the three tuned parameters."""
    recs = load("sensitivity")
    panels = [("T", "temperature", r"temperature $T$", True),
              ("K", "kappa", r"footprint gain $\kappa$", False),
              ("S", "sketch_dim", r"sketch dimension $d$", True)]
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.05))
    for ax, (prefix, field, xlab, logx) in zip(axes, panels):
        sub = [r for r in recs
               if r["config"].get("tag", "").startswith(prefix)
               and r["config"].get("tag", "")[1:2].isdigit()]
        if not sub:
            continue
        accv = agg(sub, lambda r: r["config"][field],
                   lambda r: r["acc_last5"] * 100)
        wbv = agg(sub, lambda r: r["config"][field],
                  lambda r: r["mal_weight_mass"])
        xs = sorted(accv)
        # d = 0 denotes exact computation on the full p dimensions; place it at
        # the model dimension so the axis stays a dimension axis
        lab = {0: r"$p$"}
        xp = [215370 if (prefix == "S" and x == 0) else x for x in xs]
        curve(ax, xp, [accv[x][0] for x in xs], [accv[x][1] for x in xs],
              index=IDX["fedhift"], logx=logx, emphasis=True)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlab)
        ax.grid(ls=":")
        ax.set_axisbelow(True)
        ax2 = ax.twinx()
        gx, gy = smooth_xy(xp, [wbv[x][0] for x in xs], logx=logx)
        ax2.plot(gx, gy, color=COL[6], ls="--", lw=1.1)
        ax2.plot(xp, [wbv[x][0] for x in xs], color=COL[6], ls="none",
                 marker="s", ms=3.0)
        ax2.set_ylim(-0.008, 0.215)
        ax2.tick_params(axis="y", colors=COL[6])
        if ax is axes[-1]:
            ax2.set_ylabel(r"$W_{\mathcal{B}}$", color=COL[6])
        else:
            ax2.set_yticklabels([])
    axes[0].set_ylabel("Accuracy (%)")
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)
    print("  wrote", fname)


def fig_scaling(fname="fig_scaling.pdf"):
    """Server aggregation time against model dimension."""
    path = os.path.join(RES, "scaling.json")
    if not os.path.exists(path):
        print("  scaling.json missing; run bench_scaling.py")
        return
    d = json.load(open(path))
    ps = np.asarray(d["p"], dtype=float)
    fig, ax = plt.subplots(figsize=(NARROW, 2.35))
    rules = ["median", "trimmed_mean", "multikrum", "rfa", "fedhift"]
    base = np.asarray(d["ms"]["fedavg"], dtype=float)
    for m in rules:
        ms = np.asarray(d["ms"][m], dtype=float)
        curve(ax, ps, ms / base, index=IDX[m], label=PRETTY[m], logx=True,
              emphasis=(m == "fedhift"))
    ax.axhline(1.0, color="black", lw=0.7, ls=":")
    ax.text(ps[0], 1.06, "FedAvg", fontsize=7.5, va="bottom")
    ax.set_yscale("log")
    ax.set_ylabel("Aggregation cost relative to FedAvg")
    ax.set_xscale("log")
    ax.set_xlabel(r"Model dimension $p$")
    ax.grid(ls=":", which="major")
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), ncol=3,
               loc="outside upper center", **LEGEND)
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)
    print("  wrote", fname)


def fig_rho(fname="fig_rho.pdf"):
    """The adaptive attack as a function of its alignment budget."""
    recs = load("adaptive_rho") + load("adaptive_full")
    if not recs:
        print("  adaptive sweeps empty")
        return
    acc = agg(recs, lambda r: (r["config"]["aggregator"],
                               r["config"]["attack_rho"]),
              lambda r: r["acc_last5"] * 100)
    auc = agg(recs, lambda r: (r["config"]["aggregator"],
                               r["config"]["attack_rho"]),
              lambda r: r["det_auc"])
    wb = agg(recs, lambda r: (r["config"]["aggregator"],
                              r["config"]["attack_rho"]),
             lambda r: r["mal_weight_mass"])
    rhos = sorted({k[1] for k in acc})
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.25))
    for m in ["fedavg", "rfa", "fedhift"]:
        xs = [r for r in rhos if (m, r) in acc]
        curve(axes[0], xs, [acc[(m, r)][0] for r in xs],
              [acc[(m, r)][1] for r in xs], index=IDX[m], label=PRETTY[m],
              emphasis=(m == "fedhift"))
    axes[0].axhline(85.1, color="black", lw=0.6, ls=":")
    axes[0].set_ylim(78.0, 88.0)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="lower left", **LEGEND)
    xs = [r for r in rhos if ("fedhift", r) in auc]
    curve(axes[1], xs, [auc[("fedhift", r)][0] for r in xs],
          [auc[("fedhift", r)][1] for r in xs], index=0, label="AUROC")
    curve(axes[1], xs, [wb[("fedhift", r)][0] for r in xs],
          [wb[("fedhift", r)][1] for r in xs], index=4,
          label=r"$W_{\mathcal{B}}$")
    axes[1].axhline(0.5, color="black", lw=0.6, ls=":")
    axes[1].axhline(0.180, color="black", lw=0.6, ls="--")
    axes[1].set_ylim(0.0, 0.62)
    axes[1].set_ylabel("FedEFT detection")
    axes[1].legend(loc="lower left", **LEGEND)
    for ax in axes:
        ax.set_xlabel(r"alignment budget $\varrho$")
        ax.grid(ls=":")
        ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)
    print("  wrote", fname)


def fig_trust(fname="fig_trust.pdf"):
    """Trust separation and the online footprint, from instrumented runs."""
    import torch
    torch.set_num_threads(1)
    from fedhift.fl import FLConfig, run
    from fedhift.data import load_federated
    import fedhift.aggregators as A

    log, tag = {}, {"v": ""}
    oc = A.FedHIFT.__call__

    def call(self, U, sizes, st):
        out = oc(self, U, sizes, st)
        log.setdefault(tag["v"], []).append(
            (list(st["client_ids"]), np.asarray(out[1]["tau"]),
             float(self.engine.last_phi)))
        return out

    A.FedHIFT.__call__ = call
    mal_of = {}
    try:
        for alpha, TAG in [(0.5, "a05"), (0.05, "a005")]:
            tag["v"] = TAG
            log[TAG] = []
            fd = load_federated("fmnist", 30, alpha, 0, 20000, 100)
            cfg = FLConfig(rounds=30, eval_every=30, aggregator="fedhift",
                           attack="sign_flip", byz_frac=0.2, alpha=alpha,
                           train_subsample=20000)
            mal_of[TAG] = set(run(cfg, fd)["malicious"])
    finally:
        A.FedHIFT.__call__ = oc

    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 1.95))
    for ax, TAG, ttl in [(axes[0], "a05", r"$\alpha=0.5$"),
                         (axes[1], "a005", r"$\alpha=0.05$")]:
        tb, tm = defaultdict(list), defaultdict(list)
        for t, (cids, tau, _phi) in enumerate(log[TAG], 1):
            for k, v in zip(cids, tau):
                (tm if k in mal_of[TAG] else tb)[t].append(v)
        for d_, c, ls, lab in [(tb, COL[1], "-", "benign"),
                               (tm, COL[6], "--", "Byzantine")]:
            rr = sorted(d_)
            gx, gy = smooth_xy(rr, [np.mean(d_[t]) for t in rr])
            ax.plot(gx, gy, color=c, ls=ls, lw=1.3)
            ax.fill_between(rr, [np.percentile(d_[t], 10) for t in rr],
                            [np.percentile(d_[t], 90) for t in rr],
                            color=c, alpha=0.16, lw=0, label=lab)
        ax.set_ylim(0, 1)
        ax.set_title(ttl, pad=2.5)
        ax.set_ylabel(r"fuzzy trust $\tau_k$" if ax is axes[0] else "")
    axes[0].legend(loc="center right", **LEGEND)
    ax = axes[2]
    for TAG, alpha, c, ls in [("a05", 0.5, COL[1], "-"),
                              ("a005", 0.05, COL[6], "--")]:
        ph = [x[2] for x in log[TAG]]
        gx, gy = smooth_xy(range(1, len(ph) + 1), ph)
        ax.plot(gx, gy, color=c, ls=ls, lw=1.3, label=r"$\alpha=%g$" % alpha)
    ax.set_ylabel(r"footprint $\varphi^{t}$")
    ax.set_title("online footprint", pad=2.5)
    ax.legend(**LEGEND)
    for ax in axes:
        ax.set_xlabel("Round")
        ax.grid(ls=":")
        ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)
    print("  wrote", fname)


# ------------------------------------------------------------------- tables --
def _cells(recs, rules, atks, seeds, metric="acc_last5", scale=100.0):
    d = {}
    for r in recs:
        c = r["config"]
        if c["seed"] in seeds:
            d[(c["aggregator"], c["attack"], c["seed"])] = r[metric] * scale
    return d


def tab_main(recs, atks, seeds, caption, label, path, ncols=4):
    rules = [m for m in ORDER if any(r["config"]["aggregator"] == m for r in recs)]
    a = agg(recs, lambda r: (r["config"]["aggregator"], r["config"]["attack"]),
            lambda r: r["acc_last5"] * 100)
    mean = {m: float(np.mean([a[(m, x)][0] for x in atks if (m, x) in a]))
            for m in rules}
    groups = [atks[i:i + ncols] for i in range(0, len(atks), ncols)]
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{%s}" % caption,
             r"\label{%s}" % label, r"\setlength{\tabcolsep}{4.0pt}",
             r"\footnotesize"]
    # the last panel carries an extra Mean column, so every panel is padded to
    # the same width or the alignment breaks
    ncol_total = max(len(groups[-1]) + 2, max(len(g) for g in groups) + 1)
    lines += [r"\begin{tabular}{@{}l%s@{}}" % ("c" * (ncol_total - 1)), r"\toprule"]
    for gi, g in enumerate(groups):
        last = gi == len(groups) - 1
        head = ["Rule"] + [ATK[x] for x in g]
        if last:
            head += [r"\textbf{Mean}"]
        head += [""] * (ncol_total - len(head))
        lines.append(" & ".join(head) + r"\\")
        lines.append(r"\midrule")
        # ties are bolded together; picking an arbitrary argmax would present a
        # tie as a win
        best = {}
        for x in g:
            top = max(a.get((m, x), (-1e9,))[0] for m in rules)
            best[x] = {m for m in rules
                       if abs(a.get((m, x), (-1e9,))[0] - top) < 0.05}
        topmean = max(mean.values())
        bestmean = {m for m in rules if abs(mean[m] - topmean) < 0.05}
        for m in rules:
            if m == "fedhift":
                lines.append(r"\midrule")
            row = [PRETTY[m]]
            for x in g:
                v = a.get((m, x))
                t = fmt(v[0], v[1]) if v else "---"
                row.append(r"\textbf{%s}" % t if m in best[x] else t)
            if last:
                t = fmt(mean[m])
                row.append(r"\textbf{%s}" % t if m in bestmean else t)
            row += [""] * (ncol_total - len(row))
            lines.append(" & ".join(row) + r"\\")
        if not last:
            lines.append(r"\midrule")
            lines.append(r"\midrule")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    write(path, lines)


def tab_fairness(recs, path):
    rules = [m for m in ORDER if any(r["config"]["aggregator"] == m for r in recs)]
    atk = [x for x in ATK if x != "none"]
    sub = [r for r in recs if r["config"]["attack"] != "none"]
    cols = [("benign_jain", "Jain $J$", 3, 1.0),
            ("benign_worst10_bacc", r"Worst-10\%", 1, 100.0),
            ("benign_mean_bacc", "Mean bal.", 1, 100.0),
            ("benign_jain_f1", r"Jain $F_1$", 3, 1.0),
            ("mal_weight_mass", r"$W_{\mathcal{B}}$", 3, 1.0),
            ("det_auc", "AUROC", 3, 1.0)]
    tabs = {c[0]: agg(sub, lambda r: r["config"]["aggregator"],
                      lambda r, k=c[0]: r[k] * c[3]) for c in cols}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Outcome fairness and detection behaviour on "
             r"Fashion-MNIST, averaged over the seven attacks and five seeds. "
             r"Per-client scores are measured on test sets whose label mix "
             r"matches each client's own training data; the Jain indices, the "
             r"worst decile and the mean are over honest clients only, and the "
             r"balanced columns use per-class recall, which is invariant to how "
             r"easy each client's own label mix is. $W_{\mathcal{B}}$ is the "
             r"aggregate weight mass captured by Byzantine clients and AUROC "
             r"measures how well the assigned weights separate benign from "
             r"Byzantine participants. The reference value for $W_{\mathcal{B}}$ "
             r"is $0.180$, the mass the sample-size prior itself assigns to the "
             r"Byzantine clients in these cohorts, and it is what FedAvg, "
             r"FedAvg+clip and RFA report, since those three weight by the size "
             r"prior; the coordinate median and the trimmed mean report $0.195$, "
             r"the uniform $1/m$ mass, because they weight every client "
             r"equally. Neither exposes a per-client trust score, so their "
             r"AUROC is that of the weighting they use rather than of any "
             r"detector.}",
             r"\label{tab:fairness}", r"\setlength{\tabcolsep}{4.5pt}",
             r"\footnotesize",
             r"\begin{tabular}{@{}l%s@{}}" % ("c" * len(cols)), r"\toprule",
             "Rule & " + " & ".join(c[1] for c in cols) + r"\\", r"\midrule"]
    best = {}
    for k, _n, prec, _s in cols:
        lo = k in ("mal_weight_mass",)
        pool = {m: tabs[k][m][0] for m in rules if m in tabs[k]}
        top = (min if lo else max)(pool.values())
        tol = 0.5 * 10 ** (-prec)
        best[k] = {m for m, v in pool.items() if abs(v - top) < tol}
    for m in rules:
        if m == "fedhift":
            lines.append(r"\midrule")
        row = [PRETTY[m]]
        for k, _n, prec, _s in cols:
            v = tabs[k].get(m)
            t = fmt(v[0], None, prec) if v else "---"
            row.append(r"\textbf{%s}" % t if m in best[k] else t)
        lines.append(" & ".join(row) + r"\\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    write(path, lines)


def tab_fair_balanced(recs, path):
    """Matched-distribution fairness beside class-balanced fairness."""
    rules = [m for m in ORDER if any(r["config"]["aggregator"] == m for r in recs)]
    sub = [r for r in recs if r["config"]["attack"] != "none"]
    A_ = [("benign_mean", 1, 100.0), ("benign_worst10", 1, 100.0),
          ("benign_jain", 3, 1.0)]
    B_ = [("benign_mean_bacc", 1, 100.0), ("benign_worst10_bacc", 1, 100.0),
          ("benign_jain_bacc", 3, 1.0), ("benign_jain_f1", 3, 1.0)]
    tabs = {}
    for k, _p, s in A_ + B_:
        tabs[k] = agg(sub, lambda r: r["config"]["aggregator"],
                      lambda r, kk=k, ss=s: r[kk] * ss)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Outcome fairness measured two ways on Fashion-MNIST, "
             r"averaged over the seven attacks and five seeds. Metric~A grades "
             r"each client on a private test set whose label mix matches its "
             r"own training data, which measures local utility but also "
             r"inherits the differing difficulty of those label mixes. "
             r"Metric~B grades the same predictions by per-class recall, which "
             r"is invariant to the class proportions of the evaluation set, and "
             r"adds the macro $F_1$ Jain index. A fairness claim that survives "
             r"both is much harder to attribute to the choice of evaluation "
             r"distribution.}",
             r"\label{tab:fair_balanced}", r"\setlength{\tabcolsep}{3.6pt}",
             r"\footnotesize",
             r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
             r"& \multicolumn{3}{c}{Metric A: matched client distribution}"
             r"& \multicolumn{4}{c}{Metric B: class-balanced scoring}\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
             r"Rule & Mean & Worst-10\% & Jain & Mean & Worst-10\% & Jain "
             r"& Jain $F_1$\\", r"\midrule"]
    for m in rules:
        if m == "fedhift":
            lines.append(r"\midrule")
        row = [PRETTY[m]]
        for k, prec, _s in A_ + B_:
            v = tabs[k].get(m)
            row.append(fmt(v[0], None, prec) if v else "---")
        lines.append(" & ".join(row) + r"\\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    write(path, lines)


def tab_adaptive(path):
    """The three adaptive variants against every rule."""
    main = [r for r in load("main_v6") if r["config"]["attack"] == "adaptive"]
    full = load("adaptive_full") + load("adaptive_rho")
    best_non = agg([r for r in load("main_v6")
                    if r["config"]["attack"] in ("scaling", "sign_flip")],
                   lambda r: (r["config"]["aggregator"], r["config"]["attack"]),
                   lambda r: r["acc_last5"] * 100)
    rules = ["fedavg", "fedavg_clip", "median", "multikrum", "rfa", "fltrust",
             "klcos", "fedhift"]

    def cell(rule, rho, field="acc_last5", scale=100.0):
        if rho == 0.7:
            v = [r[field] * scale for r in main
                 if r["config"]["aggregator"] == rule]
        else:
            v = [r[field] * scale for r in full
                 if r["config"]["aggregator"] == rule
                 and abs(r["config"].get("attack_rho", 0.7) - rho) < 1e-9]
        return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)

    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Robustness to a white-box adaptive adversary on "
             r"Fashion-MNIST ($K=30$, $\alpha=0.5$, 20\% Byzantine). The "
             r"adversary reads the four published statistics and transmits "
             r"$\varrho\,\hat{\bm\mu}+\sqrt{1-\varrho^{2}}\,\hat{\mathbf b}$ at "
             r"the cohort median norm, so the alignment budget $\varrho$ selects "
             r"the variant. \emph{Trust-aware} ($\varrho=0.9$) spends almost "
             r"everything on conforming to the trust statistics; "
             r"\emph{clip-aware} ($\varrho=0$) sits exactly at the median-norm "
             r"ball and spends everything on the orthogonal damage direction; "
             r"\emph{combined} ($\varrho=0.7$) does both. The strongest "
             r"non-adaptive attack on each rule is reproduced for reference. "
             r"Accuracy in \%, mean $\pm$ s.d.; five seeds for the combined "
             r"column, three for the others.}",
             r"\label{tab:adaptive}", r"\setlength{\tabcolsep}{4.0pt}",
             r"\footnotesize",
             r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Rule & Best non-adaptive & Trust-aware & Clip-aware & Combined\\",
             r"\midrule"]
    for m in rules:
        if m == "fedhift":
            lines.append(r"\midrule")
        bn = min([best_non[(m, a)][0] for a in ("scaling", "sign_flip")
                  if (m, a) in best_non] or [np.nan])
        row = [PRETTY[m], fmt(bn)]
        for rho in (0.9, 0.0, 0.7):
            mu, sd = cell(m, rho)
            row.append(fmt(mu, sd))
        lines.append(" & ".join(row) + r"\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{@{}l@{}}{\emph{Diagnostics for FedEFT}}\\")
    for field, name, prec in [("mal_weight_mass",
                               r"Byzantine weight mass $W_{\mathcal{B}}$", 3),
                              ("det_auc", r"Detection AUROC", 3)]:
        row = [name, "---"]
        for rho in (0.9, 0.0, 0.7):
            mu, _ = cell("fedhift", rho, field, 1.0)
            row.append(fmt(mu, None, prec))
        lines.append(" & ".join(row) + r"\\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    write(path, lines)


def tab_cost(path):
    """Measured server cost, and the same cost at three model dimensions."""
    recs = [r for r in load("main_v6") if r["config"]["aggregator"] != "fedprox"]
    t = agg(recs, lambda r: r["config"]["aggregator"],
            lambda r: r["agg_time_s"] * 1000.0 / r["config"]["rounds"])
    rules = [m for m in ORDER if m in t]
    base = t["fedavg"][0]
    sc = None
    p_sc = os.path.join(RES, "scaling.json")
    if os.path.exists(p_sc):
        sc = json.load(open(p_sc))
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Server aggregation cost. The first two columns are "
             r"measured over every round of the $370$-run Fashion-MNIST grid at "
             r"$p=215{,}370$ and $m=10$. The remaining columns are the "
             r"synthetic dimension sweep of Fig.~\ref{fig:scaling}, which "
             r"isolates the aggregation call from training. FedEFT's relative "
             r"cost falls as $p$ grows because the trust statistics are "
             r"computed on a fixed $2^{14}$-dimensional sketch while the "
             r"order-statistic rules touch all $p$ coordinates. Below about "
             r"$10^{5}$ parameters every entry is dominated by fixed per-call "
             r"overhead rather than by the $O(mp)$ work, which is why the "
             r"ratios there are unstable and are not quoted in the text.}",
             r"\label{tab:cost}", r"\setlength{\tabcolsep}{4.5pt}",
             r"\footnotesize",
             r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Rule & ms/round & $\times$FedAvg & $\times$ at $10^{5}$ "
             r"& $\times$ at $10^{6}$ & $\times$ at $3\!\times\!10^{6}$\\",
             r"\midrule"]
    for m in rules:
        if m == "fedhift":
            lines.append(r"\midrule")
        row = [PRETTY[m], fmt(t[m][0]), fmt(t[m][0] / base, None, 2) + r"$\times$"]
        if sc and m in sc["ms"]:
            ps = sc["p"]
            f = np.asarray(sc["ms"][m]) / np.asarray(sc["ms"]["fedavg"])
            for target in (1e5, 1e6, 3e6):
                i = int(np.argmin([abs(p - target) for p in ps]))
                row.append(fmt(f[i], None, 2) + r"$\times$")
        else:
            row += ["---"] * 3
        lines.append(" & ".join(row) + r"\\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    write(path, lines)


def tab_stats(path):
    """Every paired comparison behind a claim in the text."""
    rows = []

    def block(name, cells, base_recs, rules, ref="fedhift", metric="acc_last5",
              scale=100.0):
        idx = {}
        for r in base_recs:
            c = r["config"]
            idx[(c["aggregator"],) + tuple(c[k] for k in cells)] = r[metric] * scale
        keys = sorted({k[1:] for k in idx if k[0] == ref})
        diffs, ps = {}, {}
        for m in rules:
            if m == ref:
                continue
            d = [idx[(ref,) + k] - idx[(m,) + k] for k in keys
                 if (m,) + k in idx and (ref,) + k in idx]
            if len(d) < 3:
                continue
            d = np.asarray(d)
            diffs[m] = d
            ps[m] = perm_p(d)
        adj = holm_map(ps)
        for m in diffs:
            d = diffs[m]
            lo, hi = t_ci(d)
            rows.append((name, PRETTY[m], len(d), d.mean(), lo, hi,
                         float(d.mean() / (d.std(ddof=1) + 1e-12)),
                         ps[m], adj[m], int((d > 0).sum())))

    fm = [r for r in load("main_v6") if r["config"]["aggregator"] != "fedprox"]
    rules = [m for m in ORDER]
    block(r"F-MNIST, all", ("attack", "seed"), fm, rules)
    block(r"F-MNIST, magnitude", ("attack", "seed"),
          [r for r in fm if r["config"]["attack"] in ("scaling", "sign_flip")],
          rules)
    cf = [r for r in load("main_cifar") + load("cifar_v6")
          if r["config"]["aggregator"] != "fedprox"]
    block(r"CIFAR-10, all", ("attack", "seed"), cf, rules)
    ni = load("noniid")
    block(r"Dirichlet sweep", ("alpha", "seed"), ni,
          [m for m in ORDER if any(r["config"]["aggregator"] == m for r in ni)])

    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Every paired comparison behind a claim in the text. "
             r"Each row compares FedEFT against one rule over matched cells, "
             r"since every rule sees the identical federation, sampling "
             r"sequence and adversarial set for a given seed. $n$ is the number "
             r"of paired cells, $\Delta$ the mean difference in accuracy points "
             r"with its $95\%$ interval, $p$ the "
             r"two-sided exact sign-flip permutation $p$-value and "
             r"$p_{\mathrm{Holm}}$ the same after Holm correction within the "
             r"block. The smallest attainable $p$ is $2^{1-n}$, so a block with "
             r"few cells cannot resolve at any effect size.}",
             r"\label{tab:stats}", r"\setlength{\tabcolsep}{2.6pt}",
             r"\scriptsize",
             r"\begin{tabular}{@{}llrrcrrr@{}}", r"\toprule",
             r"Setting & vs & $n$ & $\Delta$ & 95\% CI & $p$ "
             r"& $p_{\mathrm{Holm}}$ & wins\\", r"\midrule"]
    last = None
    for (name, m, n, dm, lo, hi, dz, p, ph, w) in rows:
        setting = name if name != last else ""
        if name != last and last is not None:
            lines.append(r"\midrule")
        last = name
        lines.append(f"{setting} & {SHORT[m] if m in SHORT else m} & {n} & "
                     f"{dm:+.2f} & $[{lo:+.1f},{hi:+.1f}]$ & "
                     f"{p:.3f} & {ph:.3f} & {w}/{n}" + r"\\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    write(path, lines)


def main(what="all"):
    if what in ("all", "figures"):
        print("figures:")
        fig_hetero()
        fig_fairness()
        fig_ablation()
        fig_sensitivity()
        fig_scaling()
        fig_rho()
        try:
            fig_trust()
        except Exception as e:                                # pragma: no cover
            print("  trust figure skipped:", e)
    if what in ("all", "tables"):
        print("tables:")
        fm = [r for r in load("main_v6") if r["config"]["aggregator"] != "fedprox"]
        seeds = sorted({r["config"]["seed"] for r in fm})
        atks = ["none", "label_flip", "sign_flip", "gauss", "scaling", "alie",
                "ipm", "adaptive"]
        tab_main(fm, atks, seeds,
                 r"Global test accuracy (\%, mean $\pm$ s.d.\ over five seeds) "
                 r"on Fashion-MNIST with $K=30$ clients, Dirichlet "
                 r"$\alpha=0.5$ and 20\% Byzantine participants, averaged over "
                 r"the last five evaluation points. FedAvg+clip is the "
                 r"sample-size prior carrying only the median-norm clip and "
                 r"KL-cos is the same entropic allocation driven by a plain "
                 r"cosine instead of the fuzzy engine; both are controls rather "
                 r"than baselines. The table is split into two panels for "
                 r"width and the mean is over all eight conditions. Best per "
                 r"column in bold.",
                 "tab:main_fmnist", os.path.join(TAB, "tab_main_fmnist.tex"))
        cf = [r for r in load("main_cifar") + load("cifar_v6")
              if r["config"]["aggregator"] != "fedprox"]
        tab_main(cf, ["none", "label_flip", "sign_flip", "scaling", "alie"],
                 sorted({r["config"]["seed"] for r in cf}),
                 r"Global test accuracy (\%, mean $\pm$ s.d.\ over three seeds) "
                 r"on CIFAR-10 under the same federation ($K=30$, "
                 r"$\alpha=0.5$, 20\% Byzantine participants, 40 communication "
                 r"rounds). Both non-fuzzy controls are included. Best per "
                 r"column in bold.",
                 "tab:main_cifar", os.path.join(TAB, "tab_main_cifar.tex"),
                 ncols=5)
        tab_fairness(fm, os.path.join(TAB, "tab_fairness.tex"))
        tab_fair_balanced(fm, os.path.join(TAB, "tab_fair_balanced.tex"))
        tab_adaptive(os.path.join(TAB, "tab_adaptive.tex"))
        tab_cost(os.path.join(TAB, "tab_cost.tex"))
        tab_stats(os.path.join(TAB, "tab_stats.tex"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
