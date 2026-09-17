"""Build every figure and LaTeX table in the paper from the result JSONs."""
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
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
    "axes.linewidth": 0.7, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7.2, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
    "lines.linewidth": 1.25, "lines.markersize": 3.4,
    "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "grid.linewidth": 0.4, "grid.alpha": 0.35,
})

PRETTY = {"fedavg": "FedAvg", "fedprox": "FedProx", "median": "Coord. Median",
          "trimmed_mean": "Trimmed Mean", "multikrum": "Multi-Krum", "rfa": "RFA",
          "fltrust": "FLTrust", "fedhift": "FedEFT (ours)"}
ATK = {"none": "No attack", "label_flip": "Label flip", "sign_flip": "Sign flip",
       "gauss": "Gaussian", "scaling": "Scaling", "alie": "ALIE", "ipm": "IPM"}
ORDER = ["fedavg", "median", "trimmed_mean", "multikrum", "rfa", "fltrust", "fedhift"]
COL = {"fedavg": "#8c8c8c", "median": "#4c72b0", "trimmed_mean": "#55a868",
       "multikrum": "#c44e52", "rfa": "#8172b3", "fltrust": "#ccb974",
       "fedhift": "#d1341a", "fedprox": "#937860"}
MK = {"fedavg": "o", "median": "s", "trimmed_mean": "^", "multikrum": "v",
      "rfa": "D", "fltrust": "P", "fedhift": "*"}


def load(suite):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, suite, "*.json"))):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def agg(records, keyfn, valfn):
    d = defaultdict(list)
    for r in records:
        d[keyfn(r)].append(valfn(r))
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in d.items()}


# --------------------------------------------------------------------------- #
def fig_robustness_bars(recs, dataset, attacks, fname):
    m = agg(recs, lambda r: (r["config"]["aggregator"], r["config"]["attack"]),
            lambda r: r["acc_last5"] * 100)
    meth = [x for x in ORDER if any(k[0] == x for k in m)]
    fig, ax = plt.subplots(figsize=(7.1, 1.95))
    n = len(meth)
    x = np.arange(len(attacks))
    w = 0.8 / n
    for i, mm in enumerate(meth):
        mu = [m.get((mm, a), (np.nan,) * 3)[0] for a in attacks]
        sd = [m.get((mm, a), (np.nan,) * 3)[1] for a in attacks]
        ax.bar(x + (i - (n - 1) / 2) * w, mu, w, yerr=sd, capsize=1.4,
               label=PRETTY[mm], color=COL[mm],
               edgecolor="black" if mm == "fedhift" else "none",
               linewidth=0.6, error_kw=dict(lw=0.5))
    ax.set_xticks(x)
    ax.set_xticklabels([ATK[a] for a in attacks])
    ax.set_ylabel("Global test accuracy (\\%)" if False else "Global test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", ls=":")
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def fig_curves(recs, attacks, fname):
    fig, axes = plt.subplots(1, len(attacks), figsize=(7.1, 1.68), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, attacks):
        for mm in ORDER:
            rr = [r for r in recs if r["config"]["aggregator"] == mm
                  and r["config"]["attack"] == a]
            if not rr:
                continue
            rounds = rr[0]["history"]["round"]
            acc = np.mean([r["history"]["acc"] for r in rr], axis=0) * 100
            ax.plot(rounds, acc, color=COL[mm], marker=MK[mm], markevery=3,
                    label=PRETTY[mm], lw=1.7 if mm == "fedhift" else 1.0,
                    zorder=5 if mm == "fedhift" else 2)
        ax.set_title(ATK[a])
        ax.grid(ls=":")
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(rounds) + 1)
    axes[0].set_ylabel("Global test accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=7, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.13), columnspacing=1.1, handletextpad=0.4)
    fig.text(0.5, -0.04, "Communication round", ha="center")
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def fig_fairness(recs, fname):
    """Unfairness 1-J on a log axis (lower is better)."""
    fig, ax = plt.subplots(figsize=(3.45, 1.85))
    atks = ["label_flip", "sign_flip", "scaling", "alie"]
    m = agg(recs, lambda r: (r["config"]["aggregator"], r["config"]["attack"]),
            lambda r: 1.0 - r["benign_jain"])
    meth = [x for x in ORDER if any(k[0] == x for k in m)]
    x = np.arange(len(atks))
    w = 0.8 / len(meth)
    for i, mm in enumerate(meth):
        mu = [m.get((mm, a), (np.nan,) * 3)[0] for a in atks]
        ax.bar(x + (i - (len(meth) - 1) / 2) * w, mu, w, color=COL[mm],
               label=PRETTY[mm],
               edgecolor="black" if mm == "fedhift" else "none", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([ATK[a] for a in atks], fontsize=7)
    ax.set_ylabel(r"Unfairness $1-J$")
    ax.set_yscale("log")
    ax.set_ylim(2e-3, 2.0)
    ax.grid(axis="y", ls=":")
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, fontsize=5.6, loc="upper center",
              columnspacing=0.8, handletextpad=0.3, handlelength=1.1)
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def _sweep_panel(ax, recs, xkey, xlabel, logx=False, metric="acc_last5"):
    for mm in ORDER:
        rr = [r for r in recs if r["config"]["aggregator"] == mm]
        if not rr:
            continue
        m = agg(rr, lambda r: r["config"][xkey], lambda r: r[metric] * 100)
        xs = sorted(m)
        ax.errorbar(xs, [m[x][0] for x in xs], yerr=[m[x][1] for x in xs],
                    color=COL[mm], marker=MK[mm], label=PRETTY[mm], capsize=1.5,
                    lw=1.7 if mm == "fedhift" else 1.0, elinewidth=0.6)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(ls=":")
    ax.set_axisbelow(True)


def fig_sweeps(ni, bz, fname):
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.7), sharey=True)
    _sweep_panel(axes[0], ni, "alpha", r"concentration $\alpha$", logx=True)
    _sweep_panel(axes[1], bz, "byz_frac", "Byzantine fraction")
    axes[0].set_ylabel("Accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.90), fontsize=6.0, columnspacing=0.9,
               handletextpad=0.35, borderaxespad=0.0)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def fig_trust_dynamics(fname):
    """Trust trajectories and the FOU factor on a single instrumented run."""
    import torch
    sys.path.insert(0, HERE)
    torch.set_num_threads(1)
    from fedhift.fl import FLConfig, run
    from fedhift.data import load_federated
    import fedhift.aggregators as A

    log = {}
    oc = A.FedHIFT.__call__

    def call(self, U, sizes, st):
        out = oc(self, U, sizes, st)
        log.setdefault(TAG, []).append(
            (list(st["client_ids"]), np.asarray(out[1]["tau"]),
             float(out[1]["hetero"]), float(self.engine.last_phi),
             np.asarray(out[1]["w"])))
        return out

    A.FedHIFT.__call__ = call
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.55))
    for ax, (alpha, TAG) in zip(axes[:2], [(0.5, "a05"), (0.05, "a005")]):
        log[TAG] = []
        fd = load_federated("fmnist", 30, alpha, 0, 20000, 100)
        cfg = FLConfig(rounds=30, eval_every=30, aggregator="fedhift",
                       attack="sign_flip", byz_frac=0.2, alpha=alpha,
                       train_subsample=20000)
        r = run(cfg, fd)
        mal = set(r["malicious"])
        tb = defaultdict(list)
        tm = defaultdict(list)
        for t, (cids, tau, h, phi, w) in enumerate(log[TAG], 1):
            for k, v in zip(cids, tau):
                (tm if k in mal else tb)[t].append(v)
        rr = sorted(tb)
        ax.plot(rr, [np.mean(tb[t]) for t in rr], color="#1a6fd1", label="benign")
        ax.fill_between(rr, [np.percentile(tb[t], 10) for t in rr],
                        [np.percentile(tb[t], 90) for t in rr], color="#1a6fd1", alpha=0.18)
        rm = sorted(tm)
        ax.plot(rm, [np.mean(tm[t]) for t in rm], color="#d1341a", label="Byzantine")
        ax.fill_between(rm, [np.percentile(tm[t], 10) for t in rm],
                        [np.percentile(tm[t], 90) for t in rm], color="#d1341a", alpha=0.18)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Communication round")
        ax.set_title(r"$\alpha=%g$" % alpha)
        ax.grid(ls=":")
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"fuzzy trust $\tau_k$")
    axes[0].legend(frameon=False, loc="center right")

    ax = axes[2]
    for TAG, alpha, c in [("a05", 0.5, "#1a6fd1"), ("a005", 0.05, "#d1341a")]:
        ph = [x[3] for x in log[TAG]]
        ax.plot(range(1, len(ph) + 1), ph, color=c, label=r"$\alpha=%g$" % alpha)
    ax.set_xlabel("Communication round")
    ax.set_ylabel(r"FOU factor $\varphi^{(t)}$", labelpad=1.0)
    ax.grid(ls=":")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.4)
    fig.subplots_adjust(wspace=0.32)
    A.FedHIFT.__call__ = oc
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def fig_ablation(recs, fname):
    names = ["wo_align", "wo_peer", "wo_norm", "wo_stab",
             "type1", "no_rep", "no_prior", "no_clip"]
    lbl = {"wo_align": "w/o alignment", "wo_peer": "w/o peer agr.",
           "wo_norm": "w/o magnitude", "wo_stab": "w/o stability",
           "type1": "type-1 sets", "no_rep": "no reputation",
           "no_prior": "no size prior", "no_clip": "no norm clip"}
    atks = ["label_flip", "sign_flip", "ipm"]
    acc = agg(recs, lambda r: (r["config"]["tag"], r["config"]["attack"]),
              lambda r: r["acc_last5"] * 100)
    wb = agg(recs, lambda r: (r["config"]["tag"], r["config"]["attack"]),
             lambda r: r["mal_weight_mass"])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 1.85), sharey=True)
    y = np.arange(len(names))
    h = 0.8 / len(atks)
    for i, a in enumerate(atks):
        base = acc.get(("full", a), (np.nan,) * 3)[0]
        axes[0].barh(y + (i - 1) * h, [acc.get((n, a), (np.nan,) * 3)[0] - base
                                       for n in names], h, label=ATK[a])
        b2 = wb.get(("full", a), (np.nan,) * 3)[0]
        axes[1].barh(y + (i - 1) * h, [wb.get((n, a), (np.nan,) * 3)[0] - b2
                                       for n in names], h, label=ATK[a])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lbl[n] for n in names])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("accuracy change vs. full model (pp)")
    axes[1].set_xlabel(r"change in Byzantine weight mass $W_{\mathcal{B}}$")
    for ax in axes:
        ax.axvline(0, color="k", lw=0.7)
        ax.grid(axis="x", ls=":")
        ax.set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=6.8, loc="lower right")
    fig.subplots_adjust(wspace=0.06)
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def fig_fou(recs, recs_T, fname):
    """Interval type-2 engine vs. its type-1 reduction: a controlled null result."""
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=False)
    ax = axes[0]
    a = agg(recs, lambda r: (r["config"]["type1"], r["config"]["alpha"]),
            lambda r: r["acc_last5"] * 100)
    als = sorted({r["config"]["alpha"] for r in recs})
    for tp, col, lab, mk in [(False, "#d1341a", "interval type-2 (default)", "o"),
                             (True, "#1a6fd1", "type-1 reduction", "s")]:
        mu = [a[(tp, x)][0] for x in als]
        sd = [a[(tp, x)][1] for x in als]
        ax.errorbar(als, mu, yerr=sd, color=col, marker=mk, label=lab,
                    capsize=2, elinewidth=0.7, lw=1.3)
    ax.set_xscale("log")
    ax.set_xlabel(r"Dirichlet concentration $\alpha$")
    ax.set_ylabel("Global test accuracy (%)")
    ax.grid(ls=":")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=5.6, loc="lower right", handlelength=1.4)
    ax.set_title(r"$T=0.2$", fontsize=7.5)

    ax = axes[1]
    a = agg(recs_T, lambda r: (r["config"]["type1"], r["config"]["alpha"],
                               r["config"]["temperature"]),
            lambda r: r["acc_last5"] * 100)
    Ts = sorted({r["config"]["temperature"] for r in recs_T})
    for al, ls in [(0.05, "-"), (0.5, "--")]:
        for tp, col in [(False, "#d1341a"), (True, "#1a6fd1")]:
            mu = [a[(tp, al, t)][0] for t in Ts]
            sd = [a[(tp, al, t)][1] for t in Ts]
            ax.errorbar(Ts, mu, yerr=sd, color=col, ls=ls, marker="o" if not tp else "s",
                        capsize=2, elinewidth=0.7, lw=1.3,
                        label=r"%s, $\alpha=%g$" % ("IT2" if not tp else "type-1", al))
    ax.set_xscale("log")
    ax.set_xticks(Ts)
    ax.set_xticklabels([str(t) for t in Ts])
    ax.minorticks_off()
    ax.set_xlabel(r"entropic temperature $T$")
    ax.set_ylabel("Global test accuracy (%)")
    ax.grid(ls=":")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=5.0, ncol=1, loc="lower right",
              handlelength=1.4, labelspacing=0.25)
    ax.set_title(r"$\alpha\in\{0.05,0.5\}$", fontsize=7.5)
    fig.subplots_adjust(wspace=0.42)
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def fig_sensitivity(recs, fname):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.68))
    grp = defaultdict(list)
    for r in recs:
        grp[r["config"]["tag"][0]].append(r)

    def panel(ax, tagch, field, xlabel, logx=True, last=False, xticks=None):
        rr = grp.get(tagch, [])
        m = agg(rr, lambda r: r["config"][field], lambda r: r["acc_last5"] * 100)
        f = agg(rr, lambda r: r["config"][field], lambda r: r["benign_jain"])
        xs = sorted(m)
        ax.errorbar(xs, [m[x][0] for x in xs], yerr=[m[x][1] for x in xs],
                    color="#d1341a", marker="o", capsize=1.5, elinewidth=0.6)
        if logx:
            ax.set_xscale("log")
        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(t) for t in xticks])
            ax.minorticks_off()
        ax.set_xlabel(xlabel)
        if tagch == "T":
            ax.set_ylabel("Accuracy (%)", color="#d1341a")
        ax.tick_params(axis="y", colors="#d1341a", labelsize=6.6)
        ax.grid(ls=":")
        ax.set_axisbelow(True)
        ax2 = ax.twinx()
        ax2.plot(xs, [f[x][0] for x in xs], color="#1a6fd1", marker="s", ls="--", lw=1.0)
        ax2.tick_params(axis="y", colors="#1a6fd1", labelsize=6.4)

    panel(axes[0], "T", "temperature", r"temperature $T$",
          xticks=[0.05, 0.2, 0.8, 4.0])
    panel(axes[1], "K", "kappa", r"FOU gain $\kappa$", logx=False)
    rr = grp.get("S", [])
    m = agg(rr, lambda r: (r["config"]["sketch_dim"] or 10 ** 6),
            lambda r: r["acc_last5"] * 100)
    f = agg(rr, lambda r: (r["config"]["sketch_dim"] or 10 ** 6),
            lambda r: r["benign_jain"])
    xs = sorted(m)
    ax = axes[2]
    ax.errorbar(xs, [m[x][0] for x in xs], yerr=[m[x][1] for x in xs],
                color="#d1341a", marker="o", capsize=1.5, elinewidth=0.6)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([r"$2^{10}$", r"$2^{12}$", r"$2^{14}$", "exact"])
    ax.minorticks_off()
    ax.set_xlabel(r"sketch dimension $d$")
    ax.tick_params(axis="y", colors="#d1341a", labelsize=6.6)
    ax.grid(ls=":")
    ax.set_axisbelow(True)
    ax2 = ax.twinx()
    ax2.plot(xs, [f[x][0] for x in xs], color="#1a6fd1", marker="s", ls="--", lw=1.0)
    ax2.set_ylabel("Jain index", color="#1a6fd1")
    ax2.tick_params(axis="y", colors="#1a6fd1", labelsize=6.4)
    fig.subplots_adjust(wspace=0.42)
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


# ------------------------------------------------------------------ tables
def latex_main_table(recs, attacks, caption, label, path):
    accm = agg(recs, lambda r: (r["config"]["aggregator"], r["config"]["attack"]),
               lambda r: r["acc_last5"] * 100)
    meth = [x for x in ORDER if any(k[0] == x for k in accm)]
    cols = "l" + "c" * len(attacks) + "c"
    lines = [r"\begin{table*}[!t]", r"\centering",
             r"\caption{%s}" % caption, r"\label{%s}" % label,
             r"\setlength{\tabcolsep}{4.2pt}",
             r"\begin{tabular}{%s}" % cols, r"\hline",
             "Aggregation rule & " + " & ".join(ATK[a] for a in attacks) +
             r" & \textbf{Mean} \\", r"\hline"]
    best = {}
    for a in attacks:
        vals = {mm: accm.get((mm, a), (-1,))[0] for mm in meth}
        best[a] = max(vals, key=vals.get)
    for mm in meth:
        cells = []
        row = []
        for a in attacks:
            v = accm.get((mm, a))
            if v is None:
                cells.append("--")
                continue
            s = ("%.1f" % v[0]) if v[2] < 2 else ("%.1f\\,$\\pm$\\,%.1f" % (v[0], v[1]))
            cells.append(r"\textbf{%s}" % s if best[a] == mm else s)
            row.append(v[0])
        nm = PRETTY[mm]
        if mm == "fedhift":
            lines.append(r"\hline")
        lines.append(nm + " & " + " & ".join(cells) +
                     " & %.1f \\\\" % (np.mean(row) if row else np.nan))
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    open(path, "w").write("\n".join(lines) + "\n")


def latex_fairness_table(recs, attacks, caption, label, path):
    keys = [("benign_jain", "Jain", 3, 1.0), ("benign_worst10", "Worst-10\\%", 1, 100.0),
            ("benign_std", "Std", 1, 100.0), ("mal_weight_mass", "$W_{\\mathcal{B}}$", 3, 1.0),
            ("det_auc", "AUROC", 3, 1.0)]
    ms = {k: agg(recs, lambda r: r["config"]["aggregator"], lambda r, k=k: r[k[0]] * k[3])
          for k in keys}
    meth = [x for x in ORDER if x in ms[keys[0]]]
    lines = [r"\begin{table}[!t]", r"\centering", r"\caption{%s}" % caption,
             r"\label{%s}" % label, r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l%s}" % ("c" * len(keys)), r"\hline",
             "Rule & " + " & ".join(k[1] for k in keys) + r" \\", r"\hline"]
    for mm in meth:
        cells = []
        for k in keys:
            v = ms[k].get(mm)
            cells.append("--" if v is None or not np.isfinite(v[0])
                         else ("%." + str(k[2]) + "f") % v[0])
        lines.append(PRETTY[mm] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(lines) + "\n")


def latex_cost_table(recs, path):
    m = agg(recs, lambda r: r["config"]["aggregator"], lambda r: r["agg_time_s"] * 1000 /
            r["config"]["rounds"])
    t = agg(recs, lambda r: r["config"]["aggregator"], lambda r: r["wall_time_s"])
    meth = [x for x in ORDER if x in m]
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{Server-side aggregation cost per round (Fashion-MNIST, "
             r"$m=10$ participants, $p=%d$ parameters).}" % recs[0]["num_params"],
             r"\label{tab:cost}", r"\begin{tabular}{lcc}", r"\hline",
             r"Rule & Server time / round (ms) & Overhead vs.\ FedAvg \\", r"\hline"]
    base = m["fedavg"][0]
    for mm in meth:
        lines.append("%s & %.1f & $\\times$%.1f \\\\" % (PRETTY[mm], m[mm][0], m[mm][0] / base))
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(lines) + "\n")


def _ipm_records():
    """The IPM condition was run as its own suite; fold it into the main table."""
    out = []
    for r in load("ablation_ipm"):
        c = r["config"]
        if c["aggregator"] != "fedhift" or c["tag"] == "full":
            out.append(r)
    return out


def main():
    fm = load("main_fmnist")
    fm_r = [r for r in fm if r["config"]["aggregator"] != "fedprox"] + _ipm_records()
    if fm_r:
        A = ["none", "label_flip", "sign_flip", "gauss", "scaling", "alie", "ipm"]
        fig_robustness_bars(fm_r, "fmnist", A, "fig_bars_fmnist.png")
        fig_curves(fm_r, ["none", "label_flip", "sign_flip", "scaling"],
                   "fig_curves_fmnist.png")
        fig_fairness(fm_r, "fig_fairness.png")
        latex_main_table(fm_r, A,
                         "Global test accuracy (\\%, mean $\\pm$ s.d.\\ over three seeds, "
                         "averaged over the last five evaluation points) on Fashion-MNIST "
                         "with $K=30$ clients, Dirichlet $\\alpha=0.5$ and 20\\% Byzantine "
                         "participants. Best per column in bold.",
                         "tab:main_fmnist", os.path.join(TAB, "tab_main_fmnist.tex"))
        latex_fairness_table([r for r in fm_r if r["config"]["attack"] != "none"],
                             A, "Fairness and detection behaviour on Fashion-MNIST, "
                                "averaged over the six attacks and all seeds. Jain, "
                                "worst-decile accuracy and its standard deviation are "
                                "computed over the honest clients only; $W_{\\mathcal{B}}$ is "
                                "the aggregate weight mass captured by Byzantine clients "
                                "($0.18$ under uniform weighting) and AUROC measures how well "
                                "the assigned weights separate benign from Byzantine "
                                "participants.",
                             "tab:fairness", os.path.join(TAB, "tab_fairness.tex"))
        latex_cost_table(fm_r, os.path.join(TAB, "tab_cost.tex"))

    cf = load("main_cifar")
    if cf:
        A = ["none", "label_flip", "sign_flip", "scaling", "alie"]
        fig_robustness_bars(cf, "cifar10", A, "fig_bars_cifar.png")
        latex_main_table(cf, A,
                         "Global test accuracy (\\%) on CIFAR-10 under the same federation "
                         "($K=30$, $\\alpha=0.5$, 20\\% Byzantine participants, 40 "
                         "communication rounds, single seed). Best per column in bold.",
                         "tab:main_cifar", os.path.join(TAB, "tab_main_cifar.tex"))

    ni = load("noniid")
    bz = load("byzfrac") + [r for r in load("main_fmnist")
                            if r["config"]["attack"] == "sign_flip"
                            and r["config"]["aggregator"] in
                            ("fedavg", "median", "multikrum", "rfa", "fltrust", "fedhift")]
    if bz and ni:
        fig_sweeps(ni, bz, "fig_sweeps.png")
    ab = load("ablation") + [r for r in load("ablation_ipm")
                             if r["config"]["aggregator"] == "fedhift"]
    if ab:
        fig_ablation([r for r in ab if not r["config"]["tag"].endswith("_a005")],
                     "fig_ablation.png")
    fo, ft = load("fou"), load("fou_temp")
    if fo and ft:
        fig_fou(fo, ft, "fig_fou.png")
    se = load("sensitivity")
    if se:
        fig_sensitivity(se, "fig_sensitivity.png")
    try:
        fig_trust_dynamics("fig_trust.png")
    except Exception as e:
        print("trust dynamics figure skipped:", e)
    print("figures ->", sorted(os.listdir(FIG)))
    print("tables  ->", sorted(os.listdir(TAB)))


if __name__ == "__main__":
    main()
