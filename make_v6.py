"""Tables and figures for the revision, built from the main_v6 grid.

Every figure obeys the publisher's graphics guidance through plotstyle.py: one
serif family at 9 pt, one-column (3.5 in) or two-column (7.16 in) widths, a
color-vision-safe palette, and color never used as the only channel.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotstyle import ONE_COL, TWO_COL, LEGEND, apply_style, style_for, COL

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

ORDER = ["fedavg", "fedavg_clip", "median", "trimmed_mean", "multikrum",
         "rfa", "fltrust", "klcos", "fedhift"]
PRETTY = {"fedavg": "FedAvg", "fedavg_clip": "FedAvg+clip",
          "median": "Coord. med.", "trimmed_mean": "Trimmed mean",
          "multikrum": "Multi-Krum", "rfa": "RFA", "fltrust": "FLTrust",
          "klcos": "KL-cos (ours)", "fedhift": "FedEFT (ours)",
          "fedprox": "FedProx"}
ATK = ["none", "label_flip", "sign_flip", "gauss", "scaling", "alie", "ipm",
       "adaptive"]
ATKNAME = {"none": "No attack", "label_flip": "Label flip",
           "sign_flip": "Sign flip", "gauss": "Gaussian", "scaling": "Scaling",
           "alie": "ALIE", "ipm": "IPM", "adaptive": "Adaptive"}


def load(suite):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, suite, "*.json"))):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def complete_seeds(recs, rules, attacks):
    """Seeds for which every (rule, attack) cell exists, so the grid is balanced."""
    have = defaultdict(set)
    for r in recs:
        c = r["config"]
        have[c["seed"]].add((c["aggregator"], c["attack"]))
    need = {(m, a) for m in rules for a in attacks}
    return sorted(s for s in have if need <= have[s])


def agg(recs, keyfn, valfn):
    d = defaultdict(list)
    for r in recs:
        d[keyfn(r)].append(valfn(r))
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in d.items()}


# --------------------------------------------------------------- main table --
def main_table(recs, seeds, attacks, caption, label, path, wide=True):
    recs = [r for r in recs if r["config"]["seed"] in seeds]
    m = agg(recs, lambda r: (r["config"]["aggregator"], r["config"]["attack"]),
            lambda r: r["acc_last5"] * 100)
    meth = [x for x in ORDER if any(k[0] == x for k in m)]
    best = {a: max(meth, key=lambda mm: m.get((mm, a), (-1,))[0]) for a in attacks}
    env = "table*" if wide else "table"
    L = [r"\begin{%s}[!t]" % env, r"\centering", r"\caption{%s}" % caption,
         r"\label{%s}" % label, r"\setlength{\tabcolsep}{3.4pt}", r"\footnotesize",
         r"\begin{tabular}{l%s}" % ("c" * (len(attacks) + 1)), r"\hline",
         "Aggregation rule & " + " & ".join(ATKNAME[a] for a in attacks) +
         r" & \textbf{Mean} \\", r"\hline"]
    for mm in meth:
        cells, row = [], []
        for a in attacks:
            v = m.get((mm, a))
            if v is None:
                cells.append("--")
                continue
            txt = "%.1f\\,$\\pm$\\,%.1f" % (v[0], v[1])
            cells.append(r"\textbf{%s}" % txt if best[a] == mm else txt)
            row.append(v[0])
        if mm in ("klcos", "fedhift"):
            L.append(r"\hline")
        L.append(PRETTY[mm] + " & " + " & ".join(cells) +
                 " & %.1f \\\\" % (np.mean(row) if row else np.nan))
    L += [r"\hline", r"\end{tabular}", r"\end{%s}" % env]
    open(path, "w").write("\n".join(L) + "\n")


# ----------------------------------------------------------- fairness table --
def fairness_table(recs, seeds, path, caption):
    recs = [r for r in recs if r["config"]["seed"] in seeds
            and r["config"]["attack"] != "none"]
    keys = [("benign_jain", "$J_{\\mathrm{acc}}$", 4, 1.0),
            ("benign_jain_bacc", "$J_{\\mathrm{bal}}$", 4, 1.0),
            ("benign_mean_f1", "macro $F_1$", 3, 1.0),
            ("benign_worst10_bacc", "w-10\\%", 1, 100.0),
            ("mal_weight_mass", "$W_{\\mathcal{B}}$", 3, 1.0),
            ("det_auc", "AUROC", 3, 1.0)]
    ms = {k: agg(recs, lambda r: r["config"]["aggregator"],
                 lambda r, k=k: r[k[0]] * k[3]) for k in keys}
    cost = agg(recs, lambda r: r["config"]["aggregator"],
               lambda r: r["agg_time_s"] * 1000 / r["config"]["rounds"])
    meth = [x for x in ORDER if x in ms[keys[0]]]
    base = cost["fedavg"][0]
    L = [r"\begin{table}[!t]", r"\centering", r"\caption{%s}" % caption,
         r"\label{tab:fairness}", r"\setlength{\tabcolsep}{1.9pt}", r"\scriptsize",
         r"\begin{tabular}{l%s}" % ("c" * (len(keys) + 1)), r"\hline",
         "Rule & " + " & ".join(k[1] for k in keys) + r" & ms/rd \\", r"\hline"]
    for mm in meth:
        cells = []
        for k in keys:
            v = ms[k].get(mm)
            cells.append("--" if v is None or not np.isfinite(v[0])
                         else ("%." + str(k[2]) + "f") % v[0])
        c = cost.get(mm)
        cells.append("--" if c is None else "%.1f" % c[0])
        L.append(PRETTY[mm] + " & " + " & ".join(cells) + r" \\")
    L += [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")


# -------------------------------------------------------------- components ---
def fig_components(recs, seeds, fname):
    """What each component of the rule is worth, per attack."""
    recs = [r for r in recs if r["config"]["seed"] in seeds]
    m = agg(recs, lambda r: (r["config"]["aggregator"], r["config"]["attack"]),
            lambda r: r["acc_last5"] * 100)
    chain = ["fedavg", "fedavg_clip", "klcos", "fedhift"]
    apply_style()
    fig, ax = plt.subplots(figsize=(TWO_COL * 0.62, 1.95))
    x = np.arange(len(ATK))
    w = 0.8 / len(chain)
    for i, mm in enumerate(chain):
        mu = [m.get((mm, a), (np.nan,) * 3)[0] for a in ATK]
        sd = [m.get((mm, a), (np.nan,) * 3)[1] for a in ATK]
        ax.bar(x + (i - (len(chain) - 1) / 2) * w, mu, w, yerr=sd, capsize=1.3,
               label=PRETTY[mm], color=COL[i], edgecolor="black", linewidth=0.4,
               error_kw=dict(lw=0.5), hatch=["", "//", "\\\\", "xx"][i])
    ax.set_xticks(x)
    ax.set_xticklabels([ATKNAME[a] for a in ATK], rotation=22, ha="right")
    ax.set_ylabel("Global test accuracy (\\%)" if False else "Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", ls=":")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


# ------------------------------------------------- adaptive alignment budget --
def fig_rho(fname="fig_rho.png"):
    """Damage against detectability as the white-box attack varies its
    alignment budget rho."""
    recs = load("adaptive_rho")
    if not recs:
        print("adaptive_rho is empty")
        return
    acc = agg(recs, lambda r: (r["config"]["aggregator"], r["config"]["attack_rho"]),
              lambda r: r["acc_last5"] * 100)
    auc = agg(recs, lambda r: (r["config"]["aggregator"], r["config"]["attack_rho"]),
              lambda r: r["det_auc"])
    wb = agg(recs, lambda r: (r["config"]["aggregator"], r["config"]["attack_rho"]),
             lambda r: r["mal_weight_mass"])
    rhos = sorted({k[1] for k in acc})
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(ONE_COL, 2.05))
    for i, mm in enumerate(["fedavg", "rfa", "fedhift"]):
        st = style_for(i * 3)
        axes[0].errorbar(rhos, [acc.get((mm, r), (np.nan,) * 3)[0] for r in rhos],
                         yerr=[acc.get((mm, r), (np.nan,) * 3)[1] for r in rhos],
                         label=PRETTY[mm], capsize=1.5, elinewidth=0.6, **st)
    axes[0].set_ylabel("Accuracy (%)")
    # a fixed window plus the clean-federation reference, so that a flat curve
    # reads as "no damage" rather than as an arbitrary magnification
    axes[0].axhline(85.1, color="black", lw=0.6, ls=":")
    axes[0].set_ylim(78.0, 88.0)
    axes[0].legend(loc="lower left", **LEGEND)
    st = style_for(0)
    axes[1].errorbar(rhos, [auc.get(("fedhift", r), (np.nan,) * 3)[0] for r in rhos],
                     yerr=[auc.get(("fedhift", r), (np.nan,) * 3)[1] for r in rhos],
                     label="AUROC", capsize=1.5, elinewidth=0.6, **st)
    st = style_for(4)
    axes[1].errorbar(rhos, [wb.get(("fedhift", r), (np.nan,) * 3)[0] for r in rhos],
                     yerr=[wb.get(("fedhift", r), (np.nan,) * 3)[1] for r in rhos],
                     label="$W_{\\mathcal{B}}$", capsize=1.5, elinewidth=0.6, **st)
    axes[1].axhline(0.5, color="black", lw=0.6, ls=":")
    axes[1].axhline(0.181, color="black", lw=0.6, ls="--")
    axes[1].set_ylabel("FedEFT detection")
    axes[1].set_ylim(0.0, 0.62)
    axes[1].legend(loc="lower left", **LEGEND)
    for ax in axes:
        ax.set_xlabel("alignment budget $\\varrho$")
        ax.grid(ls=":")
        ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)
    print("rho sweep:")
    for r in rhos:
        print(f"  rho={r:<4g} " + "  ".join(
            f"{PRETTY[m]}={acc.get((m, r), (np.nan,)*3)[0]:.1f}"
            for m in ["fedavg", "rfa", "fedhift"]) +
            f"  AUROC={auc.get(('fedhift', r), (np.nan,)*3)[0]:.3f}"
            f"  W_B={wb.get(('fedhift', r), (np.nan,)*3)[0]:.3f}")


# ------------------------------------------------------------------- main ----
def main():
    recs = load("main_v6")
    recs = [r for r in recs if r["config"]["aggregator"] != "fedprox"]
    if not recs:
        print("main_v6 is empty")
        return
    seeds = complete_seeds(recs, ORDER, ATK)
    print("complete seeds:", seeds)
    if not seeds:
        return
    ns = len(seeds)
    main_table(recs, seeds, ATK,
               "Global test accuracy (\\%%, mean $\\pm$ s.d.\\ over %d seeds) on "
               "Fashion-MNIST with $K=30$ clients, Dirichlet $\\alpha=0.5$ and 20\\%% "
               "Byzantine participants, averaged over the last five evaluation "
               "points. KL-cos is the same allocation and clipping driven by a plain "
               "cosine score instead of the fuzzy engine. Best per column in bold." % ns,
               "tab:main_fmnist", os.path.join(TAB, "tab_main_fmnist.tex"))
    fairness_table(recs, seeds, os.path.join(TAB, "tab_fairness.tex"),
                   "Outcome fairness, detection and server cost on Fashion-MNIST, "
                   "averaged over the seven attack conditions and %d seeds. "
                   "$J_{\\mathrm{acc}}$ and $J_{\\mathrm{bal}}$ are Jain indices over "
                   "per-client accuracy and per-client balanced accuracy; macro $F_1$ "
                   "and the worst-decile column are also computed on balanced "
                   "per-client scores, which are invariant to how easy each client's "
                   "own label mix is. $W_{\\mathcal{B}}$ is the weight mass captured by "
                   "Byzantine clients (0.18 under uniform weighting) and AUROC "
                   "separates benign from Byzantine participants, benign positive; "
                   "for the median, the trimmed mean and RFA these two columns "
                   "describe the reference weighting, since those rules expose no "
                   "per-client weight. The last column is aggregation time per round "
                   "at $m=10$, $p=215{,}370$." % ns)
    fig_components(recs, seeds, "fig_components.png")
    fig_rho()
    print("wrote tab_main_fmnist, tab_fairness, fig_components")


def report(recs, seeds):
    """Print every quantity the results section quotes, from the balanced grid."""
    recs = [r for r in recs if r["config"]["seed"] in seeds]
    for metric, scale, nm in [("acc_last5", 100.0, "accuracy"),
                              ("benign_jain", 1.0, "Jain (accuracy)"),
                              ("benign_jain_bacc", 1.0, "Jain (balanced)"),
                              ("benign_mean_bacc", 100.0, "mean balanced acc"),
                              ("benign_mean_f1", 1.0, "mean macro F1"),
                              ("benign_worst10_bacc", 100.0, "worst-10% balanced"),
                              ("mal_weight_mass", 1.0, "Byzantine weight mass"),
                              ("det_auc", 1.0, "AUROC")]:
        m = agg(recs, lambda r: (r["config"]["aggregator"], r["config"]["attack"]),
                lambda r: r[metric] * scale)
        meth = [x for x in ORDER if any(k[0] == x for k in m)]
        print(f"\n### {nm}")
        print(f"{'':22s}" + "".join(f"{ATKNAME[a][:9]:>10s}" for a in ATK) + f"{'MEAN':>10s}")
        for mm in meth:
            row = [m.get((mm, a), (np.nan,) * 3)[0] for a in ATK]
            f = "%10.2f" if scale != 1.0 else "%10.4f"
            print(f"{PRETTY[mm]:22s}" + "".join(f % v for v in row) +
                  (f % np.nanmean(row)))
    sd = agg(recs, lambda r: (r["config"]["aggregator"], r["config"]["attack"]),
             lambda r: r["acc_last5"] * 100)
    print("\n### accuracy s.d. by rule and attack")
    for mm in [x for x in ORDER if any(k[0] == x for k in sd)]:
        print(f"{PRETTY[mm]:22s}" +
              "".join("%10.2f" % sd.get((mm, a), (np.nan,) * 3)[1] for a in ATK))
    print("\n### accuracy range across conditions")
    for mm in [x for x in ORDER if any(k[0] == x for k in sd)]:
        v = [sd[(mm, a)][0] for a in ATK if (mm, a) in sd]
        print(f"  {PRETTY[mm]:22s} min={min(v):6.2f} max={max(v):6.2f} range={max(v)-min(v):6.2f}")


if __name__ == "__main__":
    main()
    rs = [r for r in load("main_v6") if r["config"]["aggregator"] != "fedprox"]
    if rs:
        report(rs, complete_seeds(rs, ORDER, ATK))
