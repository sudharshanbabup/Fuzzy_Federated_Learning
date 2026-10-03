"""Every table and figure-panel JSON of the FODM manuscript, from the run records.

    python make_fodm.py            # tables (tables_fodm/) + panel JSONs (matlab/data/)
    python make_fodm.py stats      # also print the paired statistics quoted in the text

Figures are then rendered by matlab/matlab_fig_1.m (MATLAB or GNU Octave):
    cd matlab && matlab_fig_1
Numbers quoted in the manuscript are printed to logs/fodm_numbers.txt so that
text, tables and figures are generated from the same records.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from significance import holm, perm_p, t_ci  # noqa: E402
from fodm.common import STYLE, NAMES, PAL, dump, series  # noqa: E402

RES = os.path.join(HERE, "results")
TAB = os.path.join(HERE, "tables_fodm")
os.makedirs(TAB, exist_ok=True)
NUM = open(os.path.join(HERE, "logs", "fodm_numbers.txt"), "w")

ATK = {"none": "No attack", "label_flip": "Label flip", "sign_flip": "Sign flip",
       "gauss": "Gaussian", "scaling": "Scaling", "alie": "ALIE", "ipm": "IPM",
       "adaptive": "Adaptive", "backdoor": "Backdoor", "backdoor_boost": "Boosted bd."}
ATK8 = ["none", "label_flip", "sign_flip", "gauss", "scaling", "alie", "ipm", "adaptive"]


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    NUM.write(s + "\n")


_CACHE = {}


def load(suite):
    if suite not in _CACHE:
        out = []
        for f in sorted(glob.glob(os.path.join(RES, suite, "*.json"))):
            with open(f) as fh:
                out.append(json.load(fh))
        _CACHE[suite] = out
    return _CACHE[suite]


def C(r, k, d=None):
    return r["config"].get(k, d)


def cell(recs, key, metric="acc_last5", scale=100.0):
    """dict key(r) -> list of metric values"""
    d = defaultdict(list)
    for r in recs:
        v = r.get(metric)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        d[key(r)].append(v * scale)
    return d


def ms(v):
    return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)


def fmt(mu, sd=None, p=1):
    if not np.isfinite(mu):
        return "---"
    return f"{mu:.{p}f}" if sd is None else f"{mu:.{p}f}\\,$\\pm$\\,{sd:.{p}f}"


def write(name, lines):
    with open(os.path.join(TAB, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote", name)


def paired(recs, a, b, cellkey, metric="acc_last5", scale=100.0, filt=None):
    """Paired differences a - b over matched cells."""
    da, db = {}, {}
    for r in recs:
        if filt and not filt(r):
            continue
        v = r.get(metric)
        if v is None or not np.isfinite(v):
            continue
        k = cellkey(r)
        if C(r, "aggregator") == a:
            da[k] = v * scale
        elif C(r, "aggregator") == b:
            db[k] = v * scale
    ks = sorted(set(da) & set(db))
    return np.array([da[k] - db[k] for k in ks])


def family(recs, ref, others, cellkey, metric="acc_last5", scale=100.0, filt=None):
    rows = {}
    for o in others:
        d = paired(recs, ref, o, cellkey, metric, scale, filt)
        if len(d) == 0:
            continue
        lo, hi = t_ci(d)
        rows[o] = dict(n=len(d), mean=float(d.mean()), lo=lo, hi=hi, p=perm_p(d),
                       wins=int((d > 0).sum()))
    if rows:
        ph = holm([rows[o]["p"] for o in rows])
        for o, q in zip(rows, ph):
            rows[o]["ph"] = q
    return rows


def ptxt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# =========================================================================== #
# Tables
# =========================================================================== #
def acc_table(name, recs, rules, atks, caption, label, ncols=4, extra_cols=None,
              dagger=()):
    a = cell(recs, lambda r: (C(r, "aggregator"), C(r, "attack")))
    rules = [m for m in rules if any((m, x) in a for x in atks)]
    mean = {m: float(np.mean([np.mean(a[(m, x)]) for x in atks if (m, x) in a])) for m in rules}
    groups = [atks[i:i + ncols] for i in range(0, len(atks), ncols)]
    width = max(len(g) for g in groups) + 1 + 1
    L = [r"\begin{table}[!tbp]", r"\centering", r"\caption{%s}" % caption, r"\label{%s}" % label,
         r"\setlength{\tabcolsep}{3.0pt}", r"\footnotesize",
         r"\begin{tabular}{@{}l%s@{}}" % ("c" * (width - 1)), r"\toprule"]
    for gi, g in enumerate(groups):
        last = gi == len(groups) - 1
        head = ["Rule"] + [ATK[x] for x in g] + ([r"\textbf{Mean}"] if last else [])
        head += [""] * (width - len(head))
        L += [" & ".join(head) + r"\\", r"\midrule"]
        best = {x: max(np.mean(a[(m, x)]) for m in rules if (m, x) in a) for x in g}
        topm = max(mean.values())
        for m in rules:
            if m == "fedhift":
                L.append(r"\midrule")
            nm = NAMES[m] + (r"$^\dagger$" if m in dagger else "")
            row = [nm]
            for x in g:
                v = a.get((m, x))
                if not v:
                    row.append("---")
                    continue
                mu, sd = ms(v)
                t = fmt(mu, sd)
                row.append(r"\textbf{%s}" % t if abs(mu - best[x]) < 0.05 else t)
            if last:
                t = fmt(mean[m])
                row.append(r"\textbf{%s}" % t if abs(mean[m] - topm) < 0.05 else t)
            row += [""] * (width - len(row))
            L.append(" & ".join(row) + r"\\")
        if not last:
            L.append(r"\midrule")
    L += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    write(name, L)
    return a, mean


def tab_fmnist():
    old = [r for r in load("main_v6") if C(r, "aggregator") != "fedprox"]
    new = [r for r in load("modern") if C(r, "aggregator") in ("flame", "bulyan", "dnc")]
    rules = ["fedavg", "fedavg_clip", "median", "trimmed_mean", "multikrum", "rfa",
             "fltrust", "flame", "bulyan", "dnc", "klcos", "fedhift"]
    acc_table("tab_main_fmnist.tex", old + new, rules, ATK8,
              "Fashion-MNIST accuracy (\\%, mean\\,$\\pm$\\,s.d.; $K=30$, $\\alpha=0.5$, "
              "20\\% Byzantine). Five seeds; $^\\dagger$three seeds, run with FedEFT "
              "re-executed under the same software stack for the paired tests "
              "(Table~\\ref{tab:stats}). Best per column in bold.",
              "tab:main_fmnist", dagger=("flame", "bulyan", "dnc"))


def tab_cifar():
    recs = load("main_cifar") + load("cifar_v6")
    rules = ["fedavg", "fedavg_clip", "median", "trimmed_mean", "multikrum", "rfa",
             "fltrust", "klcos", "fedhift"]
    acc_table("tab_main_cifar.tex", recs, rules, ["none", "label_flip", "sign_flip",
                                                    "scaling", "alie"],
              "CIFAR-10 accuracy (\\%, three seeds, 40 rounds; $K=30$, $\\alpha=0.5$, "
              "20\\% Byzantine). Best per column in bold.", "tab:main_cifar", ncols=5)


def tab_femnist():
    recs = load("femnist")
    if not recs:
        return
    rules = ["fedavg", "fedavg_clip", "median", "trimmed_mean", "multikrum", "rfa",
             "fltrust", "flame", "bulyan", "dnc", "klcos", "fedhift"]
    atks = ["none", "label_flip", "sign_flip", "scaling", "backdoor"]
    a = cell(recs, lambda r: (C(r, "aggregator"), C(r, "attack")))
    asr = cell([r for r in recs if C(r, "attack") == "backdoor"],
               lambda r: C(r, "aggregator"), "asr_last5")
    jain = cell([r for r in recs if C(r, "attack") != "none"],
                lambda r: C(r, "aggregator"), "benign_worst10")
    rules = [m for m in rules if any((m, x) in a for x in atks)]
    mean = {m: float(np.mean([np.mean(a[(m, x)]) for x in atks if (m, x) in a])) for m in rules}
    L = [r"\begin{table}[!tbp]", r"\centering",
         r"\caption{FEMNIST with natural writer heterogeneity ($K=100$ writers, $m=20$, "
         r"62 classes, 50 rounds, 20\% Byzantine, two seeds): accuracy (\%), backdoor "
         r"attack success rate (ASR, \%, lower is better) and worst-decile honest-client "
         r"accuracy averaged over the four attacks (W10, \%). Best per column in bold.}",
         r"\label{tab:femnist}", r"\setlength{\tabcolsep}{3.2pt}", r"\footnotesize",
         r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r"Rule & No attack & Label flip & Sign flip & Scaling & Backdoor & ASR$\downarrow$ & W10 & \textbf{Mean}\\",
         r"\midrule"]
    best = {x: max(np.mean(a[(m, x)]) for m in rules if (m, x) in a) for x in atks}
    basr = min(np.mean(asr[m]) for m in rules if m in asr)
    bw = max(np.mean(jain[m]) for m in rules if m in jain)
    topm = max(mean.values())
    for m in rules:
        if m == "fedhift":
            L.append(r"\midrule")
        row = [NAMES[m]]
        for x in atks:
            v = a.get((m, x))
            mu, sd = ms(v)
            t = fmt(mu, sd)
            row.append(r"\textbf{%s}" % t if abs(mu - best[x]) < 0.05 else t)
        mu = np.mean(asr[m]) if m in asr else np.nan
        row.append(r"\textbf{%s}" % fmt(mu) if abs(mu - basr) < 0.05 else fmt(mu))
        mu = np.mean(jain[m]) if m in jain else np.nan
        row.append(r"\textbf{%s}" % fmt(mu) if abs(mu - bw) < 0.05 else fmt(mu))
        row.append(r"\textbf{%s}" % fmt(mean[m]) if abs(mean[m] - topm) < 0.05 else fmt(mean[m]))
        L.append(" & ".join(row) + r"\\")
    L += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    write("tab_femnist.tex", L)
    say("FEMNIST means:", {m: round(mean[m], 2) for m in rules})
    say("FEMNIST ASR:", {m: round(float(np.mean(asr[m])), 1) for m in rules if m in asr})


def tab_stats():
    """FedEFT against every comparator, pooled per study."""
    rules = ["fedavg", "fedavg_clip", "median", "trimmed_mean", "multikrum", "rfa",
             "fltrust", "flame", "bulyan", "dnc", "klcos"]
    ck = lambda r: (C(r, "attack"), C(r, "seed"), C(r, "alpha"))
    studies = []
    fm = [r for r in load("main_v6") if C(r, "aggregator") != "fedprox"]
    studies.append(("F-MNIST", family(fm, "fedhift", rules, ck)))
    studies.append(("F-MNIST$^\\dagger$", family(load("modern"), "fedhift",
                                                  ["flame", "bulyan", "dnc"], ck)))
    studies.append(("CIFAR-10", family(load("main_cifar") + load("cifar_v6"), "fedhift", rules, ck)))
    if load("femnist"):
        studies.append(("FEMNIST", family([r for r in load("femnist") if C(r, "byz_frac") == 0.2],
                                          "fedhift", rules, ck)))
    studies.append(("Dirichlet sweep", family(load("noniid"), "fedhift", rules, ck)))
    L = [r"\begin{table}[!tbp]", r"\centering",
         r"\caption{Paired comparison of FedEFT against each rule, pooled over all matched "
         r"(attack, seed[, $\alpha$]) cells of each study: mean accuracy difference in "
         r"points and Holm-corrected exact permutation $p$-value (in parentheses); "
         r"$n$ is the number of cells. $^\dagger$FedEFT re-run under the updated software "
         r"stack against the three modern rules. Positive values favour FedEFT.}",
         r"\label{tab:stats}", r"\setlength{\tabcolsep}{2.2pt}", r"\footnotesize",
         r"\begin{tabular}{@{}l%s@{}}" % ("c" * len(studies)), r"\toprule",
         "Comparator & " + " & ".join(s[0] for s in studies) + r"\\"]
    L.append(" & ".join(["$n$ cells"] + [str(next(iter(s[1].values()))["n"]) if s[1] else "--"
                                          for s in studies]) + r"\\")
    L.append(r"\midrule")
    for o in rules:
        row = [NAMES[o]]
        for nm, fam in studies:
            if o in fam:
                q = fam[o]
                t = f"{q['mean']:+.2f} ({ptxt(q['ph'])})"
                row.append(r"\textbf{%s}" % t if q["ph"] < 0.05 else t)
            elif nm.startswith("F-MNIST") and o in ("flame", "bulyan", "dnc") and "dagger" not in nm:
                row.append("--")
            else:
                row.append("--")
        L.append(" & ".join(row) + r"\\")
    L += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    write("tab_stats.tex", L)
    for nm, fam in studies:
        for o, q in fam.items():
            say(f"STAT {nm:18s} vs {o:12s} n={q['n']:3d} d={q['mean']:+7.2f} "
                f"CI=[{q['lo']:+.2f},{q['hi']:+.2f}] p={q['p']:.4f} pH={q['ph']:.4f} wins={q['wins']}/{q['n']}")


def tab_trustfn():
    recs = load("trustfn")
    if not recs:
        return
    order = ["tr_cosine", "tr_linear", "tr_linopt", "tr_logistic", "tr_mlp", "tr_type1", "fedhift"]
    seen = ["label_flip", "sign_flip"]
    unseen = ["scaling", "alie", "ipm", "backdoor"]
    recs = [r for r in recs if C(r, "attack") in seen + unseen]   # drop stray pilot runs
    a05 = [r for r in recs if C(r, "alpha") == 0.5]
    a01 = [r for r in recs if C(r, "alpha") == 0.1]
    A = cell(a05, lambda r: (C(r, "aggregator"), C(r, "attack")))
    A1 = cell(a01, lambda r: (C(r, "aggregator"), C(r, "attack")))
    WB = cell(a05, lambda r: (C(r, "aggregator"), C(r, "attack")), "mal_weight_mass", 1.0)
    AU = cell([r for r in a05 if C(r, "attack") in ("label_flip", "sign_flip", "scaling")],
              lambda r: C(r, "aggregator"), "det_auc", 1.0)
    W10 = cell(a05, lambda r: C(r, "aggregator"), "benign_worst10")
    JN = cell(a05, lambda r: C(r, "aggregator"), "benign_jain", 1.0)
    ASR = cell([r for r in a05 if C(r, "attack") == "backdoor"], lambda r: C(r, "aggregator"),
               "asr_last5")
    AT = cell(a05, lambda r: C(r, "aggregator"), "agg_time_s", 1000.0 / 30)
    cols = []
    for m in order:
        s_seen = np.mean([np.mean(A[(m, x)]) for x in seen if (m, x) in A])
        s_un = np.mean([np.mean(A[(m, x)]) for x in unseen if (m, x) in A])
        s01 = np.mean([np.mean(A1[(m, x)]) for x in seen if (m, x) in A1]) if A1 else np.nan
        wb = np.mean([np.mean(WB[(m, x)]) for x in seen + ["scaling"] if (m, x) in WB])
        cols.append(dict(m=m, seen=s_seen, unseen=s_un, a01=s01, wb=wb,
                         auc=np.mean(AU[m]) if m in AU else np.nan,
                         w10=np.mean(W10[m]), jain=np.mean(JN[m]),
                         asr=np.mean(ASR[m]) if m in ASR else np.nan, t=np.mean(AT[m])))
    keys = [("seen", "Acc.\\ seen attacks", 1, max), ("unseen", "Acc.\\ unseen attacks", 1, max),
            ("a01", "Acc.\\ $\\alpha=0.1$", 1, max), ("wb", "$W_\\cB$", 3, min),
            ("auc", "AUROC", 3, max), ("asr", "Backdoor ASR", 1, min),
            ("w10", "Worst-10\\%", 1, max), ("jain", "Jain", 4, max), ("t", "ms/round", 1, None)]
    L = [r"\begin{table}[!tbp]", r"\centering",
         r"\caption{Trust-function comparison. Statistics, reputation, entropic allocation "
         r"and clipping are identical; only the map $\mathbf u\mapsto\hat\tau$ changes. "
         r"Fashion-MNIST, $\alpha=0.5$ unless stated, three seeds. ``Seen'' attacks (label "
         r"and sign flipping) are those the fitted scorers were trained on; ``unseen'' "
         r"averages scaling, ALIE, IPM and the backdoor (clean accuracy). $W_\cB$ and AUROC "
         r"are averaged over label flipping, sign flipping and scaling; worst-decile and "
         r"Jain over all six attacks; ms/round is the server aggregation time. Best per "
         r"row in bold.}",
         r"\label{tab:trustfn}", r"\setlength{\tabcolsep}{2.6pt}", r"\footnotesize",
         r"\begin{tabular}{@{}l%s@{}}" % ("c" * len(order)), r"\toprule",
         r" & Cosine & Linear & Linear & Logistic & MLP & Type-1 & IT2 \\",
         r"Metric &  & (equal) & (fitted) & (fitted) & (fitted) & fuzzy & fuzzy\\", r"\midrule"]
    for k, lab, p, best in keys:
        vals = [c[k] for c in cols]
        bv = best(v for v in vals if np.isfinite(v)) if best else None
        row = [lab]
        for v in vals:
            t = f"{v:.{p}f}" if np.isfinite(v) else "--"
            row.append(r"\textbf{%s}" % t if (best and np.isfinite(v) and abs(v - bv) < 0.5 * 10 ** -p) else t)
        L.append(" & ".join(row) + r"\\")
    L += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    write("tab_trustfn.tex", L)
    for c in cols:
        say("TRUSTFN", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
    # per-attack detail + paired stats vs fedhift
    ck = lambda r: (C(r, "attack"), C(r, "seed"), C(r, "alpha"))
    for m in order:
        say("TRUSTFN per-attack", m, {x: round(float(np.mean(A[(m, x)])), 2) for x in seen + unseen if (m, x) in A},
            {x: round(float(np.mean(A1[(m, x)])), 2) for x in seen if (m, x) in A1})
    for subset, filt in [("all", None), ("unseen", lambda r: C(r, "attack") in unseen and C(r, "alpha") == 0.5),
                         ("seen", lambda r: C(r, "attack") in seen)]:
        fam = family(recs, "fedhift", order[:-1], ck, filt=filt)
        for o, q in fam.items():
            say(f"TRUSTSTAT {subset:7s} fedhift-{o:12s} n={q['n']} d={q['mean']:+.2f} "
                f"CI=[{q['lo']:+.2f},{q['hi']:+.2f}] p={q['p']:.4f} pH={q['ph']:.4f} wins={q['wins']}")
    fam = family(recs, "tr_type1", ["tr_cosine", "tr_linear", "tr_linopt", "tr_logistic", "tr_mlp"], ck)
    for o, q in fam.items():
        say(f"TRUSTSTAT all     type1-{o:12s} n={q['n']} d={q['mean']:+.2f} p={q['p']:.4f} pH={q['ph']:.4f} wins={q['wins']}")


def tab_backdoor():
    recs = load("backdoor")
    if not recs:
        return
    rules = ["fedavg", "fedavg_clip", "median", "trimmed_mean", "multikrum", "rfa",
             "fltrust", "flame", "bulyan", "dnc", "klcos", "fedhift"]
    cols = [("backdoor", 1.0), ("backdoor", 0.5), ("backdoor", 0.1), ("backdoor_boost", 0.5)]
    A = cell(recs, lambda r: (C(r, "aggregator"), C(r, "attack"), C(r, "alpha")))
    S = cell(recs, lambda r: (C(r, "aggregator"), C(r, "attack"), C(r, "alpha")), "asr_last5")
    L = [r"\begin{table}[!tbp]", r"\centering",
         r"\caption{Targeted backdoor on Fashion-MNIST (20\% Byzantine): clean accuracy "
         r"(ACC, \%) and attack success rate (ASR, \%, lower is better) at three skews, "
         r"and a backdoor boosted fivefold (model replacement) at $\alpha=0.5$. Three seeds "
         r"(boosted: two). Best ASR per column in bold.}",
         r"\label{tab:backdoor}", r"\setlength{\tabcolsep}{3.0pt}", r"\footnotesize",
         r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{$\alpha=1$} & \multicolumn{2}{c}{$\alpha=0.5$} & "
         r"\multicolumn{2}{c}{$\alpha=0.1$} & \multicolumn{2}{c}{Boosted, $\alpha=0.5$}\\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"Rule & ACC & ASR & ACC & ASR & ACC & ASR & ACC & ASR\\", r"\midrule"]
    best = {c: min(np.mean(S[(m,) + c]) for m in rules if (m,) + c in S) for c in cols}
    for m in rules:
        if m == "fedhift":
            L.append(r"\midrule")
        row = [NAMES[m]]
        for c in cols:
            a = A.get((m,) + c)
            s = S.get((m,) + c)
            row.append(fmt(np.mean(a)) if a else "--")
            t = fmt(np.mean(s)) if s else "--"
            row.append(r"\textbf{%s}" % t if s and abs(np.mean(s) - best[c]) < 0.05 else t)
        L.append(" & ".join(row) + r"\\")
    L += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    write("tab_backdoor.tex", L)
    for c in cols:
        say("BACKDOOR", c, {m: (round(float(np.mean(A[(m,) + c])), 1), round(float(np.mean(S[(m,) + c])), 1))
                            for m in rules if (m,) + c in A})
    ck = lambda r: (C(r, "attack"), C(r, "seed"), C(r, "alpha"))
    fam = family(recs, "fedhift", rules[:-1], ck, metric="asr_last5")
    for o, q in fam.items():
        say(f"BDSTAT ASR fedhift-{o:12s} n={q['n']} d={q['mean']:+.2f} p={q['p']:.4f} pH={q['ph']:.4f} lower_wins={q['n']-q['wins']}")


def tab_fou():
    recs = load("fou") + load("fou_temp")
    seen = {}
    for r in recs:
        c = r["config"]
        k = (c["type1"], c["alpha"], c["temperature"], c["seed"])
        seen[k] = r["acc_last5"] * 100
    pairs = [(seen[(False,) + k[1:]] - seen[k]) for k in seen if k[0] and (False,) + k[1:] in seen]
    d = np.array(pairs)
    lo, hi = t_ci(d)
    say(f"IT2-T1 pooled n={len(d)} d={d.mean():+.3f} CI=[{lo:+.2f},{hi:+.2f}] p={perm_p(d):.3f} wins={(d>0).sum()}")


# =========================================================================== #
# Figure panels (JSON for matlab_fig_1)
# =========================================================================== #
def fig_hetero():
    recs = load("noniid")
    rules = ["fedavg", "fedavg_clip", "median", "multikrum", "rfa", "fltrust", "klcos", "fedhift"]
    A = cell(recs, lambda r: (C(r, "aggregator"), C(r, "alpha")))
    U = cell(recs, lambda r: (C(r, "aggregator"), C(r, "alpha")), "benign_jain", 1.0)
    al = sorted({C(r, "alpha") for r in recs})
    s1, s2 = [], []
    for m in rules:
        mu = [np.mean(A[(m, x)]) for x in al]
        se = [1.96 * np.std(A[(m, x)], ddof=1) / np.sqrt(len(A[(m, x)])) for x in al]
        s1.append({**series(m, al, mu), "kind": "errorbar", "err": se})
        s2.append(series(m, al, [1 - np.mean(U[(m, x)]) for x in al]))
    common = dict(layout="col", width_in=2.55, height_in=1.95, font_pt=7.5, legend_pt=5.5,
                  xscale="log", xlim=[0.04, 1.25], xticks=[0.05, 0.1, 0.3, 1],
                  xticklabels=["0.05", "0.1", "0.3", "1"],
                  xlabel="Dirichlet concentration \\alpha")
    dump({"name": "hetero_acc", **common, "ylabel": "Accuracy (%)", "ylim": [20, 92],
          "legend": "southeast", "legend_cols": 2, "lock_ylim": True, "series": s1})
    dump({"name": "hetero_unf", **common, "ylabel": "Unfairness 1 - {\\itJ}", "yscale": "log",
          "ylim": [0.002, 1.0], "legend": "none", "series": [{**s, "label": ""} for s in s2]})
    for x in al:
        say("HETERO alpha", x, {m: round(float(np.mean(A[(m, x)])), 1) for m in rules})


def fig_trustfn():
    recs = load("trustfn")
    if not recs:
        return
    order = ["tr_cosine", "tr_linear", "tr_linopt", "tr_logistic", "tr_mlp", "tr_type1", "fedhift"]
    atks = ["label_flip", "sign_flip", "scaling", "alie", "ipm", "backdoor"]
    A = cell([r for r in recs if C(r, "alpha") == 0.5], lambda r: (C(r, "aggregator"), C(r, "attack")))
    cols = [PAL["green"], PAL["grey"], PAL["amber"], PAL["purple"], PAL["pink"], PAL["sky"], PAL["blue"]]
    ser = []
    for i, m in enumerate(order):
        ser.append({"kind": "bar_group", "label": NAMES[m], "x": list(range(1, len(atks) + 1)),
                    "y": [float(np.mean(A[(m, x)])) for x in atks], "color": cols[i]})
    dump({"name": "trustfn_bars", "layout": "col", "width_in": 5.1, "height_in": 2.1,
          "font_pt": 7.5, "legend_pt": 6, "xlim": [0.5, len(atks) + 0.5], "ylim": [60, 92],
          "xticks": list(range(1, len(atks) + 1)), "xticklabels": [ATK[a] for a in atks],
          "ylabel": "Accuracy (%)", "legend": "north", "legend_cols": 7, "lock_ylim": True,
          "series": ser})


def _theory_rows(suites, filt=None):
    rows = []
    for su in suites:
        for r in load(su):
            if "theory" not in r or (filt and not filt(r)):
                continue
            for t in r["theory"]:
                rows.append({**t, "T": C(r, "temperature"), "alpha": C(r, "alpha"),
                             "beta": C(r, "byz_frac"), "attack": C(r, "attack"), "suite": su})
    return rows


def fig_theory():
    rows = _theory_rows(["theory", "temp", "byzsweep"])
    if not rows:
        return
    rows = [t for t in rows if t["round"] > 3 and "bias" in t]
    # (a) Corollary 1: measured W_B against the uniform-margin bound (rounds with delta>0)
    pos = [t for t in rows if t["delta_unif"] > 0]
    x = [t["bound_mass"] for t in pos]; y = [max(t["W_B"], 1e-6) for t in pos]
    viol = sum(1 for a, b in zip(x, y) if b > a + 1e-9)
    say(f"THEORY cor1: rounds with delta>0: {len(pos)}/{len(rows)}, violations {viol}, "
        f"median bound/measured ratio {np.median(np.array(x)/np.array(y)):.2f}")
    mf = [(t["mf_mass"], t["W_B"]) for t in rows]
    mae = np.mean([abs(a - b) for a, b in mf])
    say(f"THEORY mean-field per-round MAE {mae:.4f} over {len(mf)} rounds")
    # per-run means for mean-field
    runs = defaultdict(list)
    for t in rows:
        runs[(t["suite"], t["T"], t["alpha"], t["beta"], t["attack"])].append(t)
    pm = [(np.mean([t["mf_mass"] for t in v]), np.mean([t["W_B"] for t in v])) for v in runs.values()]
    mae_run = np.mean([abs(a - b) for a, b in pm])
    say(f"THEORY mean-field per-run MAE {mae_run:.4f} over {len(pm)} runs; max {max(abs(a-b) for a,b in pm):.4f}")
    lim = [1e-5, 2]
    dump({"name": "theory_mass", "layout": "col", "width_in": 2.55, "height_in": 1.95, "font_pt": 7.5,
          "legend_pt": 6, "xscale": "log", "yscale": "log", "xlim": lim, "ylim": lim,
          "xticks": [1e-4, 1e-2, 1], "yticks": [1e-4, 1e-2, 1],
          "xticklabels": ["10^{-4}", "10^{-2}", "1"], "yticklabels": ["10^{-4}", "10^{-2}", "1"],
          "xlabel": "Predicted {\\itW}_B", "ylabel": "Measured {\\itW}_B", "legend": "northwest",
          "lock_ylim": True,
          "series": [{"kind": "scatter", "label": "Corollary 1 bound", "x": x, "y": y,
                      "color": PAL["sky"], "marker": "o", "open": True},
                     {"kind": "scatter", "label": "Mean-field (per run)", "x": [max(a, 1e-6) for a, b in pm],
                      "y": [max(b, 1e-6) for a, b in pm], "color": PAL["verm"], "marker": "s"},
                     {"kind": "ref", "x": lim, "y": lim, "label": "Equality", "line": "-"}]})
    # (b) Theorem 2: aggregation bias against 2 nu M W_B
    bx = [2 * t["M"] * t["W_B"] for t in rows if t["W_B"] > 1e-6]
    by = [t["bias"] for t in rows if t["W_B"] > 1e-6]
    r_ = np.array(by) / np.array(bx)
    say(f"THEORY thm2: n={len(bx)} max ratio bias/bound {r_.max():.3f} median {np.median(r_):.3f}")
    lim2 = [1e-4, 3]
    dump({"name": "theory_bias", "layout": "col", "width_in": 2.55, "height_in": 1.95, "font_pt": 7.5,
          "legend_pt": 6, "xscale": "log", "yscale": "log", "xlim": lim2, "ylim": lim2,
          "xticks": [1e-3, 1e-2, 1e-1, 1], "yticks": [1e-3, 1e-2, 1e-1, 1],
          "xticklabels": ["10^{-3}", "10^{-2}", "10^{-1}", "1"], "yticklabels": ["10^{-3}", "10^{-2}", "10^{-1}", "1"],
          "xlabel": "Bound 2\\nu{\\itM_t}{\\itW}_B", "ylabel": "Measured ||\\Delta - \\Delta^{\\ast}||",
          "legend": "northwest", "lock_ylim": True,
          "series": [{"kind": "scatter", "label": "Rounds", "x": bx, "y": by, "color": PAL["blue"],
                      "marker": "o", "open": True},
                     {"kind": "ref", "x": lim2, "y": lim2, "label": "Theorem 2", "line": "-"}]})


def fig_byzsweep():
    recs = load("byzsweep")
    if not recs:
        return
    rules = ["fedavg_clip", "flame", "klcos", "fedhift"]
    A = cell(recs, lambda r: (C(r, "aggregator"), C(r, "byz_frac")))
    W = cell(recs, lambda r: (C(r, "aggregator"), C(r, "byz_frac")), "mal_weight_mass", 1.0)
    P = cell(recs, lambda r: (C(r, "aggregator"), C(r, "byz_frac")), "prior_mass", 1.0)
    bs = sorted({C(r, "byz_frac") for r in recs})
    s1 = []
    for m in rules:
        xs = [b for b in bs if (m, b) in A]
        s1.append({**series(m, [100 * b for b in xs], [np.mean(A[(m, b)]) for b in xs]),
                   "kind": "errorbar", "err": [float(np.std(A[(m, b)])) for b in xs]})
        say("BYZSWEEP acc", m, {b: round(float(np.mean(A[(m, b)])), 1) for b in xs})
    dump({"name": "byz_acc", "layout": "col", "width_in": 2.55, "height_in": 1.95, "font_pt": 7.5,
          "legend_pt": 6, "xlabel": "Byzantine fraction \\beta (%)", "ylabel": "Accuracy (%)",
          "xlim": [3, 42], "xticks": [5, 10, 20, 30, 40], "legend": "southwest", "series": s1,
          "lock_ylim": True, "ylim": [10, 92]})
    # W_B, prior mass and mean-field prediction for FedEFT, plus the trust margin
    rows = _theory_rows(["byzsweep"])
    mfp = defaultdict(list); dm = defaultdict(list)
    for t in rows:
        if t["round"] > 3 and "mf_mass" in t:
            mfp[t["beta"]].append(t["mf_mass"]); dm[t["beta"]].append(t["delta_mean"])
    xs = [b for b in bs if ("fedhift", b) in W]
    s2 = [series("fedhift", [100 * b for b in xs], [np.mean(W[("fedhift", b)]) for b in xs], label="FedEFT {\\itW}_B"),
          {"label": "Mean-field prediction", "x": [100 * b for b in xs], "y": [float(np.mean(mfp[b])) for b in xs],
           "color": PAL["verm"], "marker": "s", "line": "--", "open": True},
          {"label": "Size prior \\Pi_B", "x": [100 * b for b in xs], "y": [np.mean(P[("fedhift", b)]) for b in xs],
           "color": PAL["grey"], "marker": "v", "line": ":"},
          {"label": "Trust margin \\delta", "x": [100 * b for b in xs], "y": [float(np.mean(dm[b])) for b in xs],
           "color": PAL["green"], "marker": "d", "line": "-."}]
    dump({"name": "byz_mass", "layout": "col", "width_in": 2.55, "height_in": 1.95, "font_pt": 7.5,
          "legend_pt": 6, "xlabel": "Byzantine fraction \\beta (%)", "ylabel": "Weight mass / margin",
          "xlim": [3, 42], "xticks": [5, 10, 20, 30, 40], "ylim": [0, 0.75], "legend": "northwest",
          "lock_ylim": True, "series": s2})
    say("BYZSWEEP W_B fedhift", {b: round(float(np.mean(W[("fedhift", b)])), 3) for b in xs},
        "prior", {b: round(float(np.mean(P[("fedhift", b)])), 3) for b in xs},
        "mf", {b: round(float(np.mean(mfp[b])), 3) for b in xs}, "delta", {b: round(float(np.mean(dm[b])), 3) for b in xs})


def fig_temp():
    recs = load("temp")
    if not recs:
        return
    rows = _theory_rows(["temp"])
    Ts = sorted({C(r, "temperature") for r in recs})
    out = {}
    for cond, (atk, al) in {"a05": ("sign_flip", 0.5), "a01": ("sign_flip", 0.1)}.items():
        sub = [r for r in recs if C(r, "alpha") == al and C(r, "attack") == atk]
        A = cell(sub, lambda r: C(r, "temperature"))
        W = cell(sub, lambda r: C(r, "temperature"), "mal_weight_mass", 1.0)
        J = cell(sub, lambda r: C(r, "temperature"), "benign_worst10")
        dist, byz = defaultdict(list), defaultdict(list)
        for t in rows:
            if t["alpha"] == al and t["attack"] == atk and t["round"] > 3 and "delta_H" in t:
                dist[t["T"]].append((np.exp(t["delta_H"] / t["T"]) - 1) ** 2)
                byz[t["T"]].append((t["M"] * t["W_B"]) ** 2)
        out[cond] = (A, W, J, dist, byz)
        say("TEMP", cond, {T: (round(float(np.mean(A[T])), 1), round(float(np.mean(W[T])), 4),
                              round(float(np.mean(J[T])), 1), round(float(np.median(dist[T])), 3),
                              round(float(np.mean(byz[T])), 5)) for T in Ts if T in A})
    common = dict(layout="col", width_in=2.55, height_in=1.95, font_pt=7.5, legend_pt=6,
                  xscale="log", xlim=[0.04, 5], xticks=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 4],
                  xticklabels=[".05", ".1", ".2", ".4", ".8", "1.6", "4"], xlabel="Temperature {\\itT}")
    s = []
    for cond, lab, col, mk in [("a05", "\\alpha = 0.5", PAL["blue"], "o"), ("a01", "\\alpha = 0.1", PAL["verm"], "s")]:
        A, W, J, dist, byz = out[cond]
        xs = [T for T in Ts if T in A]
        s.append({"kind": "errorbar", "label": "Acc., " + lab, "x": xs, "y": [float(np.mean(A[T])) for T in xs],
                  "err": [float(np.std(A[T])) for T in xs], "color": col, "marker": mk, "line": "-"})
        s.append({"label": "Worst 10%, " + lab, "x": xs, "y": [float(np.mean(J[T])) for T in xs],
                  "color": col, "marker": mk, "line": "--", "open": True})
    dump({"name": "temp_acc", **common, "ylabel": "Accuracy (%)", "legend": "north", "legend_cols": 1,
          "ylim": [0, 160], "yticks": [0, 20, 40, 60, 80, 100], "lock_ylim": True, "series": s})
    s = []
    for cond, lab, col, mk in [("a05", "\\alpha = 0.5", PAL["blue"], "o"), ("a01", "\\alpha = 0.1", PAL["verm"], "s")]:
        A, W, J, dist, byz = out[cond]
        xs = [T for T in Ts if T in A]
        s.append({"label": "(e^{\\delta_H/T}-1)^2, " + lab, "x": xs, "y": [float(np.median(dist[T])) for T in xs],
                  "color": col, "marker": mk, "line": "-"})
        s.append({"label": "(M_tW_B)^2, " + lab, "x": xs, "y": [max(float(np.mean(byz[T])), 1e-7) for T in xs],
                  "color": col, "marker": mk, "line": "--", "open": True})
    dump({"name": "temp_terms", **common, "ylabel": "Penalty factor", "yscale": "log", "legend": "north",
          "ylim": [1e-6, 1e9], "yticks": [1e-6, 1e-3, 1, 1e3, 1e6],
          "yticklabels": ["10^{-6}", "10^{-3}", "1", "10^{3}", "10^{6}"], "lock_ylim": True, "series": s})


def fig_longhorizon():
    recs = load("longhorizon")
    if not recs:
        return
    rules = ["fedavg", "fedavg_clip", "rfa", "fedhift"]
    get = lambda m, atk, rho=None: [x for x in recs if C(x, "aggregator") == m and C(x, "attack") == atk
                                    and (rho is None or C(x, "attack_rho") == rho)]
    s1, s2 = [], []
    for m in rules:
        c = get(m, "none")[0]["history"]
        for rho in (0.7, 0.0):
            h = get(m, "adaptive", rho)[0]["history"]
            say("LONGH", m, rho, "acc last3", round(100 * np.mean(h["acc"][-3:]), 1),
                "clean", round(100 * np.mean(c["acc"][-3:]), 1), "proj end", round(h["proj"][-1], 1),
                "clean proj", round(c["proj"][-1], 2), "drift", round(h["drift"][-1], 1),
                "clean drift", round(c["drift"][-1], 1))
        h = get(m, "adaptive", 0.0)[0]["history"]
        s1.append(series(m, h["round"], [100 * a for a in h["acc"]]))
        s2.append(series(m, h["round"], h["proj"]))
    c = get("fedhift", "none")[0]["history"]
    s1.append({"label": "FedEFT, no attack", "x": c["round"], "y": [100 * a for a in c["acc"]],
               "color": PAL["blue"], "marker": "none", "line": ":"})
    h = get("fedhift", "adaptive", 0.7)[0]["history"]
    s2.append({"label": "FedEFT, \\gamma = 0.7", "x": h["round"], "y": h["proj"], "color": PAL["blue"],
               "marker": "o", "line": ":", "open": True})
    for r in get("fedhift", "adaptive"):
        th = r.get("theory", [])
        if th:
            say("LONGH fedhift rho", C(r, "attack_rho"), "mean W_B", round(float(np.mean([t["W_B"] for t in th])), 3),
                "Pi_B", round(float(np.mean([t["Pi_B"] for t in th])), 3), "det_auc", round(r["det_auc"], 3))
    common = dict(layout="col", width_in=2.55, height_in=1.95, font_pt=7.5, legend_pt=6, xlim=[0, 205],
                  xlabel="Round")
    dump({"name": "lh_acc", **common, "ylabel": "Accuracy (%)", "ylim": [70, 90], "legend": "southeast",
          "lock_ylim": True, "series": s1})
    dump({"name": "lh_proj", **common, "ylabel": "Drift along attack direction", "legend": "northwest",
          "series": s2})


def fig_ruleperturb():
    recs = load("ruleperturb")
    if not recs:
        return
    eps = sorted({C(r, "rule_perturb") for r in recs})
    s1, s2 = [], []
    cols = {"label_flip": (PAL["blue"], "o"), "sign_flip": (PAL["verm"], "s"), "scaling": (PAL["green"], "d")}
    for atk in ["label_flip", "sign_flip", "scaling"]:
        A = cell([r for r in recs if C(r, "attack") == atk], lambda r: C(r, "rule_perturb"))
        W = cell([r for r in recs if C(r, "attack") == atk], lambda r: C(r, "rule_perturb"), "mal_weight_mass", 1.0)
        c, mk = cols[atk]
        s1.append({"kind": "errorbar", "label": ATK[atk], "x": [100 * e for e in eps], "y": [float(np.mean(A[e])) for e in eps],
                   "err_lo": [float(np.mean(A[e]) - np.min(A[e])) for e in eps],
                   "err_hi": [float(np.max(A[e]) - np.mean(A[e])) for e in eps], "color": c, "marker": mk})
        s2.append({"kind": "errorbar", "label": ATK[atk], "x": [100 * e for e in eps], "y": [float(np.mean(W[e])) for e in eps],
                   "err_lo": [float(np.mean(W[e]) - np.min(W[e])) for e in eps],
                   "err_hi": [float(np.max(W[e]) - np.mean(W[e])) for e in eps], "color": c, "marker": mk})
        say("RULEPERT", atk, {e: (round(float(np.mean(A[e])), 2), round(float(np.min(A[e])), 2), round(float(np.max(A[e])), 2),
                                  round(float(np.mean(W[e])), 4)) for e in eps})
    common = dict(layout="col", width_in=2.55, height_in=1.95, font_pt=7.5, legend_pt=6,
                  xlim=[-2, 22], xticks=[0, 5, 10, 20], xlabel="Consequent perturbation (%)")
    dump({"name": "rule_acc", **common, "ylabel": "Accuracy (%)", "legend": "south", "series": s1})
    dump({"name": "rule_wb", **common, "ylabel": "Byzantine mass {\\itW}_B", "legend": "north", "series": s2})


def fig_qskew():
    recs = load("qskew")
    if not recs:
        return
    rules = ["fedavg", "fedavg_clip", "rfa", "flame", "fedhift"]
    for sq in [0.5, 1.5]:
        for atk in ["none", "sign_flip"]:
            sub = [r for r in recs if C(r, "sigma_q") == sq and C(r, "attack") == atk]
            A = cell(sub, lambda r: C(r, "aggregator")); W10 = cell(sub, lambda r: C(r, "aggregator"), "benign_worst10")
            say("QSKEW", sq, atk, {m: (round(float(np.mean(A[m])), 1), round(float(np.mean(W10[m])), 1)) for m in rules if m in A})
    # w/pi against client size for honest clients (FedEFT, sigma_q = 1.5)
    pts = defaultdict(list)
    for r in recs:
        if C(r, "aggregator") != "fedhift" or C(r, "sigma_q") != 1.5:
            continue
        mal = set(r["malicious"])
        for rec in r["weight_log"]:
            pi = np.array(rec["sizes"]) / np.sum(rec["sizes"])
            for k, w, p, lab in zip(rec["clients"], rec["w"], pi, rec["labels"]):
                if lab == 1:
                    pts[(C(r, "attack"), C(r, "seed"), k)].append(w / p)
    sizes = {}
    for r in recs:
        if C(r, "sigma_q") == 1.5:
            for k, n in enumerate(r["client_sizes"]):
                sizes[(C(r, "attack"), C(r, "seed"), k)] = n
    ser = []
    for atk, col, mk in [("none", PAL["blue"], "o"), ("sign_flip", PAL["verm"], "s")]:
        ks = [k for k in pts if k[0] == atk]
        x = [sizes[k] for k in ks]; y = [float(np.mean(pts[k])) for k in ks]
        ser.append({"kind": "scatter", "label": "No attack" if atk == "none" else "Sign flip", "x": x, "y": y,
                    "color": col, "marker": mk, "open": atk == "none"})
        lo = np.array(y)
        xs_, ys_ = np.array(x), np.array(y)
        q1, q2 = np.quantile(xs_, [1 / 3, 2 / 3])
        say("QSKEW terciles", atk, "small", round(float(ys_[xs_ <= q1].mean()), 2), "mid",
            round(float(ys_[(xs_ > q1) & (xs_ <= q2)].mean()), 2), "large", round(float(ys_[xs_ > q2].mean()), 2),
            "cuts", float(q1), float(q2))
        say("QSKEW w/pi", atk, "n", len(y), "min", round(float(lo.min()), 3), "max", round(float(lo.max()), 3),
            "corr(log n, w/pi)", round(float(np.corrcoef(np.log(x), y)[0, 1]), 3))
    ser.append({"kind": "hline", "y": 1.0, "line": ":", "label": "FedAvg share"})
    dump({"name": "qskew_ratio", "layout": "col", "width_in": 2.55, "height_in": 1.95, "font_pt": 7.5, "legend_pt": 6,
          "xscale": "log", "xlabel": "Client size {\\itn_k}", "ylabel": "Relative share {\\itw_k}/\\pi_k",
          "yscale": "log", "ylim": [0.4, 14], "yticks": [0.5, 1, 2, 4], "yticklabels": ["0.5", "1", "2", "4"],
          "legend": "northeast", "legend_cols": 1, "lock_ylim": True, "series": ser})


def fig_churn():
    recs = load("churn")
    if not recs:
        return
    key = lambda r: (C(r, "aggregator") + ("_norep" if C(r, "tag") == "norep" else ""), C(r, "dropout"))
    A = cell(recs, key); W = cell(recs, key, "mal_weight_mass", 1.0)
    drs = sorted({C(r, "dropout") for r in recs})
    rules = ["fedavg_clip", "flame", "klcos", "fedhift", "fedhift_norep"]
    ser = []
    for i, m in enumerate(rules):
        base = m.replace("_norep", "")
        st = dict(STYLE[base])
        if m == "fedhift_norep":
            st.update(label="FedEFT, no reputation", color=PAL["sky"], proposed=False, marker="^", line=":")
        ser.append({**st, "kind": "errorbar", "x": [100 * d for d in drs],
                    "y": [float(np.mean(A[(m, d)])) for d in drs],
                    "err": [float(np.std(A[(m, d)])) for d in drs]})
        say("CHURN", m, {d: (round(float(np.mean(A[(m, d)])), 1), round(float(np.mean(W[(m, d)])), 3)) for d in drs})
    dump({"name": "churn_acc", "layout": "col", "width_in": 2.55, "height_in": 1.95, "font_pt": 7.5, "legend_pt": 5.5,
          "xlim": [-5, 65], "xticks": [0, 30, 60], "xlabel": "Unavailable clients (%)",
          "ylabel": "Accuracy (%)", "ylim": [76, 98], "yticks": [76, 80, 84, 88], "legend": "north", "legend_cols": 1, "lock_ylim": True,
          "series": ser})


def fig_ablation():
    recs = load("ablation") + [r for r in load("ablation_ipm") if C(r, "aggregator") == "fedhift"]
    recs = [r for r in recs if not str(C(r, "tag", "")).endswith("_a005")]
    names = ["wo_align", "wo_peer", "wo_norm", "wo_stab", "type1", "no_rep", "no_prior", "no_clip"]
    lbl = ["{\\itu}_1", "{\\itu}_2", "{\\itu}_3", "{\\itu}_4", "T1", "rep", "prior", "clip"]
    atks = [("label_flip", PAL["blue"]), ("sign_flip", PAL["verm"]), ("ipm", PAL["green"])]
    A = cell(recs, lambda r: (C(r, "tag"), C(r, "attack")))
    W = cell(recs, lambda r: (C(r, "tag"), C(r, "attack")), "mal_weight_mass", 1.0)
    s1, s2 = [], []
    for i, (a, col) in enumerate(atks):
        x = [j + 1 + (i - 1) * 0.26 for j in range(len(names))]
        s1.append({"kind": "bar", "label": ATK[a], "x": x, "width": 0.25, "color": col,
                   "y": [float(np.mean(A[(n, a)]) - np.mean(A[("full", a)])) for n in names]})
        s2.append({"kind": "bar", "label": ATK[a], "x": x, "width": 0.25, "color": col,
                   "y": [float(np.mean(W[(n, a)]) - np.mean(W[("full", a)])) for n in names]})
        say("ABL", a, "full acc", round(float(np.mean(A[("full", a)])), 2), "W", round(float(np.mean(W[("full", a)])), 3),
            {n: (round(float(np.mean(A[(n, a)]) - np.mean(A[("full", a)])), 2), round(float(np.mean(W[(n, a)])), 3)) for n in names})
    common = dict(layout="col", width_in=2.55, height_in=1.95, font_pt=7.5, legend_pt=6,
                  xlim=[0.4, len(names) + 0.6], xticks=list(range(1, len(names) + 1)), xticklabels=lbl,
                  xlabel="Component removed or replaced")
    dump({"name": "abl_acc", **common, "ylabel": "\\Delta accuracy (points)", "ylim": [-4, 1.5],
          "legend": "southwest", "lock_ylim": True, "series": s1 + [{"kind": "hline", "y": 0, "line": "-", "color": [0, 0, 0]}]})
    dump({"name": "abl_wb", **common, "ylabel": "\\Delta {\\itW}_B", "ylim": [-0.05, 0.2],
          "legend": "northeast", "lock_ylim": True, "series": s2 + [{"kind": "hline", "y": 0, "line": "-", "color": [0, 0, 0]}]})


def fig_rho():
    recs = load("adaptive_rho")
    acc = cell(recs, lambda r: (C(r, "aggregator"), C(r, "attack_rho")))
    auc = cell(recs, lambda r: (C(r, "aggregator"), C(r, "attack_rho")), "det_auc", 1.0)
    wb = cell(recs, lambda r: (C(r, "aggregator"), C(r, "attack_rho")), "mal_weight_mass", 1.0)
    rhos = sorted({k[1] for k in acc})
    s1 = []
    for m in ["fedavg", "rfa", "fedhift"]:
        xs = [r for r in rhos if (m, r) in acc]
        s1.append({**series(m, xs, [np.mean(acc[(m, r)]) for r in xs]), "kind": "errorbar",
                   "err": [np.std(acc[(m, r)]) for r in xs]})
    common = dict(layout="col", width_in=2.55, height_in=1.95, font_pt=7.5, legend_pt=6,
                  xlim=[-0.05, 0.95], xticks=[0, 0.3, 0.5, 0.7, 0.9], xlabel="Alignment budget \\gamma")
    dump({"name": "rho_acc", **common, "ylabel": "Accuracy (%)", "ylim": [80, 88], "legend": "south",
          "lock_ylim": True, "series": s1 + [{"kind": "hline", "y": 85.1, "line": ":", "label": "FedAvg, no attack"}]})
    xs = [r for r in rhos if ("fedhift", r) in auc]
    s2 = [{"kind": "errorbar", "label": "Detection AUROC", "x": xs, "y": [np.mean(auc[("fedhift", r)]) for r in xs],
           "err": [np.std(auc[("fedhift", r)]) for r in xs], "color": PAL["blue"], "marker": "o"},
          {"kind": "errorbar", "label": "{\\itW}_B", "x": xs, "y": [np.mean(wb[("fedhift", r)]) for r in xs],
           "err": [np.std(wb[("fedhift", r)]) for r in xs], "color": PAL["verm"], "marker": "s", "line": "--"},
          {"kind": "hline", "y": 0.5, "line": ":", "label": "Chance AUROC"},
          {"kind": "hline", "y": 0.18, "line": "--", "label": "Size prior \\Pi_B", "color": [0.6, 0.6, 0.6]}]
    dump({"name": "rho_det", **common, "ylabel": "FedEFT detection", "ylim": [0, 0.75], "legend": "north",
          "legend_cols": 2, "lock_ylim": True, "series": s2})


def fig_scaling():
    d = json.load(open(os.path.join(RES, "scaling.json")))
    ps = np.asarray(d["p"], float)
    base = np.asarray(d["ms"]["fedavg"], float)
    ser = [series(m, ps, np.asarray(d["ms"][m], float) / base) for m in ["median", "trimmed_mean", "multikrum", "rfa", "fedhift"]]
    dump({"name": "cost_scaling", "layout": "col", "width_in": 2.55, "height_in": 1.95, "font_pt": 7.5, "legend_pt": 6,
          "xscale": "log", "yscale": "log", "xlabel": "Model dimension {\\itp}", "ylabel": "Cost relative to FedAvg",
          "xticks": [1e4, 1e5, 1e6], "legend": "northeast", "legend_cols": 2, "series": ser + [{"kind": "hline", "y": 1, "line": ":"}]})
    say("SCALING", {m: [round(v, 1) for v in (np.asarray(d["ms"][m], float) / base)] for m in d["ms"]}, list(ps))


def trustfn_margins():
    """Uniform trust margin and honest distortion factor per trust function."""
    d = defaultdict(lambda: ([], []))
    for r in load("trustfn"):
        if C(r, "alpha") != 0.5 or C(r, "attack") not in ("label_flip", "sign_flip", "scaling"):
            continue
        for rec in r["weight_log"]:
            t = np.array(rec["tau"]); l = np.array(rec["labels"])
            if (l == 0).any() and (l == 1).sum() > 1:
                d[C(r, "aggregator")][0].append(t[l == 1].min() - t[l == 0].max())
                d[C(r, "aggregator")][1].append(t[l == 1].max() - t[l == 1].min())
    for k, (a, b) in d.items():
        say("TRUSTMARGIN", k, "delta", round(float(np.mean(a)), 3), "delta_H", round(float(np.mean(b)), 3),
            "distortion e^(dH/T)", round(float(np.mean(np.exp(np.array(b) / 0.2))), 1))


def fig_alloc():
    """Closed-form allocation on a real cohort: relative share against reputation."""
    r = [x for x in load("theory") if C(x, "attack") == "sign_flip" and C(x, "alpha") == 0.5][0]
    rec = [w for w in r["weight_log"] if w["labels"].count(0) == 2 and w["round"] > 5][0]
    rep = np.array(rec["tau"]); lab = np.array(rec["labels"]); pi = np.array(rec["sizes"]) / np.sum(rec["sizes"])
    o = np.argsort(rep)
    ser = []
    for T, col, mk, ls in [(0.8, PAL["grey"], "v", ":"), (0.4, PAL["green"], "d", "-."),
                           (0.2, PAL["blue"], "o", "-"), (0.1, PAL["verm"], "s", "--")]:
        w = pi * np.exp(rep / T); w /= w.sum()
        ser.append({"label": "{\\itT} = %g" % T, "x": rep[o].tolist(), "y": (w / pi)[o].tolist(),
                    "color": col, "marker": mk, "line": ls, "proposed": T == 0.2})
    ser.append({"kind": "hline", "y": 1.0, "line": ":", "label": "FedAvg ({\\itT} \\rightarrow \\infty)"})
    byz = rep[lab == 0]
    ann = [{"x": float(byz.mean()), "y": 1.2, "text": "Byzantine", "italic": True},
           {"x": 0.72, "y": 0.25, "text": "honest", "italic": True}]
    ser.append({"kind": "vline", "x": float((byz.max() + rep[lab == 1].min()) / 2)})
    dump({"name": "alloc_tilt", "layout": "col", "width_in": 2.9, "height_in": 1.95, "font_pt": 7.5, "legend_pt": 6,
          "xlabel": "Reputation {\\itr_k}", "ylabel": "Relative share {\\itw_k}/\\pi_k", "ylim": [0, 3],
          "legend": "auto", "lock_ylim": True, "series": ser, "annotations": ann})
    say("ALLOC round", rec["round"], "rep", np.round(rep[o], 3).tolist(), "labels", lab[o].tolist())


def main():
    for f in (trustfn_margins, tab_fmnist, tab_cifar, tab_femnist, tab_stats, tab_trustfn, tab_backdoor, tab_fou,
              fig_hetero, fig_theory, fig_byzsweep, fig_temp, fig_longhorizon,
              fig_ruleperturb, fig_qskew, fig_churn, fig_ablation, fig_rho, fig_scaling, fig_alloc):
        try:
            f()
        except Exception as e:  # keep going so partial campaigns still render
            import traceback
            print("!!", f.__name__, e)
            traceback.print_exc()
    NUM.close()


if __name__ == "__main__":
    main()
