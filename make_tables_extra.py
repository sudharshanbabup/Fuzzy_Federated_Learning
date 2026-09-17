"""Generate the two hand-laid tables (FOU comparison, sensitivity) from results.

These two are tables rather than figures purely for page economy; the underlying
numbers are the same ones make_figures.py plots.
"""
import glob, json, os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "tables")
os.makedirs(TAB, exist_ok=True)


def _load(suite):
    return [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "results", suite, "*.json"))]


def fou_table():
    d, t = defaultdict(list), defaultdict(list)
    for r in _load("fou"):
        c = r["config"]
        d[(c["type1"], c["alpha"])].append(r["acc_last5"] * 100)
    for r in _load("fou_temp"):
        c = r["config"]
        t[(c["type1"], c["temperature"], c["alpha"])].append(r["acc_last5"] * 100)
    als = sorted({k[1] for k in d})
    Ts = sorted({k[1] for k in t})
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Interval type-2 engine against its type-1 reduction under label "
         r"flipping (mean $\pm$ s.d.\ of global accuracy, \%). Top: across the Dirichlet "
         r"concentration at $T=0.2$, four seeds. Bottom: across the entropic temperature "
         r"at $\alpha=0.05$, three seeds. No difference exceeds the seed spread.}",
         r"\label{tab:fou}", r"\setlength{\tabcolsep}{2.2pt}", r"\footnotesize",
         r"\begin{tabular}{lccccc}", r"\hline",
         r"$\alpha$ & " + " & ".join("%g" % a for a in als) + r"\\", r"\hline"]
    for tp, nm in [(False, "IT2 engine"), (True, "type-1 reduction")]:
        L.append(nm + " & " + " & ".join(
            "%.1f\\,$\\pm$\\,%.1f" % (np.mean(d[(tp, a)]), np.std(d[(tp, a)])) for a in als) + r"\\")
    L.append(r"$\Delta$ & " + " & ".join(
        "$%+.2f$" % (np.mean(d[(False, a)]) - np.mean(d[(True, a)])) for a in als) + r"\\")
    L += [r"\hline", r"$T$ ($\alpha=0.05$) & " + " & ".join("%g" % x for x in Ts) + " & \\\\", r"\hline"]
    for tp, nm in [(False, "IT2 engine"), (True, "type-1 reduction")]:
        L.append(nm + " & " + " & ".join(
            "%.1f\\,$\\pm$\\,%.1f" % (np.mean(t[(tp, T, 0.05)]), np.std(t[(tp, T, 0.05)]))
            for T in Ts) + " & \\\\")
    L.append(r"$\Delta$ & " + " & ".join(
        "$%+.2f$" % (np.mean(t[(False, T, 0.05)]) - np.mean(t[(True, T, 0.05)])) for T in Ts) + " & \\\\")
    L += [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "tab_fou.tex"), "w").write("\n".join(L) + "\n")


def sens_table():
    d = defaultdict(lambda: defaultdict(list))
    for r in _load("sensitivity"):
        c = r["config"]
        tag = c["tag"][0]
        key = c["temperature"] if tag == "T" else (c["kappa"] if tag == "K"
                                                   else (c["sketch_dim"] or 10 ** 6))
        d[tag][key].append((r["acc_last5"] * 100, r["benign_jain"], r["mal_weight_mass"]))

    def rows(tag):
        ks = sorted(d[tag])
        return (ks,
                [np.mean([x[0] for x in d[tag][k]]) for k in ks],
                [np.mean([x[1] for x in d[tag][k]]) for k in ks],
                [np.mean([x[2] for x in d[tag][k]]) for k in ks])

    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Hyper-parameter sensitivity on Fashion-MNIST: accuracy (\%), Jain "
         r"index and Byzantine weight mass $W_{\mathcal{B}}$; two seeds, four for the $\kappa$ block. The "
         r"temperature and sketch blocks use sign flipping at $\alpha=0.5$; the $\kappa$ "
         r"block uses label flipping at $\alpha=0.1$, where the FOU has the most to do.}",
         r"\label{tab:sens}", r"\setlength{\tabcolsep}{3.2pt}", r"\footnotesize",
         r"\begin{tabular}{lcccccc}", r"\hline"]
    ks, a, j, w = rows("T")
    L += [r"temperature $T$ & " + " & ".join("%g" % k for k in ks) + r"\\", r"\hline",
          "accuracy & " + " & ".join("%.1f" % x for x in a) + r"\\",
          "Jain & " + " & ".join("%.3f" % x for x in j) + r"\\",
          r"$W_{\mathcal{B}}$ & " + " & ".join("%.3f" % x for x in w) + r"\\", r"\hline"]
    ks, a, j, w = rows("K")
    L += [r"FOU gain $\kappa$ & " + " & ".join("%g" % k for k in ks) + " & & \\\\", r"\hline",
          "accuracy & " + " & ".join("%.1f" % x for x in a) + " & & \\\\",
          "Jain & " + " & ".join("%.3f" % x for x in j) + " & & \\\\", r"\hline"]
    ks, a, j, w = rows("S")
    L += [r"sketch $d$ & " + " & ".join([r"$2^{10}$", r"$2^{12}$", r"$2^{14}$", "exact"]) + " & & \\\\",
          r"\hline",
          "accuracy & " + " & ".join("%.1f" % x for x in a) + " & & \\\\",
          "Jain & " + " & ".join("%.3f" % x for x in j) + " & & \\\\",
          r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "tab_sens.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    fou_table()
    sens_table()
    print("wrote tab_fou.tex, tab_sens.tex")
