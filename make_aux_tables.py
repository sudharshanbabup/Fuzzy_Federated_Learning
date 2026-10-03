"""Six auxiliary tables of the manuscript (outcome fairness, adaptive adversary,
server cost, type-2 vs type-1, hyper-parameter sensitivity, sketch error).

The numbers are computed by the table functions in make_sn.py and
make_tables_extra.py (and read from results/sketch_error.json); this script
only applies the manuscript's captions, rule names and booktabs rules, and
writes the result to tables_fodm/.

    python make_aux_tables.py
"""
import json
import os
import re
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tables_fodm")
os.makedirs(OUT, exist_ok=True)

CAPTION = {
    "tab_cost": r"Server aggregation cost at the Fashion-MNIST model size (ms per round and multiples of FedAvg) and relative cost at three synthetic model dimensions.",
    "tab_fair_balanced": r"Outcome fairness on Fashion-MNIST measured two ways, averaged over the seven attacks and five seeds.",
    "tab_sens": r"Hyper-parameter sensitivity on Fashion-MNIST: accuracy, Jain index and Byzantine weight mass $W_{\mathcal{B}}$.",
    "tab_sketch": r"Error of the sketched cosine against exact computation over twenty rounds of a sign-flipping run at $\alpha=0.5$.",
    "tab_fou": r"Interval type-2 engine against its type-1 reduction under label flipping (accuracy \%, mean $\pm$ s.d.).",
    "tab_adaptive": r"Robustness to the white-box adaptive adversary on Fashion-MNIST at three alignment budgets (trust-aware $\gamma=0.9$, clip-aware $\gamma=0$, combined $\gamma=0.7$).",
}


def _set_caption(s, name):
    i = s.index(r"\caption{")
    j = s.index(r"\label{", i)
    return s[:i] + r"\caption{" + CAPTION[name] + "}\n" + s[j:]


def _normalise(s, name):
    s = _set_caption(s, name)
    s = re.sub(r"\\begin\{table\}\[[^\]]*\]", r"\\begin{table}[!tbp]", s)
    s = s.replace("Coord. median", "Median").replace("Trimmed mean", "Trim. mean")
    s = s.replace(" (proposed)", " ").replace(" (ours)", " ").replace(r"\varrho", r"\gamma")
    if r"\hline" in s:                      # plain rules -> booktabs
        n = s.count(r"\hline")
        parts = s.split(r"\hline")
        out = parts[0]
        for k, p in enumerate(parts[1:], 1):
            out += (r"\toprule" if k == 1 else r"\botrule" if k == n else r"\midrule") + p
        s = out
    return s


def _write(name, s):
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write(s if s.endswith("\n") else s + "\n")
    print("  wrote", name + ".tex")


def sketch_table():
    d = json.load(open(os.path.join(HERE, "results", "sketch_error.json")))["dims"]
    L = [r"\begin{table}[!tbp]", r"\centering", r"\caption{" + CAPTION["tab_sketch"] + "}",
         r"\label{tab:sketch}", r"\setlength{\tabcolsep}{4pt}", r"\footnotesize",
         r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
         r"& \multicolumn{3}{c}{cosine matrix} & \multicolumn{4}{c}{alignment $u_{k,1}$}\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
         r"$d$ & median & 95th pct & max & median & 95th pct & $\tau_b$ & top-half\\", r"\midrule"]
    for k in sorted(d, key=int):
        r = d[k]
        L.append("$2^{%d}$ & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.2f\\\\" % (
            round(__import__("math").log2(int(k))), r["cos_med"], r["cos_p95"], r["cos_max"],
            r["u1_med"], r["u1_p95"], r["u1_tau"], r["u1_top"]))
    L += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    _write("tab_sketch", "\n".join(L))


def main():
    import make_sn
    import make_tables_extra
    with tempfile.TemporaryDirectory() as tmp:
        fm = [r for r in make_sn.load("main_v6") if r["config"]["aggregator"] != "fedprox"]
        make_sn.tab_fair_balanced(fm, os.path.join(tmp, "tab_fair_balanced.tex"))
        make_sn.tab_adaptive(os.path.join(tmp, "tab_adaptive.tex"))
        make_sn.tab_cost(os.path.join(tmp, "tab_cost.tex"))
        make_tables_extra.TAB = tmp
        make_tables_extra.fou_table()
        make_tables_extra.sens_table()
        for name in ("tab_fair_balanced", "tab_adaptive", "tab_cost", "tab_fou", "tab_sens"):
            _write(name, _normalise(open(os.path.join(tmp, name + ".tex")).read(), name))
    sketch_table()


if __name__ == "__main__":
    main()
