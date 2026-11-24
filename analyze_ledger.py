"""Ledger study: tables and evaluation metrics from run records in results_ledger/.

    python analyze_ledger.py [results_dir] [output_dir] [tables_dir]
"""
import glob
import itertools
import json
import os
import sys

import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "results_ledger"
ODIR = sys.argv[2] if len(sys.argv) > 2 else "generated"
TDIR = sys.argv[3] if len(sys.argv) > 3 else "tables"

os.makedirs(TDIR, exist_ok=True)
D = os.path.join(ODIR, "panels") if os.path.isdir(ODIR) and ODIR == "matlab" else ODIR
os.makedirs(D, exist_ok=True)

DEFAULT_PALETTE = {
    "fedhift": {"label": "FedEFT", "color": [0.0, 0.447, 0.698], "marker": "o", "line": "-", "proposed": True},
    "fedhift_inf": {"label": "FedEFT, no ledger", "color": [0.0, 0.447, 0.698], "marker": "o", "line": "-.", "open": True},
    "fedavg": {"label": "FedAvg", "color": [0.35, 0.35, 0.35], "marker": "v", "line": ":"},
    "fedavg_clip": {"label": "FedAvg+clip", "color": [0.835, 0.369, 0.0], "marker": "s", "line": "--"},
    "klcos": {"label": "KL-cos", "color": [0.0, 0.62, 0.451], "marker": "d", "line": "-."},
    "median": {"label": "Median", "color": [0.482, 0.196, 0.58], "marker": "^", "line": ":"},
    "trimmed_mean": {"label": "Trim. mean", "color": [0.8, 0.475, 0.655], "marker": ">", "line": "-."},
    "multikrum": {"label": "Multi-Krum", "color": [0.902, 0.624, 0.0], "marker": "p", "line": "--"},
    "rfa": {"label": "RFA", "color": [0.337, 0.706, 0.914], "marker": "h", "line": "-."},
    "fltrust": {"label": "FLTrust", "color": [0.55, 0.34, 0.16], "marker": "<", "line": ":"},
    "flame": {"label": "FLAME", "color": [0.0, 0.0, 0.0], "marker": "x", "line": "--"},
}
pal_path = os.path.join(ODIR, "palette.json")
if os.path.exists(pal_path):
    P = json.load(open(pal_path))
else:
    P = DEFAULT_PALETTE
NUM = {}


def load(name):
    f = os.path.join(RES, name + ".json")
    return json.load(open(f)) if os.path.exists(f) else None


def sty(rule, **kw):
    s = dict(P[rule]); s["id"] = rule; s.update(kw); return s


def perm_p(d):
    """Exact two sided sign flip permutation test of mean(d) = 0."""
    d = np.asarray(d, float); n = len(d)
    obs = abs(d.mean())
    if n <= 16:
        cnt = 0
        for signs in itertools.product([-1, 1], repeat=n):
            cnt += abs((d * signs).mean()) >= obs - 1e-12
        return cnt / 2 ** n
    rng = np.random.default_rng(0)
    s = rng.choice([-1, 1], size=(200000, n))
    return float((np.abs((s * d).mean(1)) >= obs - 1e-12).mean())


def tci(d):
    from scipy import stats
    d = np.asarray(d, float); n = len(d)
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return d.mean() - h, d.mean() + h


# ------------------------------------------------------------------ long horizon
LH = {k: (load("lhk_" + k) or load("lh_" + k)) for k in ["fedavg_clip_none", "fedavg_clip_ad0.0", "fedavg_clip_ad0.7", "fedhift_none",
                                   "fedhift_ad0.0", "fedhift_ad0.7", "fedhift_L_none", "fedhift_L_ad0.0",
                                   "fedhift_L_ad0.7", "fedhift_c1.1_ad0.0", "fedhift_c1.5_ad0.0", "fedhift_c2.0_ad0.0"]}
old_acc = json.load(open(os.path.join(MDIR, "aux", "old_acc.json")))
old_proj = json.load(open(os.path.join(MDIR, "aux", "old_proj.json")))
rd = old_acc["round"]


def h(k, key):
    return [round(100 * v, 3) if key == "acc" else round(v, 3) for v in LH[k]["history"][key]]


def coal(k):
    """per round coalition weight and prior from the weight log"""
    W, Pi = [], []
    for rec in LH[k]["weight_log"]:
        w = np.array(rec["w"]); s = np.array(rec["sizes"]); pr = s / s.sum()
        lab = np.array(rec["labels"])
        W.append(float(w[lab == 0].sum())); Pi.append(float(pr[lab == 0].sum()))
    return np.array(W), np.array(Pi)


NUM["LhAccNone"] = "%.1f" % (100 * LH["fedhift_none"]["acc_last5"])
NUM["LhAccInf"] = "%.1f" % (100 * LH["fedhift_ad0.0"]["acc_last5"])
NUM["LhAccLed"] = "%.1f" % (100 * LH["fedhift_L_ad0.0"]["acc_last5"])
NUM["LhAccClip"] = "%.1f" % (100 * LH["fedavg_clip_ad0.0"]["acc_last5"])
NUM["LhAccLedNone"] = "%.1f" % (100 * LH["fedhift_L_none"]["acc_last5"])
NUM["LhProjInf"] = "%.1f" % LH["fedhift_ad0.0"]["history"]["proj"][-1]
NUM["LhProjLed"] = "%.1f" % LH["fedhift_L_ad0.0"]["history"]["proj"][-1]
NUM["LhProjClip"] = "%.1f" % LH["fedavg_clip_ad0.0"]["history"]["proj"][-1]
NUM["LhProjRatio"] = "%.1f" % (LH["fedhift_ad0.0"]["history"]["proj"][-1] / LH["fedhift_L_ad0.0"]["history"]["proj"][-1])
NUM["LhProjLedOverClip"] = "%.2f" % (LH["fedhift_L_ad0.0"]["history"]["proj"][-1] / LH["fedavg_clip_ad0.0"]["history"]["proj"][-1])
NUM["LhProjInfSeven"] = "%.1f" % LH["fedhift_ad0.7"]["history"]["proj"][-1]
NUM["LhProjLedSeven"] = "%.1f" % LH["fedhift_L_ad0.7"]["history"]["proj"][-1]
NUM["LhProjClipSeven"] = "%.1f" % LH["fedavg_clip_ad0.7"]["history"]["proj"][-1]
NUM["LhAccInfSeven"] = "%.1f" % (100 * LH["fedhift_ad0.7"]["acc_last5"])
NUM["LhAccLedSeven"] = "%.1f" % (100 * LH["fedhift_L_ad0.7"]["acc_last5"])
NUM["LhAccClipSeven"] = "%.1f" % (100 * LH["fedavg_clip_ad0.7"]["acc_last5"])
for k, tag in [("fedhift_ad0.0", "Inf"), ("fedhift_L_ad0.0", "Led"), ("fedavg_clip_ad0.0", "Clip"),
               ("fedhift_ad0.7", "InfSeven"), ("fedhift_L_ad0.7", "LedSeven")]:
    NUM["LhWb" + tag] = "%.3f" % LH[k]["mal_weight_mass"]
    NUM["LhPi" + tag] = "%.3f" % LH[k]["prior_mass"]
WL, PL = coal("fedhift_L_ad0.0")
NUM["LhCumRatioLed"] = "%.3f" % (WL.sum() / PL.sum())
NUM["LhCumRatioMax"] = "%.3f" % max(np.cumsum(WL)[t] / np.cumsum(PL)[t] for t in range(len(WL)) if np.cumsum(PL)[t] > 0)
# per client ledger check: max over clients of W_k / P_k at the end
mx = 0.0
for k in ["fedhift_L_ad0.0", "fedhift_L_none", "fedhift_L_ad0.7"]:
    Wk, Pk = {}, {}
    for rec in LH[k]["weight_log"]:
        s = np.array(rec["sizes"]); pr = s / s.sum()
        for c, w, p in zip(rec["clients"], rec["w"], pr):
            Wk[c] = Wk.get(c, 0) + w; Pk[c] = Pk.get(c, 0) + p
            mx = max(mx, Wk[c] / Pk[c])
NUM["LhMaxClientRatio"] = "%.3f" % mx
NUM["LhFullLog"] = "yes" if all(len(LH[k]["weight_log"]) == 200 for k in ["fedhift_L_ad0.0", "fedhift_L_none", "fedhift_L_ad0.7"]) else "no"

# Fig. 8(c), (d): long horizon accuracy and drift
for key, name, ylab, yl in [("acc", "fig8c_lh_acc", "Accuracy (%)", [60, 92]), ("proj", "fig8d_lh_drift", "Drift along attack direction", [0, 125])]:
    old = old_acc if key == "acc" else old_proj
    ser = [sty("fedavg", x=rd, y=old["fedavg"]), sty("rfa", x=rd, y=old["rfa"]),
           sty("fedavg_clip", x=rd, y=h("fedavg_clip_ad0.0", key)),
           sty("fedhift_inf", x=rd, y=h("fedhift_ad0.0", key)),
           sty("fedhift", label="FedEFT, ledger {\\itc} = 1.25", x=rd, y=h("fedhift_L_ad0.0", key))]
    S = {"name": name, "layout": "pair", "xlabel": "Round", "ylabel": ylab, "xlim": [0, 205 if key == "acc" else 262], "ylim": yl,
         "xticks": [0, 50, 100, 150, 200], "series": ser}
    if key == "acc":
        S["legend"] = "southeast"
        S["gains"] = [{"type": "ygap", "from": "fedhift", "to": "fedhift_inf", "at": 200, "unit": "diff", "fmt": "+%.1f", "pos": "left"}]
    else:
        S["legend"] = "northwest"
        S["gains"] = [{"type": "ygap", "from": "fedhift", "to": "fedhift_inf", "at": 200, "unit": "ratio", "fmt": "%.1f\u00d7 less", "pos": "right",
                       "text": "%s\u00d7 less" % NUM["LhProjRatio"]}]
    json.dump(S, open(os.path.join(D, name + ".json"), "w"))

# Fig. 9(a): running coalition share of weight against the ledger bound
ser = []
for k, rule, lab in [("fedhift_ad0.0", "fedhift_inf", None), ("fedhift_L_ad0.0", "fedhift", "FedEFT, ledger {\\itc} = 1.25"),
                     ("fedavg_clip_ad0.0", "fedavg_clip", None)]:
    W, Pi = coal(k)
    r = np.cumsum(W) / np.maximum(np.cumsum(Pi), 1e-12)
    t = np.arange(1, len(W) + 1)
    idx = t % 10 == 0
    kw = {"x": t[idx].tolist(), "y": np.round(r[idx], 4).tolist()}
    if lab: kw["label"] = lab
    ser.append(sty(rule, **kw))
ser.append({"kind": "hline", "y": 1.25, "line": "--", "label": "Ledger bound {\\itc}"})
json.dump({"name": "fig9a_cumshare", "layout": "trio", "xlabel": "Round {\\itt}",
           "ylabel": "\\Sigma{\\itW}_{\\itB} / \\Sigma\\Pi_{\\itB}", "xlim": [0, 205], "ylim": [0, 2.4], "lock_ylim": True,
           "xticks": [0, 50, 100, 150, 200], "legend": "southeast", "legend_cols": 1, "series": ser},
          open(os.path.join(D, "fig9a_cumshare.json"), "w"))

# Fig. 9(b): budget sweep (c = 1 is FedAvg+clip, c = inf the uncapped rule)
cs = [1.0, 1.1, 1.25, 1.5, 2.0]
keys = ["fedavg_clip_ad0.0", "fedhift_c1.1_ad0.0", "fedhift_L_ad0.0", "fedhift_c1.5_ad0.0", "fedhift_c2.0_ad0.0"]
acc = [100 * LH[k]["acc_last5"] for k in keys]
drift = [LH[k]["history"]["proj"][-1] for k in keys]
NUM["SweepRows"] = " & ".join("%.1f / %.1f" % (a, d) for a, d in zip(acc, drift))
json.dump({"name": "fig9b_csweep", "layout": "trio", "xscale": "log", "xlabel": "Promotion budget {\\itc}",
           "ylabel": "Final drift along attack direction", "xticks": [1, 1.1, 1.25, 1.5, 2],
           "xticklabels": ["1", "1.1", "1.25", "1.5", "2"], "xlim": [0.97, 2.1], "ylim": [0, 125],
           "series": [sty("fedhift", label="FedEFT with ledger", x=cs, y=np.round(drift, 2).tolist()),
                      {"kind": "hline", "y": round(LH["fedhift_ad0.0"]["history"]["proj"][-1], 2), "line": "--",
                       "label": "No ledger ({\\itc} = \\infty)", "color": [0.0, 0.447, 0.698]},
                      {"kind": "ref", "x": cs, "y": [round(drift[0] * c, 2) for c in cs], "label": "{\\itc} \\times clipped averaging"}],
           "legend": "northwest"}, open(os.path.join(D, "fig9b_csweep.json"), "w"))
# Fig. 9(c): the price of the ledger, boosted backdoor ASR against the budget
bc, basr, bwb = [], [], []
for c, pre in [(1.1, "bdb_fedhift_c1.1"), (1.25, "bdb_fedhift_L"), (1.5, "bdb_fedhift_c1.5"), (2.0, "bdb_fedhift_c2.0")]:
    rr = [load(f"{pre}_s{q}") for q in range(2)]
    if all(rr):
        bc.append(c); basr.append(float(np.mean([100 * r["asr_last5"] for r in rr]))); bwb.append(float(np.mean([r["mal_weight_mass"] for r in rr])))
rinf = [load(f"bdb_fedhift_s{q}") for q in range(2)]
ainf = float(np.mean([100 * r["asr_last5"] for r in rinf]))
rclip_asr = 98.2   # FedAvg+clip, first study (c = 1)
if bc:
    NUM["BdbSweep"] = ", ".join("%.1f\\%% at $c=%g$" % (a, c) for c, a in zip(bc, basr))
    json.dump({"name": "fig9c_backdoor", "layout": "trio", "xscale": "log", "xlabel": "Promotion budget {\\itc}",
               "ylabel": "Boosted backdoor ASR (%)", "xticks": [1, 1.1, 1.25, 1.5, 2], "xticklabels": ["1", "1.1", "1.25", "1.5", "2"],
               "xlim": [0.97, 2.1], "ylim": [40, 105],
               "series": [sty("fedhift", label="FedEFT with ledger", x=[1.0] + bc, y=[rclip_asr] + [round(a, 2) for a in basr]),
                          {"kind": "hline", "y": round(ainf, 2), "line": "--", "label": "No ledger ({\\itc} = \\infty)", "color": [0.0, 0.447, 0.698]}],
               "legend": "southwest"}, open(os.path.join(D, "fig9c_backdoor.json"), "w"))
json.dump({"c": cs, "acc": acc, "drift": drift}, open(os.path.join(MDIR, "aux", "csweep.json"), "w"))

# Fig. 3: allocation on one logged cohort (ledger run, sign flipping, seed 0)
R = load("main_fedhift_L_sign_flip_s0")
if R is not None:
    from math import exp
    rec = R["weight_log"][3]
    tau = np.array(rec["tau"]); s = np.array(rec["sizes"]); pr = s / s.sum(); lab = np.array(rec["labels"]); wl = np.array(rec["w"])
    o = np.argsort(tau)
    ser = []
    for T, rule, kw in [(0.8, "fedavg", {"label": "{\\itT} = 0.8", "line": ":"}), (0.2, "fedhift_inf", {"label": "{\\itT} = 0.2, no ledger"}),
                        (0.1, "fedavg_clip", {"label": "{\\itT} = 0.1", "line": "--"})]:
        w = pr * np.exp((tau - tau.max()) / T); w /= w.sum()
        ser.append(sty(rule, x=np.round(tau[o], 4).tolist(), y=np.round((w / pr)[o], 4).tolist(), **kw))
    ser.append(sty("fedhift", label="{\\itT} = 0.2, ledger {\\itc} = 1.25", x=np.round(tau[o], 4).tolist(), y=np.round((wl / pr)[o], 4).tolist()))
    ser.append({"kind": "hline", "y": 1.0, "line": ":", "label": "FedAvg share"})
    byz = tau[lab == 0]
    ann = [{"x": float(byz.mean()), "y": 1.5, "text": "Byzantine", "italic": True}] if len(byz) else []
    json.dump({"name": "fig3_alloc", "layout": "col", "xlabel": "Reputation {\\itr_k}", "ylabel": "Relative share {\\itw_k}/{\\it\\pi_k}",
               "series": ser, "annotations": ann, "legend": "northwest"}, open(os.path.join(D, "fig3_alloc.json"), "w"))
    NUM["AllocRound"] = str(rec["round"])

# ------------------------------------------------------------------ eight attacks, paired
ATK = ["none", "label_flip", "sign_flip", "gauss", "scaling", "alie", "ipm", "adaptive"]
NAME = {"none": "No attack", "label_flip": "Label flip", "sign_flip": "Sign flip", "gauss": "Gaussian", "scaling": "Scaling",
        "alie": "ALIE", "ipm": "IPM", "adaptive": "Adaptive"}
rows, diffs, wb_inf, wb_led = [], [], [], []
fair = {"inf": [], "led": []}
for a in ATK:
    A, B, WA, WB_ = [], [], [], []
    for s in range(3):
        ra, rb = load(f"main_fedhift_{a}_s{s}"), load(f"main_fedhift_L_{a}_s{s}")
        if ra is None or rb is None: continue
        A.append(100 * ra["acc_last5"]); B.append(100 * rb["acc_last5"])
        WA.append(ra["mal_weight_mass"]); WB_.append(rb["mal_weight_mass"])
        fair["inf"].append((ra["benign_worst10"], ra["benign_jain"])); fair["led"].append((rb["benign_worst10"], rb["benign_jain"]))
    if not A: continue
    diffs += list(np.array(B) - np.array(A))
    rows.append((NAME[a], np.mean(A), np.std(A), np.mean(B), np.std(B), np.mean(WA) if a != "none" else None, np.mean(WB_) if a != "none" else None))
d = np.array(diffs)
if len(d) > 1:
    lo, hi = tci(d)
    NUM["MainDiff"] = "$%+.2f$" % d.mean()
    NUM["MainCI"] = "$[%+.2f,%+.2f]$" % (lo, hi)
    NUM["MainP"] = "%.2f" % perm_p(d)
    NUM["MainN"] = str(len(d))
    NUM["MainWins"] = str(int((d > 0).sum()))
    NUM["MainWorstInf"] = "%.1f" % (100 * np.mean([x[0] for x in fair["inf"]]))
    NUM["MainWorstLed"] = "%.1f" % (100 * np.mean([x[0] for x in fair["led"]]))
    NUM["MainJainInf"] = "%.4f" % np.mean([x[1] for x in fair["inf"]])
    NUM["MainJainLed"] = "%.4f" % np.mean([x[1] for x in fair["led"]])

extra = []
for tag, pref, lab in [("a0.05", "nid", "LF, $\\alpha=0.05$"), ("a0.1", "nid", "LF, $\\alpha=0.1$")]:
    A = [load(f"{pref}_fedhift_{tag}_s{s}") for s in range(3)]; B = [load(f"{pref}_fedhift_L_{tag}_s{s}") for s in range(3)]
    A = [r for r in A if r]; B = [r for r in B if r]
    if A and B and len(A) == len(B):
        extra.append((lab, A, B))
for pref, lab in [("bdb", "Boosted backdoor"), ("onoff", "On/off sign flip")]:
    A = [load(f"{pref}_fedhift_s{s}") for s in range(2)]; B = [load(f"{pref}_fedhift_L_s{s}") for s in range(2)]
    A = [r for r in A if r]; B = [r for r in B if r]
    if A and B and len(A) == len(B):
        extra.append((lab, A, B))

L = []
L.append("\\begin{table}[!t]\n\\centering\n\\caption{FedEFT without ($c=\\infty$) and with the ledger ($c=1.25$) on identical seeds: accuracy (\\%, mean\\,$\\pm$\\,s.d.) and Byzantine weight mass. Fashion-MNIST, $\\alpha=0.5$ unless stated; three seeds, two for the boosted backdoor and the on/off attack (honest for ten rounds); LF: label flipping.}\n\\label{tab:ledger}\n\\setlength{\\tabcolsep}{1.8pt}\n\\scriptsize\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n& \\multicolumn{2}{c}{Accuracy} & \\multicolumn{2}{c}{$W_\\cB$}\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nCondition & $c=\\infty$ & $c=1.25$ & $c=\\infty$ & $c=1.25$\\\\\n\\midrule")
for n, ma, sa, mb, sb, wa, wb in rows:
    ba, bb = ("\\textbf{%.1f}" % ma, "%.1f" % mb) if ma > mb + 0.05 else (("%.1f" % ma, "\\textbf{%.1f}" % mb) if mb > ma + 0.05 else ("\\textbf{%.1f}" % ma, "\\textbf{%.1f}" % mb))
    w1 = "n/a" if wa is None else "%.3f" % wa
    w2 = "n/a" if wb is None else "%.3f" % wb
    L.append("%s & %s\\,$\\pm$\\,%.1f & %s\\,$\\pm$\\,%.1f & %s & %s\\\\" % (n, ba, sa, bb, sb, w1, w2))
L.append("\\midrule")
for lab, A, B in extra:
    ma, mb = np.mean([100 * r["acc_last5"] for r in A]), np.mean([100 * r["acc_last5"] for r in B])
    sa, sb = np.std([100 * r["acc_last5"] for r in A]), np.std([100 * r["acc_last5"] for r in B])
    wa, wb = np.mean([r["mal_weight_mass"] for r in A]), np.mean([r["mal_weight_mass"] for r in B])
    ba, bb = ("\\textbf{%.1f}" % ma, "%.1f" % mb) if ma > mb + 0.05 else (("%.1f" % ma, "\\textbf{%.1f}" % mb) if mb > ma + 0.05 else ("\\textbf{%.1f}" % ma, "\\textbf{%.1f}" % mb))
    row = "%s & %s\\,$\\pm$\\,%.1f & %s\\,$\\pm$\\,%.1f & %.3f & %.3f\\\\" % (lab, ba, sa, bb, sb, wa, wb)
    if "backdoor" in lab:
        asa = np.mean([100 * r["asr_last5"] for r in A]); asb = np.mean([100 * r["asr_last5"] for r in B])
        NUM["BdbAsrInf"], NUM["BdbAsrLed"] = "%.1f" % asa, "%.1f" % asb
        NUM["BdbAccInf"], NUM["BdbAccLed"] = "%.1f" % ma, "%.1f" % mb
    if "0.05" in lab:
        NUM["NidAccInf"], NUM["NidAccLed"] = "%.1f" % ma, "%.1f" % mb
    if "On/off" in lab:
        NUM["OnoffAccInf"], NUM["OnoffAccLed"] = "%.1f" % ma, "%.1f" % mb
        NUM["OnoffWbInf"], NUM["OnoffWbLed"] = "%.3f" % wa, "%.3f" % wb
    L.append(row)
L.append("\\midrule\n200 rounds, $\\gamma=0$ & %s & %s & %s & %s\\\\" % (NUM["LhAccInf"], "\\textbf{" + NUM["LhAccLed"] + "}", NUM["LhWbInf"], "\\textbf{" + NUM["LhWbLed"] + "}"))
L.append("200 rounds, $\\gamma=0.7$ & %s & %s & %s & %s\\\\" % (NUM["LhAccInfSeven"], "\\textbf{" + NUM["LhAccLedSeven"] + "}", NUM["LhWbInfSeven"], "\\textbf{" + NUM["LhWbLedSeven"] + "}"))
L.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(os.path.join(TDIR, "tab_ledger.tex"), "w").write("\n".join(L) + "\n")
with open(os.path.join(TDIR, "ledger_numbers.tex"), "w") as f:
    f.write("% generated by codes_v2/analyze_ledger.py\n")
    for k, v in NUM.items():
        f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print(json.dumps(NUM, indent=0))
