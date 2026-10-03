"""Shared helpers for the FODM figure/table exporters."""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
DDIR = os.path.join(HERE, "matlab", "data")
os.makedirs(DDIR, exist_ok=True)

# one colour/marker per scheme, reused in every figure of the paper
PAL = {"blue": [0.000, 0.447, 0.698], "verm": [0.835, 0.369, 0.000],
       "green": [0.000, 0.620, 0.451], "purple": [0.482, 0.196, 0.580],
       "pink": [0.800, 0.475, 0.655], "amber": [0.902, 0.624, 0.000],
       "sky": [0.337, 0.706, 0.914], "grey": [0.350, 0.350, 0.350],
       "black": [0.0, 0.0, 0.0], "brown": [0.55, 0.34, 0.16]}
STYLE = {
    "fedhift":      dict(label="FedEFT", color=PAL["blue"], marker="o", line="-", proposed=True),
    "fedavg":       dict(label="FedAvg", color=PAL["grey"], marker="v", line=":"),
    "fedavg_clip":  dict(label="FedAvg+clip", color=PAL["verm"], marker="s", line="--"),
    "klcos":        dict(label="KL-cos", color=PAL["green"], marker="d", line="-."),
    "median":       dict(label="Median", color=PAL["purple"], marker="^", line=":"),
    "trimmed_mean": dict(label="Trim. mean", color=PAL["pink"], marker=">", line="-."),
    "multikrum":    dict(label="Multi-Krum", color=PAL["amber"], marker="p", line="--"),
    "rfa":          dict(label="RFA", color=PAL["sky"], marker="h", line="-."),
    "fltrust":      dict(label="FLTrust", color=PAL["brown"], marker="<", line=":"),
    "flame":        dict(label="FLAME", color=PAL["black"], marker="x", line="--"),
    "bulyan":       dict(label="Bulyan", color=PAL["pink"], marker="*", line=":"),
    "dnc":          dict(label="DnC", color=PAL["purple"], marker="+", line="--"),
}
NAMES = {k: v["label"] for k, v in STYLE.items()}
NAMES.update({"tr_cosine": "Cosine", "tr_linear": "Linear (equal)", "tr_linopt": "Linear (fitted)",
              "tr_logistic": "Logistic", "tr_mlp": "MLP", "tr_type1": "Type-1 fuzzy",
              "fedprox": "FedProx"})


def series(rule, x, y, **kw):
    st = {k: v for k, v in STYLE[rule].items()}
    st.update(kw)
    return {"x": list(map(float, x)), "y": list(map(float, y)), **st}


def dump(panel):
    with open(os.path.join(DDIR, panel["name"] + ".json"), "w") as f:
        json.dump(panel, f)


def load(suite, slim=True):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, suite, "*.json"))):
        d = json.load(open(f))
        if slim:
            for k in ("weight_log",):
                d.pop(k, None)
        out.append(d)
    return out


def ci95(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return 0.0
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))
