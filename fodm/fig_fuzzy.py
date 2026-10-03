"""Fuzzy-model figures: interval type-2 antecedents and trust surfaces."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fodm.common import dump, PAL
from fedhift.fuzzy import FuzzyTrustEngine, IT2Config, gaussian

cfg = IT2Config()
x = np.linspace(0, 1, 101)
phi = 1.5
s = []
for j, (c, nm, col) in enumerate([(0.0, "LOW", PAL["verm"]), (1.0, "HIGH", PAL["blue"])]):
    up = gaussian(x, c, cfg.sigma0 * phi); lo = gaussian(x, c, cfg.sigma0 / phi)
    s.append({"label": f"{nm}, upper MF", "x": x.tolist(), "y": up.tolist(), "color": col, "marker": "none", "line": "-"})
    s.append({"label": f"{nm}, lower MF", "x": x.tolist(), "y": lo.tolist(), "color": col, "marker": "none", "line": "--"})
    s.append({"label": f"{nm}, type-1", "x": x.tolist(), "y": gaussian(x, c, cfg.sigma0).tolist(), "color": PAL["grey"] if j else PAL["grey"], "marker": "none", "line": ":" if j else "-."})
dump({"name": "fuzzy_mf", "layout": "col", "width_in": 2.55, "height_in": 1.95, "font_pt": 7.5, "legend_pt": 6,
      "xlabel": "Normalised statistic {\\itu}", "ylabel": "Membership grade", "xlim": [0, 1], "ylim": [0, 1.55],
      "lock_ylim": True, "legend": "north", "legend_cols": 2, "series": s})

g = np.linspace(0, 1, 41)
A, Bm = np.meshgrid(g, g)
def cmap(lv, signed=False):
    n = len(lv) - 1
    t = np.linspace(0, 1, n)
    b = np.array(PAL["blue"]); r = np.array(PAL["verm"]); w = np.ones(3)
    if signed:
        out = [list((1 - 2 * a) * r + 2 * a * w) if a < 0.5 else list((2 - 2 * a) * w + (2 * a - 1) * b) for a in t]
    else:
        out = [list((1 - a) * w * 0.98 + a * b) for a in t]
    return out

def surf(fix, pair, type1, phi_val):
    e = FuzzyTrustEngine(cfg, type1=type1)
    u = np.tile(np.array(fix, float), (A.size, 1))
    u[:, pair[0]] = A.ravel(); u[:, pair[1]] = Bm.ravel()
    if type1:
        tau, _ = e(u, 0.0)
    else:
        e.cfg = IT2Config(kappa=(phi_val - 1.0)); tau, _ = e(u, cfg.sigma_ref)
    return tau.reshape(A.shape)

lab = ["Alignment {\\itu}_1", "Peer agreement {\\itu}_2", "Magnitude regularity {\\itu}_3", "Temporal consistency {\\itu}_4"]
panels = [("fuzzy_surf_t1_12", (0, 1), True, 1.0, "Type-1"),
          ("fuzzy_surf_it2_12", (0, 1), False, 1.5, "IT2"),
          ("fuzzy_surf_t1_13", (0, 2), True, 1.0, "Type-1"),
          ("fuzzy_diff_12", (0, 1), None, 1.5, "diff")]
for name, pair, t1, ph, tag in panels:
    fix = [0.9, 0.9, 0.9, 0.9]
    if tag == "diff":
        Z = surf(fix, pair, False, ph) - surf(fix, pair, True, 1.0)
        lv = np.round(np.linspace(-0.4, 0.4, 17), 3).tolist()
        cbl = "\\tau_{IT2} - \\tau_{T1}"
    else:
        Z = surf(fix, pair, t1, ph); lv = np.round(np.linspace(0, 1, 11), 2).tolist(); cbl = "Trust \\tau"
    dump({"name": name, "layout": "col", "width_in": 2.55, "height_in": 1.95, "font_pt": 7.5,
          "margins_in": [0.42, 0.36, 0.10, 0.08],
          "xlabel": lab[pair[0]], "ylabel": lab[pair[1]], "xlim": [0, 1], "ylim": [0, 1],
          "series": [{"kind": "contourf", "x": g.tolist(), "y": g.tolist(), "z": Z.tolist(),
                      "levels": lv, "cmap": cmap(lv, tag == "diff"),
                      "label_levels": lv[1:-1]}]})
    print(name, float(Z.min()), float(Z.max()))
