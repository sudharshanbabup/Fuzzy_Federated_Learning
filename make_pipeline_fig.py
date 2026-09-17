"""Draw the FedEFT server-side pipeline schematic (vector PDF).

The schematic uses the same face and the same size scale as every other figure
in the paper, and the same colour-vision-safe palette; the two block colours are
additionally distinguished by their border weight, so the distinction between
"already in FedAvg" and "added by FedEFT" survives greyscale printing.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures_sn")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
from snstyle import MED, apply_style

apply_style()
plt.rcParams.update({"figure.constrained_layout.use": False})

FG = "#111111"
NEW = "#dbe9f6"      # blocks added by FedEFT
OLD = "#eeeeee"      # blocks already present in FedAvg
EDGE_NEW = "#0072B2"
EDGE_OLD = "#5a5a5a"

fig, ax = plt.subplots(figsize=(MED, MED * 0.92))
ax.set_xlim(0, 10.6)
ax.set_ylim(0, 10.6)
ax.axis("off")


def box(x, y, w, h, text, new=True, fs=8.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.16",
                       linewidth=1.3 if new else 0.7,
                       facecolor=NEW if new else OLD,
                       edgecolor=EDGE_NEW if new else EDGE_OLD, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=FG, zorder=3, linespacing=1.25)
    return (x + w / 2, y, x + w / 2, y + h)


def arrow(a, b, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=7,
                                 linewidth=0.75, color="#333333",
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


c1 = box(0.4, 9.2, 9.2, 1.0, "clients $k\\in\\mathcal{S}_t$: local SGD "
                             "$\\rightarrow\\ \\Delta_k^t$", new=False, fs=8.5)
c2 = box(0.4, 7.9, 4.3, 0.95, "fixed coordinate sketch\n"
                              "$\\mathbf{z}_k=(\\Delta_k)_{\\mathcal{I}}$")
c3 = box(5.3, 7.9, 4.3, 0.95, "norms $\\|\\Delta_k\\|$,\nmedian $M_t$", new=False)
c4 = box(0.4, 6.35, 9.2, 1.15,
         "four statistics:  alignment $u_{k,1}$ $\\cdot$ peer agreement $u_{k,2}$\n"
         "magnitude regularity $u_{k,3}$ $\\cdot$ temporal stability $u_{k,4}$")
c5 = box(0.4, 4.95, 4.3, 1.0, "type-1 pass $\\rightarrow$\n"
                              "heterogeneity $\\hat{s}^t$")
c6 = box(5.3, 4.95, 4.3, 1.0, "FOU factor $\\varphi^t$\n"
                              "(directional sets only)")
c7 = box(0.4, 3.45, 9.2, 1.1, "interval type-2 inference (16 rules) $\\rightarrow$ "
                              "exact\ncentre-of-sets type reduction $\\rightarrow$ "
                              "$\\hat{\\tau}_k$")
c8 = box(0.4, 2.15, 4.3, 0.95, "reputation memory\n$\\tau_k\\leftarrow(1-\\rho)\\tau_k+\\rho\\hat{\\tau}_k$")
c9 = box(5.3, 2.15, 4.3, 0.95, "$w_k\\propto\\pi_k e^{\\tau_k/T}$\n(entropic allocation)")
c10 = box(0.4, 0.85, 9.2, 0.95, "clip to $\\nu M_t$, average, "
                                "$\\theta^t=\\theta^{t-1}+\\sum_k w_k\\tilde{\\Delta}_k$",
          new=False, fs=8.5)

arrow((2.5, 9.2), (2.5, 8.85))
arrow((7.4, 9.2), (7.4, 8.85))
arrow((2.5, 7.9), (2.5, 7.5))
arrow((7.4, 7.9), (7.4, 7.5))
arrow((2.5, 6.35), (2.5, 5.95))
arrow((4.7, 5.45), (5.3, 5.45))
arrow((7.4, 6.35), (7.4, 5.95), rad=0.0)
arrow((2.5, 4.95), (2.5, 4.55))
arrow((7.4, 4.95), (7.4, 4.55))
arrow((2.5, 3.45), (2.5, 3.1))
arrow((4.7, 2.62), (5.3, 2.62))
arrow((7.4, 2.15), (7.4, 1.8))
arrow((9.75, 1.32), (9.75, 9.7), rad=0.0, style="-")
ax.plot([9.6, 9.75], [1.32, 1.32], color="#333333", lw=0.75)
arrow((9.75, 9.7), (9.6, 9.7))
ax.text(10.15, 5.5, "next round", rotation=90, ha="center", va="center", fontsize=8.0)

fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
fig.savefig(os.path.join(OUT, "fig_pipeline.pdf"))
print("wrote", os.path.join(OUT, "fig_pipeline.pdf"))
