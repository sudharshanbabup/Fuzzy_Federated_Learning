"""Shared figure style for every plot in the paper.

The settings follow the publisher's graphics guidance: the same Times face the manuscript
uses, at 9 to 10 points throughout, consistent padding, one-column and two-column widths taken
from the class itself, and a high-contrast palette that avoids red-green
pairs. Color is never the only channel: every series also carries its own
marker and line style, so the figures remain readable in greyscale and to
readers with color vision deficiencies. Raster output is written at 600 dpi.
"""
from __future__ import annotations

import glob

import matplotlib
from matplotlib import font_manager

# The manuscript sets its text in Times (the class loads mathptmx, whose text
# font is URW Nimbus Roman). TeX Gyre Termes is the OpenType release of that
# same face and ships with the TeX distribution, so registering it here makes
# the lettering inside every figure identical to the lettering around it.
for _f in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf"):
    try:
        font_manager.fontManager.addfont(_f)
    except Exception:                                  # pragma: no cover
        pass

# Figures are authored at the exact column and text widths of the IEEE Access
# class (242.67 pt and 505.12 pt), so \includegraphics[width=\columnwidth]
# reproduces them at 1:1 and the lettering reaches the page at the point size
# set below rather than at that size times an accidental scale factor.
ONE_COL = 3.358        # inches, 85.3 mm, = 242.67 pt
TWO_COL = 6.989        # inches, 177.5 mm, = 505.12 pt

# Okabe-Ito, an eight-color palette designed for color vision deficiency.
COL = ["#000000", "#0072B2", "#E69F00", "#009E73", "#CC79A7",
       "#56B4E9", "#D55E00", "#595959", "#6A3D9A"]
MARK = ["o", "s", "^", "v", "D", "P", "X", "*", "h"]
LINE = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]


def apply_style(base: float = 9.0) -> None:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Times New Roman", "Nimbus Roman",
                       "Liberation Serif", "FreeSerif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": base - 1.0,
        "xtick.labelsize": base - 1.0,
        "ytick.labelsize": base - 1.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "axes.labelpad": 2.0,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "legend.borderpad": 0.2,
        "legend.labelspacing": 0.25,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.9,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        # never crop to the ink: a tight bounding box changes the saved width
        # from figure to figure and LaTeX would then rescale each one by a
        # different factor, defeating the uniform type size
        "savefig.bbox": None,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.03,
        "figure.constrained_layout.wspace": 0.03,
    })


LEGEND = {"frameon": False, "handletextpad": 0.4, "columnspacing": 0.9,
          "borderaxespad": 0.15, "handlelength": 1.5}


def style_for(index: int) -> dict:
    """Color, marker and line style for series ``index``."""
    return {"color": COL[index % len(COL)],
            "marker": MARK[index % len(MARK)],
            "linestyle": LINE[index % len(LINE)]}
