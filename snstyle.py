"""Shared figure style for the Springer Nature manuscript.

One face, one size scale, one palette, for every graphic in the paper.

* Type is TeX Gyre Heros, the OpenType release of URW Nimbus Sans, which is
  metrically Helvetica. Nature Portfolio asks for Helvetica or Arial in
  figures, so this is the compliant face and it is used everywhere, including
  the schematic.
* Axis labels and titles are 9 pt, tick labels and legends 8 pt, everywhere.
  No figure-specific size overrides.
* Figures are authored at the exact width they are included at, so
  ``\\includegraphics[width=...]`` reproduces them 1:1 and the type reaches the
  page at the size set here rather than at that size times an accidental scale
  factor. ``savefig.bbox`` is therefore left alone and constrained layout is
  used instead: a tight bounding box changes the saved width from figure to
  figure and LaTeX would then rescale each one differently.
* The palette is Okabe-Ito, which is colour-vision safe and contains no
  red-green pair. Colour is never the only channel: every line series also
  carries its own marker and dash pattern and the bar charts carry hatching, so
  the figures survive greyscale printing.
* Output is vector PDF.
"""
from __future__ import annotations

import glob

import matplotlib
from matplotlib import font_manager

for _f in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-*.otf"):
    try:
        font_manager.fontManager.addfont(_f)
    except Exception:                                       # pragma: no cover
        pass

# sn-jnl.cls single column: \textwidth = 372 pt = 5.148 in.
TEXT_W = 5.148
WIDE = TEXT_W                    # \includegraphics[width=\textwidth]
MED = 0.86 * TEXT_W              # [width=0.86\textwidth]
NARROW = 0.70 * TEXT_W           # [width=0.70\textwidth]

# Okabe-Ito.
COL = ["#000000", "#0072B2", "#E69F00", "#009E73", "#CC79A7",
       "#56B4E9", "#D55E00", "#595959", "#6A3D9A"]
MARK = ["o", "s", "^", "v", "D", "P", "X", "*", "h"]
LINE = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]

LEGEND = {"frameon": False, "handletextpad": 0.4, "columnspacing": 0.9,
          "borderaxespad": 0.2, "handlelength": 1.6}


def apply_style(base: float = 9.0) -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial",
                            "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
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
        "lines.linewidth": 1.2,
        "lines.markersize": 3.4,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "legend.borderpad": 0.25,
        "legend.labelspacing": 0.3,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.03,
        "figure.constrained_layout.wspace": 0.03,
        "savefig.bbox": None,
        "pdf.fonttype": 42,          # embed as TrueType, editable downstream
        "ps.fonttype": 42,
    })


def style_for(index: int) -> dict:
    """Colour, marker and line style for series ``index``."""
    return {"color": COL[index % len(COL)],
            "marker": MARK[index % len(MARK)],
            "linestyle": LINE[index % len(LINE)]}


def smooth_xy(x, y, n: int = 240, logx: bool = False):
    """A shape-preserving curve through the measured points.

    The measurements themselves are always drawn as markers with error bars;
    this only supplies the connecting line. PCHIP is used rather than a cubic
    spline because it is monotone between samples and therefore cannot invent
    an overshoot the data does not contain, which a smoothing spline can.
    Falls back to the straight polyline when SciPy is unavailable or fewer than
    three points are present.
    """
    import numpy as np
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    xs = np.log10(x) if logx else x
    try:
        from scipy.interpolate import PchipInterpolator
        f = PchipInterpolator(xs, y)
    except Exception:                                       # pragma: no cover
        return (x, y)
    g = np.linspace(xs[0], xs[-1], n)
    return (10.0 ** g if logx else g), f(g)


def curve(ax, x, y, yerr=None, *, index: int = 0, label=None, logx: bool = False,
          emphasis: bool = False, **kw):
    """Draw one series as a smooth curve plus its measured points.

    The line is the PCHIP interpolant; the markers and error bars sit on the
    measurements, so nothing in the figure claims a value that was not run.
    """
    st = style_for(index)
    st.update(kw)
    marker = st.pop("marker")
    ls = st.pop("linestyle")
    color = st.pop("color")
    lw = 1.8 if emphasis else 1.1
    gx, gy = smooth_xy(x, y, logx=logx)
    ax.plot(gx, gy, color=color, linestyle=ls, linewidth=lw, zorder=3,
            solid_capstyle="round", **st)
    ax.errorbar(x, y, yerr=yerr, color=color, marker=marker, linestyle="none",
                capsize=1.6, elinewidth=0.6, markersize=3.4, zorder=4,
                label=label, **st)
