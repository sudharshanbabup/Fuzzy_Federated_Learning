"""Regenerate the trust-dynamics numbers quoted in Section VII-D.

Prints, per Dirichlet concentration under sign flipping: the mean fuzzy trust of
benign and Byzantine clients, the margin delta, the Byzantine weight mass
predicted by the intermediate bound in the proof of the weight-mass corollary,
the measured mass, and the mean FOU factor.  Also contrasts the two-stage
heterogeneity estimate with a naive whole-cohort MAD.

Usage:  python verify_trust.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_num_threads(2)

from fedhift.aggregators import FedHIFT, mad          # noqa: E402
from fedhift.data import load_federated               # noqa: E402
from fedhift.fl import FLConfig, run                  # noqa: E402

WARMUP = 3


def sweep(attack="sign_flip", alphas=(1.0, 0.5, 0.1, 0.05), rounds=30, T=0.2,
          latex=None):
    orig = FedHIFT.__call__
    store: list = []

    def patched(self, U, sizes, st):
        out = orig(self, U, sizes, st)
        a = 2 * out[1]["u"][:, 0] - 1                  # raw cosine
        store.append((list(st["client_ids"]), np.asarray(out[1]["tau"]),
                      np.asarray(out[1]["w"]), np.asarray(sizes, dtype=float),
                      float(out[1]["hetero"]), float(mad(a)),
                      float(self.engine.last_phi)))
        return out

    FedHIFT.__call__ = patched
    rows = []
    try:
        print(f"attack={attack}  T={T}  rounds={rounds}  (warm-up {WARMUP} rounds dropped)")
        print(f"{'alpha':>6} {'tau_ben':>8} {'tau_byz':>8} {'delta':>7} "
              f"{'W_B pred':>9} {'W_B meas':>9} {'phi':>6} "
              f"{'s_hat':>7} {'MAD_all':>8}")
        for al in alphas:
            store.clear()
            fd = load_federated("fmnist", 30, al, 0, 20000, 100)
            cfg = FLConfig(rounds=rounds, eval_every=rounds, aggregator="fedhift",
                           attack=attack, byz_frac=0.2, alpha=al,
                           train_subsample=20000, temperature=T)
            r = run(cfg, fd)
            mal = set(r["malicious"])
            tb, tm, wb, pib, pih, ph, sh, ma = [], [], [], [], [], [], [], []
            for cids, tau, w, n, het, mad_all, phi in store[WARMUP:]:
                m = np.array([c in mal for c in cids])
                if not m.any():
                    continue
                tb += list(tau[~m]); tm += list(tau[m])
                wb.append(w[m].sum())
                pi = n / n.sum()
                pib.append(pi[m].sum()); pih.append(pi[~m].sum())
                ph.append(phi); sh.append(het); ma.append(mad_all)
            tbar, mbar = float(np.mean(tb)), float(np.mean(tm))
            d = tbar - mbar
            PB, PH = float(np.mean(pib)), float(np.mean(pih))
            pred = PB / (PB + PH * np.exp(d / T))
            rows.append((al, tbar, mbar, d, pred, float(np.mean(wb)),
                         float(np.mean(ph)), float(np.mean(sh)),
                         float(np.mean(ma))))
            print(f"{al:6g} {tbar:8.3f} {mbar:8.3f} {d:7.3f} "
                  f"{pred:9.3f} {float(np.mean(wb)):9.3f} {float(np.mean(ph)):6.2f} "
                  f"{float(np.mean(sh)):7.3f} {float(np.mean(ma)):8.3f}")
    finally:
        FedHIFT.__call__ = orig
    if latex:
        write_table(rows, latex, T)
    return rows


def write_table(rows, path, T):
    """Emit the trust-dynamics table quoted in Section VI-D."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdr = " & ".join("$%g$" % r[0] for r in rows)
    def line(name, i, fmt="%.3f"):
        return name + " & " + " & ".join("$" + fmt % r[i] + "$" for r in rows) + r"\\"
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Inside the engine under sign flipping at $T=%g$, averaged over "
         r"rounds $4$ to $30$ of one instrumented run per column. $\hat s^t$ is the "
         r"two-stage heterogeneity estimate and $\mathrm{MAD}_{\mathcal S_t}$ the naive "
         r"whole-cohort alternative the adversary can inflate. The predicted mass "
         r"substitutes the two population means into the intermediate expression of "
         r"Corollary~\ref{cor:mass}.}" % T,
         r"\label{tab:trust}", r"\setlength{\tabcolsep}{4pt}", r"\footnotesize",
         r"\begin{tabular}{l%s}" % ("c" * len(rows)), r"\hline",
         r"$\alpha$ & " + hdr + r"\\", r"\hline",
         line(r"benign trust $\bar\tau_\cH$", 1),
         line(r"Byzantine trust $\bar\tau_\cB$", 2),
         line(r"margin $\delta$", 3),
         line(r"$W_\cB$ predicted", 4),
         line(r"$W_\cB$ measured", 5),
         line(r"footprint $\varphi^t$", 6, "%.2f"),
         line(r"$\hat s^t$ (two-stage)", 7),
         line(r"$\mathrm{MAD}_{\mathcal S_t}$ (naive)", 8),
         r"\hline", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")


def calibration(points=None):
    """Predicted against measured Byzantine weight mass across conditions.

    Each point is one instrumented run. The prediction substitutes the two
    population mean trust scores into the intermediate expression of the
    weight-mass corollary; it is a mean-field reading of that bound, not the
    bound itself, since the corollary assumes a uniform separation that the
    means do not certify and Jensen's inequality places the exact quantity
    above the substituted one.
    """
    points = points or (
        [("sign_flip", a, 0.2) for a in (1.0, 0.5, 0.1, 0.05)]
        + [("sign_flip", 0.5, T) for T in (0.05, 0.1, 0.4, 0.8)]
        + [(a, 0.5, 0.2) for a in ("label_flip", "scaling", "gauss", "alie",
                                   "ipm", "adaptive")])
    rows = []
    for atk, al, T in points:
        r = sweep(attack=atk, alphas=(al,), T=T)
        if r:
            rows.append((atk, al, T) + tuple(r[0][1:]))
    return rows


def calibration_figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plotstyle import apply_style, COL, LEGEND, ONE_COL
    apply_style()
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.0))
    groups = {"heterogeneity $\\alpha$": [r for r in rows if r[0] == "sign_flip" and r[2] == 0.2],
              "temperature $T$": [r for r in rows if r[0] == "sign_flip" and r[2] != 0.2],
              "attack": [r for r in rows if r[0] != "sign_flip"]}
    for i, (lab, g) in enumerate(groups.items()):
        if not g:
            continue
        ax.scatter([x[6] for x in g], [x[7] for x in g], s=16, label=lab,
                   color=COL[i + 1], marker=["o", "s", "^"][i],
                   edgecolor="black", linewidth=0.3, zorder=3)
    lim = 0.05, 1.0
    ax.plot(lim, lim, color="black", lw=0.7, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predicted Byzantine weight mass")
    ax.set_ylabel("measured mass $W_{\\mathcal{B}}$")
    ax.grid(ls=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", **LEGEND)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    sweep(latex=os.path.join(HERE, "tables", "tab_trust.tex"))
    rows = calibration()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    calibration_figure(rows, os.path.join(HERE, "figures", "fig_calib.png"))
    print("\ncalibration points: attack alpha T | tau_H tau_B delta pred meas phi")
    for r in rows:
        print("  ", r[0], r[1], r[2], " ".join(f"{x:.3f}" for x in r[3:9]))
