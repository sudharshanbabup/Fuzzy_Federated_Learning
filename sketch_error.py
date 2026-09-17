"""Approximation error of the coordinate sketch, for Section VII-F.

The sketched cosine is a ratio of three sampled estimates and is therefore
consistent rather than unbiased, so its quality has to be measured rather than
asserted. This script runs a short federated training run, and at every round
compares the exact cosine similarities among the cohort updates with the ones
computed on a random coordinate subset of size d, over a range of d. It reports
mean absolute error, root mean squared error and maximum absolute error, both
for the full pairwise cosine matrix and for the alignment statistic u_{k,1},
which is the one the trust engine actually consumes.

Usage:  python sketch_error.py
Writes: figures/fig_sketch.png and tables/tab_sketch.tex
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_num_threads(2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

from fedhift.aggregators import (cosine_matrix, geometric_median,  # noqa: E402
                                 sketch, sketch_indices)
from fedhift.data import load_federated                            # noqa: E402
from fedhift.fl import FLConfig, run                               # noqa: E402
import fedhift.aggregators as A                                    # noqa: E402
from plotstyle import apply_style, COL, ONE_COL                    # noqa: E402

DIMS = [2 ** k for k in (8, 10, 12, 14)]


def _align(Z: torch.Tensor) -> np.ndarray:
    Zn = torch.linalg.norm(Z, dim=1).clamp_min(1e-12)
    Zdir = Z / Zn[:, None]
    ref = geometric_median(Zdir)
    refn = torch.linalg.norm(ref).clamp_min(1e-12)
    return ((Zdir @ ref) / refn).numpy()


def collect(attack: str = "sign_flip", rounds: int = 20, alpha: float = 0.5):
    """Return, per sketch dimension, the errors of the cosine matrix and of u1."""
    cache: list[torch.Tensor] = []
    orig = A.FedHIFT.__call__

    def patched(self, U, sizes, st):
        cache.append(U.clone())
        return orig(self, U, sizes, st)

    A.FedHIFT.__call__ = patched
    try:
        fd = load_federated("fmnist", 30, alpha, 0, 20000, 100)
        run(FLConfig(rounds=rounds, eval_every=rounds, aggregator="fedhift",
                     attack=attack, byz_frac=0.2, alpha=alpha,
                     train_subsample=20000), fd)
    finally:
        A.FedHIFT.__call__ = orig

    p = cache[0].shape[1]
    gen = torch.Generator().manual_seed(12345)
    out = {}
    for d in DIMS:
        idx = sketch_indices(p, d, gen)
        ec, ea, kt, top = [], [], [], []
        for U in cache:
            Cx = cosine_matrix(U)
            Cs = cosine_matrix(sketch(U, idx))
            iu = np.triu_indices(len(U), k=1)
            ec.append(np.abs(Cx[iu] - Cs[iu]))
            ax_, as_ = _align(U), _align(sketch(U, idx))
            ea.append(np.abs(ax_ - as_))
            # what the engine actually consumes is an ordering, so we also
            # measure how much of it survives the sketch
            sx = np.sign(ax_[:, None] - ax_[None, :])
            ss = np.sign(as_[:, None] - as_[None, :])
            iu2 = np.triu_indices(len(U), k=1)
            kt.append(float((sx[iu2] == ss[iu2]).mean() * 2 - 1))
            h = max(1, len(U) // 2)
            top.append(len(set(np.argsort(-ax_)[:h]) & set(np.argsort(-as_)[:h])) / h)
        ec = np.concatenate(ec)
        ea = np.concatenate(ea)
        out[d] = {"cos_mae": float(ec.mean()),
                  "cos_med": float(np.median(ec)),
                  "cos_p95": float(np.percentile(ec, 95)),
                  "cos_rmse": float(np.sqrt((ec ** 2).mean())),
                  "cos_max": float(ec.max()),
                  "u1_mae": float(ea.mean()),
                  "u1_med": float(np.median(ea)),
                  "u1_p95": float(np.percentile(ea, 95)),
                  "u1_rmse": float(np.sqrt((ea ** 2).mean())),
                  "u1_max": float(ea.max()),
                  "u1_tau": float(np.mean(kt)), "u1_top": float(np.mean(top))}
    return out, p


def main():
    res, p = collect()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "sketch_error.json"), "w") as fh:
        json.dump({"p": p, "dims": {str(k): v for k, v in res.items()}}, fh, indent=1)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    apply_style()

    fig, ax = plt.subplots(figsize=(ONE_COL, 1.65))
    ds = DIMS
    for key, lab, c, mk, ls in [
            ("cos_rmse", "cosine matrix, RMSE", COL[1], "o", "-"),
            ("cos_max", "cosine matrix, max", COL[1], "s", "--"),
            ("u1_rmse", "alignment $u_{k,1}$, RMSE", COL[2], "^", "-"),
            ("u1_max", "alignment $u_{k,1}$, max", COL[2], "v", "--")]:
        ax.plot(ds, [res[d][key] for d in ds], marker=mk, color=c, ls=ls, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sketch dimension $d$")
    ax.set_ylabel("absolute error")
    ax.grid(ls=":")
    ax.set_axisbelow(True)
    ax2 = ax.twinx()
    ax2.plot(ds, [res[d]["u1_tau"] for d in ds], marker="D", color=COL[3],
             ls="-.", label="rank correlation $\\tau_b$")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Kendall $\\tau_b$ of the ranking", labelpad=2)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.2, ncol=2,
              columnspacing=0.8, handletextpad=0.4, handlelength=1.6,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(os.path.join(HERE, "figures", "fig_sketch.png"))
    plt.close(fig)

    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Error of the sketched cosine against exact computation on "
         r"the full $p=%d$ dimensional updates, over twenty rounds of a "
         r"sign-flipping run at $\alpha=0.5$. Errors are absolute and are taken "
         r"over all cohort pairs (cosine matrix) and over all clients "
         r"(alignment $u_{k,1}$). $\tau_b$ is Kendall's rank correlation between "
         r"the exact and sketched alignment within a cohort and top-half the "
         r"fraction of the five highest-aligned clients the sketch recovers; "
         r"these are what the engine consumes, since the rule base reads an "
         r"ordering rather than a value.}" % p,
         r"\label{tab:sketch}", r"\setlength{\tabcolsep}{4pt}", r"\footnotesize",
         r"\begin{tabular}{lccccccc}", r"\hline",
         r"& \multicolumn{3}{c}{pairwise cosine} & \multicolumn{4}{c}{alignment $u_{k,1}$}\\",
         r"$d$ & MAE & RMSE & max & MAE & max & $\tau_b$ & top-half\\", r"\hline"]
    for d in DIMS:
        r = res[d]
        L.append("$2^{%d}$ & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\" % (
            int(np.log2(d)), r["cos_mae"], r["cos_rmse"], r["cos_max"],
            r["u1_mae"], r["u1_max"], r["u1_tau"], r["u1_top"]))
    L += [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(HERE, "tables", "tab_sketch.tex"), "w").write("\n".join(L) + "\n")
    print("wrote figures/fig_sketch.png and tables/tab_sketch.tex")
    for d in DIMS:
        print(d, {k: round(v, 5) for k, v in res[d].items()})


if __name__ == "__main__":
    main()
