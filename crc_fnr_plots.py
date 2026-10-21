"""Plots for crc_fnr_trials.py (kept separate from the CRC logic).

plot_histograms:    per alpha, histograms over trials of val FNR, val FPR, lambda_hat.
plot_alpha_summary: all alphas side by side, lambda_hat and val FNR over trials.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3a9a5b"  # FNR / FPR / lambda_hat

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.6)


def plot_histograms(df, alpha, out_path):
    """Paper-style histograms over trials: val FNR (risk), val FPR, lambda_hat."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4), constrained_layout=True,
                            facecolor=SURFACE)
    panels = [("risk", BLUE, "val FNR (risk)"), ("fpr", ORANGE, "val FPR"),
              ("lambda_hat", GREEN, "λ̂  (eşik)")]
    for ax, (col, color, label) in zip(axs, panels):
        ax.hist(df[col], bins=40, density=True, color=color, alpha=0.75)
        ax.axvline(df[col].mean(), color=INK, lw=1.2, label=f"ortalama = {df[col].mean():.4f}")
        ax.set_xlabel(label)
        _style(ax)
    axs[0].axvline(alpha, color=MUTED, lw=1.2, ls="--", label=f"α = {alpha}")
    axs[0].set_ylabel("yoğunluk")
    for ax in axs:
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"CRC FNR, α = {alpha}: {len(df)} rastgele calib/val bölünmesi",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_alpha_summary(trials, summary, out_path):
    """All alphas side by side: distribution over trials of lambda_hat and val FNR."""
    alphas = summary["alpha"].tolist()
    x = np.arange(len(alphas))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True,
                            facecolor=SURFACE)

    for ax, (col, color) in zip(axs, (("lambda_hat", GREEN), ("risk", BLUE))):
        data = [trials.loc[trials["alpha"] == a, col].values for a in alphas]
        ax.boxplot(data, positions=x, widths=0.45, patch_artist=True, whis=(2.5, 97.5),
                   boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color, lw=1),
                   whiskerprops=dict(color=color, lw=1), capprops=dict(color=color, lw=1),
                   medianprops=dict(color=color, lw=1.5),
                   flierprops=dict(marker="o", ms=2, mfc=color, mec="none", alpha=0.4))
        means = [d.mean() for d in data]
        ax.scatter(x, means, s=36, marker="D", color=INK, edgecolors=SURFACE,
                   linewidths=1.5, zorder=4)
        for xi, m in zip(x, means):
            ax.annotate(f"{m:.3f}", (xi, m), xytext=(30, 0), textcoords="offset points",
                        va="center", fontsize=7, color=INK,
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))
        ax.set_xticks(x, [f"{a:g}" for a in alphas])
        ax.set_xlabel("α")
        _style(ax)

    # targets on the risk panel: alpha (upper) and alpha - 2/(n+1) (lower)
    for xi, (a, lb) in enumerate(zip(alphas, summary["lower_bound"])):
        axs[1].hlines(a, xi - 0.35, xi + 0.35, color=INK, lw=1.2, ls="--", zorder=3)
        axs[1].hlines(lb, xi - 0.35, xi + 0.35, color=MUTED, lw=1.2, ls=":", zorder=3)

    axs[0].set_ylabel("λ̂  (eşik, p ≥ λ̂ → pozitif)")
    axs[1].set_ylabel("val FNR (risk)")
    axs[0].set_title("Bulunan λ̂", loc="left", color=INK, fontsize=10)
    axs[1].set_title("Çıkan val FNR", loc="left", color=INK, fontsize=10)
    handles = [
        Line2D([], [], marker="D", ls="", color=INK, label="ortalama"),
        Line2D([], [], color=MUTED, lw=1.5, label="kutu: %25-%75, çizgi: medyan, bıyık: %2.5-%97.5"),
        Line2D([], [], color=INK, lw=1.2, ls="--", label="α (üst sınır)"),
        Line2D([], [], color=MUTED, lw=1.2, ls=":", label="α − 2/(n+1) (alt sınır)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4,
               frameon=False)
    fig.suptitle(f"CRC FNR: her α için {trials['trial'].nunique()} trial", x=0.01,
                 ha="left", color=INK, fontsize=11)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
