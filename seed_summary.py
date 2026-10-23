#!/usr/bin/env python3
"""Mean ± std of metrics over training seeds, with per-seed plots.

Before CRC: test split rows of seed*/head_metrics.csv (written by train.py),
threshold 0.5 -> accuracy, F1, FNR, FPR.
After CRC: seed*/crc_fnr_trials/summary.csv (means over trials, written by
crc_fnr_trials.py) -> per alpha, mean and std over seeds of risk (FNR), FPR,
F1, lambda_hat.

Run from project root:
    python seed_summary.py --backbone efficientnet_b3
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR
from crc_fnr_plots import plt, _style, SURFACE, INK, BLUE, ORANGE, GREEN

PURPLE = "#8a55c9"
BEFORE_METRICS = [("acc@0.5", "Accuracy", GREEN), ("f1@0.5", "F1", PURPLE),
                  ("fnr@0.5", "FNR", BLUE), ("fpr@0.5", "FPR", ORANGE)]


def seed_dirs(base):
    return sorted(base.glob("seed[0-9]*"), key=lambda d: int(d.name[4:]))


def plot_per_seed(df, metrics, title, out_path):
    """One panel per metric: value per seed, mean line, ±1 std band."""
    fig, axs = plt.subplots(1, len(metrics), figsize=(3.75 * len(metrics), 3.4),
                            constrained_layout=True, facecolor=SURFACE)
    seeds = df["seed"].values
    for ax, (col, label, color) in zip(axs, metrics):
        m, s = df[col].mean(), df[col].std()
        ax.axhspan(m - s, m + s, color=color, alpha=0.15, lw=0, label="±1 std")
        ax.axhline(m, color=INK, lw=1.2, ls="--", label="ortalama")
        ax.plot(seeds, df[col], color=color, lw=1, alpha=0.5)
        ax.scatter(seeds, df[col], color=color, s=40, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.set_xticks(seeds)
        ax.set_xlabel("seed")
        ax.set_title(f"{label}  ({m:.4f} ± {s:.4f})", color=INK)
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
    fig.suptitle(title, color=INK)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def before_crc(base, out_dir):
    df = pd.concat([pd.read_csv(d / "head_metrics.csv") for d in seed_dirs(base)],
                   ignore_index=True)
    df = df[df["split"] == "test"].sort_values("seed").reset_index(drop=True)
    cols = [c for c, _, _ in BEFORE_METRICS]
    df = df[["seed"] + cols]
    summary = df[cols].agg(["mean", "std"])

    df.to_csv(out_dir / "before_crc_per_seed.csv", index=False)
    summary.to_csv(out_dir / "before_crc_summary.csv")
    plot_per_seed(df, BEFORE_METRICS, f"CRC öncesi, test (eşik 0.5), {len(df)} seed",
                  out_dir / "before_crc_per_seed.png")

    fmt = lambda v: f"{v:.4f}"
    print(f"\n[before CRC] test, threshold 0.5")
    print(df.to_string(index=False, float_format=fmt))
    print()
    print(summary.to_string(float_format=fmt))


def after_crc(base, out_dir):
    """Per alpha: mean and std over seeds of each seed's trial means (summary.csv)."""
    df = pd.concat([pd.read_csv(d / "crc_fnr_trials" / "summary.csv").assign(seed=int(d.name[4:]))
                    for d in seed_dirs(base)], ignore_index=True)
    cols = ["risk_mean", "fpr_mean", "f1_mean", "lambda_hat_mean"]
    g = df.groupby("alpha")[cols]
    mean, std = g.mean(), g.std()
    summary = pd.DataFrame({"n_seeds": g.size()})
    for c in cols:
        name = {"risk": "fnr"}.get(c.removesuffix("_mean"), c.removesuffix("_mean"))
        summary[f"{name}_mean"], summary[f"{name}_std"] = mean[c], std[c]
    summary = summary.reset_index()

    df = df.rename(columns={"risk_mean": "fnr_mean"})
    cols = ["fnr_mean"] + cols[1:]
    df[["seed", "alpha"] + cols].to_csv(out_dir / "after_crc_per_seed.csv", index=False)
    summary.to_csv(out_dir / "after_crc_summary.csv", index=False)

    print(f"\n[after CRC] mean ± std over seeds")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main():
    parser = argparse.ArgumentParser(description="Metrics over training seeds")
    parser.add_argument("--backbone", default="efficientnet_b3")
    args = parser.parse_args()

    base = OUTPUT_DIR / "classification" / args.backbone
    out_dir = base / "seed_summary"
    out_dir.mkdir(exist_ok=True)

    before_crc(base, out_dir)
    after_crc(base, out_dir)
    print(f"\n  saved -> {out_dir}")


if __name__ == "__main__":
    main()
