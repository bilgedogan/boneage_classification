#!/usr/bin/env python3
"""CRC for FNR only, repeated random calib/val splits (as in the CRC paper examples).

Uses the probabilities saved by train.py (p = P(label = 1) = P(bone age < 11y)).
Decision rule: predict positive iff p >= λ. Per-example loss on a positive:
    L_i(λ) = 1[p_i < λ]                    (missed positive, non-decreasing in λ)
so FNR = mean loss over positives. lambda_hat comes from get_lhat, copied from
the original conformal-risk repo (Angelopoulos et al., core/get_lhat.py):
    lambda_hat = largest λ on the grid with n/(n+1) * R_hat(λ) + B/(n+1) < alpha,
    n = number of calib positives, B = 1.

Each trial (same protocol as the paper's coco / polyps risk_histogram.py):
pool the examples, shuffle, first num_calib -> calib, rest -> val; get lambda_hat
from the calib positives, report val FNR (risk) and val FPR (efficiency) at lambda_hat.

The guarantee is in expectation over the calib draw, not per trial:
    alpha - 2/(n+1) <= E[val FNR] <= alpha
so the mean risk over trials should be <= alpha; single trials may exceed it.

Run from project root:
    python crc_fnr_trials.py --backbone efficientnet_b3 --seed 0
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from boneage_classification.config import OUTPUT_DIR
from boneage_classification.crc_fnr_plots import plot_histograms, plot_alpha_summary

ALPHAS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]


def get_lhat(calib_loss_table, lambdas, alpha, B=1):
    """Verbatim from conformal-risk/core/get_lhat.py.

    Gets the value of lambda hat that controls the marginal risk for a monotone
    risk function. The calib loss table should be ordered from small loss to large loss.
    """
    n = calib_loss_table.shape[0]
    rhat = calib_loss_table.mean(axis=0)
    lhat_idx = max(np.argmax(((n/(n+1)) * rhat + B/(n+1) ) >= alpha) - 1, 0) # Can't be -1.
    return lambdas[lhat_idx]


def trial_precomputed(fn_table, fp_table, is_pos, lambdas, alpha, num_calib, rng):
    """One random calib/val split -> (lambda_hat, val FNR, val FPR, val F1, n calib positives)."""
    perm = rng.permutation(len(is_pos))
    calib, val = perm[:num_calib], perm[num_calib:]
    calib_pos = calib[is_pos[calib]]
    val_pos, val_neg = val[is_pos[val]], val[~is_pos[val]]

    lhat = get_lhat(fn_table[calib_pos], lambdas, alpha)
    j = np.argmax(lambdas == lhat)
    fn, fp = fn_table[val_pos, j].sum(), fp_table[val_neg, j].sum()
    tp = len(val_pos) - fn
    f1 = 2 * tp / (2 * tp + fp + fn)
    return lhat, fn / len(val_pos), fp / len(val_neg), f1, len(calib_pos)


def experiment(p, y, lambdas, alpha, num_trials, num_calib, rng):
    is_pos = y == 1
    fn_table = (p[:, None] < lambdas[None, :]) & is_pos[:, None]    # rows = all examples
    fp_table = (p[:, None] >= lambdas[None, :]) & ~is_pos[:, None]
    rows = []
    for i in tqdm(range(num_trials), desc=f"alpha={alpha}"):
        lhat, risk, fpr, f1, n_pos = trial_precomputed(fn_table, fp_table, is_pos, lambdas,
                                                       alpha, num_calib, rng)
        rows.append({"trial": i, "alpha": alpha, "lambda_hat": lhat, "risk": risk,
                     "fpr": fpr, "f1": f1, "n_calib_pos": n_pos})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="CRC FNR, repeated calib/val trials")
    parser.add_argument("--backbone", default="efficientnet_b3")
    parser.add_argument("--seed", type=int, default=0, help="training seed (run dir)")
    parser.add_argument("--run_dir", type=Path, default=None,
                        help="override; default OUTPUT_DIR/classification/<backbone>/seed<seed>")
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    parser.add_argument("--num_trials", type=int, default=1000)
    parser.add_argument("--num_calib", type=int, default=1000,
                        help="calib examples per trial (all classes; FNR uses its positives)")
    parser.add_argument("--num_lam", type=int, default=1000, help="λ grid size on [0, 1]")
    parser.add_argument("--pool", choices=["calib+test", "calib"], default="calib+test",
                        help="examples that are shuffled into calib/val each trial")
    parser.add_argument("--rng_seed", type=int, default=0,
                        help="seed of the calib/val splits; same seed -> same results")
    args = parser.parse_args()

    run_dir = args.run_dir or OUTPUT_DIR / "classification" / args.backbone / f"seed{args.seed}"
    files = ["calib_probs.csv", "test_probs.csv"] if args.pool == "calib+test" else ["calib_probs.csv"]
    data = pd.concat([pd.read_csv(run_dir / f) for f in files], ignore_index=True)
    p, y = data["prob"].values, data["label"].values.astype(int)
    if not 0 < args.num_calib < len(p):
        raise ValueError(f"num_calib must be in (0, {len(p)})")

    out_dir = run_dir / "crc_fnr_trials"
    out_dir.mkdir(exist_ok=True)
    lambdas = np.linspace(0, 1, args.num_lam)

    print(f"\n[crc-fnr] run_dir={run_dir}")
    print(f"  pool={args.pool}: n={len(p)}  n_pos={int(y.sum())}  n_neg={int((y == 0).sum())}")
    print(f"  trials={args.num_trials}  num_calib={args.num_calib}  num_lam={args.num_lam}\n")

    summary, trials = [], []
    for alpha in args.alphas:
        # same seed per alpha -> every alpha sees the same num_trials splits (reproducible,
        # independent of which alphas are run)
        rng = np.random.default_rng(args.rng_seed)
        df = experiment(p, y, lambdas, alpha, args.num_trials, args.num_calib, rng)
        tag = f"alpha{alpha}".replace(".", "_")
        df.to_csv(out_dir / f"trials_{tag}.csv", index=False)
        plot_histograms(df, alpha, out_dir / f"histograms_{tag}.png")
        trials.append(df)
        n_pos = df["n_calib_pos"].mean()
        summary.append({"alpha": alpha,
                        "lower_bound": alpha - 2 / (n_pos + 1),
                        "risk_mean": df["risk"].mean(), "risk_std": df["risk"].std(),
                        "frac_risk_le_alpha": (df["risk"] <= alpha).mean(),
                        "fpr_mean": df["fpr"].mean(), "fpr_std": df["fpr"].std(),
                        "f1_mean": df["f1"].mean(), "f1_std": df["f1"].std(),
                        "lambda_hat_mean": df["lambda_hat"].mean(),
                        "lambda_hat_std": df["lambda_hat"].std(),
                        "n_calib_pos_mean": n_pos})

    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "summary.csv", index=False)
    plot_alpha_summary(pd.concat(trials, ignore_index=True), summary,
                       out_dir / "alpha_summary.png")
    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  saved -> {out_dir}")


if __name__ == "__main__":
    main()
