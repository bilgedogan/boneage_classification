#!/usr/bin/env python3
"""Fine-tunes an ImageNet backbone + linear head (BCE) for bone age < 11y.

The whole model is trained end-to-end (as in regression), so val is passed
through the full model every epoch.

train split -> fit, val split -> LR scheduling + early stopping. Probabilities
for val / calib / test are saved so any calibration method (CRC, CLS, ...) can
be applied afterwards, independently of the model.

Run from project root:
    python train.py --backbone efficientnet_b3 --seed 0
"""

import sys
import time
import argparse
import copy
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from boneage_classification.config import OUTPUT_DIR, BATCH_SIZE
from boneage_classification.data_loader import load_data, build_datasets, build_val_or_test_loader
from boneage_classification.model import BoneAgeClassifier, BACKBONES


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_logits(model, loader, device):
    """Eval-mode logits and labels of a loader -> (logits, y) on device."""
    model.eval()
    logits, ys = [], []
    for inputs, y in loader:
        logits.append(model(inputs["image_input"].to(device), inputs["sex_input"].to(device)))
        ys.append(y.to(device))
    return torch.cat(logits), torch.cat(ys)


def binary_metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    pos, neg = y == 1, y == 0
    return {
        "auc": roc_auc_score(y, p),
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        f"acc@{threshold}": (pred == y).mean(),
        f"fnr@{threshold}": (pred[pos] == 0).mean(),
        f"fpr@{threshold}": (pred[neg] == 1).mean(),
        f"f1@{threshold}": f1_score(y, pred),
    }


def train(model, train_loader, val_loader, device, lr, weight_decay, max_epochs, patience):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()

    best_loss, best_state, bad_epochs = float("inf"), None, 0
    history = {"loss": [], "val_loss": [], "lr": []}
    for epoch in range(max_epochs):
        t0 = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        for inputs, y in train_loader:
            img, sex, y = (inputs["image_input"].to(device),
                           inputs["sex_input"].to(device), y.to(device))
            loss = loss_fn(model(img, sex), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)

        val_loss = loss_fn(*predict_logits(model, val_loader, device)).item()
        current_lr = opt.param_groups[0]["lr"]
        history["loss"].append(total / n)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        print(f"  epoch {epoch + 1:3d}  loss={total / n:.4f}  val_loss={val_loss:.4f}  "
              f"lr={current_lr:.1e}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        scheduler.step(val_loss)

        if val_loss < best_loss - 1e-5:
            best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    print(f"  stopped at epoch {epoch + 1}, best val BCE={best_loss:.4f}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Fine-tuned classifier for bone age < 11y")
    parser.add_argument("--backbone", default="efficientnet_b3", choices=BACKBONES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10,
                        help="early stopping; > scheduler patience (5) so LR gets reduced first")
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="subsample the source train set (quick tests)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"\n[cls] backbone={args.backbone}  seed={args.seed}  label: boneage < 132 months")
    train_df, val_df, calib_df, test_df = load_data(sample_frac=args.sample_frac, seed=args.seed)
    splits = {"train": train_df, "val": val_df, "calib": calib_df, "test": test_df}
    for name, df in splits.items():
        print(f"  {name:5s}: n={len(df):5d}  pos={int(df['label'].sum()):5d} "
              f"({df['label'].mean():.3f})")

    device = get_device()
    print(f"  Device: {device}")
    model = BoneAgeClassifier(args.backbone).to(device)

    train_loader, val_loader = build_datasets(train_df, val_df, batch_size=args.batch_size,
                                              backbone_name=args.backbone)
    history = train(model, train_loader, val_loader, device, args.lr, args.weight_decay,
                    args.max_epochs, args.patience)

    out_dir = OUTPUT_DIR / "classification" / args.backbone / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_model.pth")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index_label="epoch")

    rows = []
    for name in ("val", "calib", "test"):
        loader = build_val_or_test_loader(splits[name], batch_size=args.batch_size,
                                          backbone_name=args.backbone)
        logits, y = predict_logits(model, loader, device)
        p = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        y = y.squeeze(1).cpu().numpy().astype(int)

        pred_df = splits[name][["id", "boneage", "male", "label"]].copy().reset_index(drop=True)
        pred_df["prob"] = p
        pred_df.to_csv(out_dir / f"{name}_probs.csv", index=False)

        m = binary_metrics(y, p)
        rows.append({"backbone": args.backbone, "seed": args.seed, "split": name, "n": len(y), **m})
        print(f"  {name:5s}: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    pd.DataFrame(rows).to_csv(out_dir / "head_metrics.csv", index=False)
    print(f"  saved -> {out_dir}")


if __name__ == "__main__":
    main()
