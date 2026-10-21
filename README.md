# Bone age < 11y classification + Conformal Risk Control (FNR)

Binary classifier on hand X-rays (RSNA Bone Age): **label = 1 if bone age < 132 months (11 years)**.
An ImageNet-pretrained backbone + linear head is fine-tuned end-to-end, then
[Conformal Risk Control](https://arxiv.org/abs/2208.02814) (Angelopoulos et al.) picks a
decision threshold that controls the **false negative rate** (missed < 11y cases) at a target level α.

## Setup

```bash
conda create -n uq python=3.11 -y
conda activate uq
pip install -r requirements.txt
```

`requirements.txt` pins PyTorch for CUDA 12.6 (`--extra-index-url .../cu126`). For another CUDA
version or CPU only, change that line and the `torch` / `torchvision` versions to match
your setup.

Tested with Python 3.11, torch 2.14.1, torchvision 0.29.1, timm 1.0.30, scikit-learn 1.9.1.

### Paths (optional `.env`)

`config.py` reads two variables from a `.env` file in the project root (or the environment):

```
DATA_DIR=./data        # default
OUTPUT_DIR=./outputs   # default
```

Relative paths are resolved against the project root.

## Data layout

```
data/
├── train.csv          # id, boneage, male
├── val.csv            # Image ID, male, Bone Age (months)
├── train/<id>.png     # 12,611 images
└── val/<id>.png       # 1,425 images
```

The source **train** set is split 50 / 25 / 25 (stratified by sex, seeded) into
train / val / calib. The source **val** set is used as the held-out **test** set.

| split | used for |
|---|---|
| train | fitting |
| val | LR scheduling + early stopping |
| calib | CRC calibration |
| test | held-out evaluation |

## Pipeline

Run everything from the project root.

### 1. Build label files (once)

```bash
python make_label_csvs.py
```

Writes `data/train_cls.csv` and `data/val_cls.csv` with columns `id, boneage, male, label`.

### 2. Train

```bash
python train.py --backbone efficientnet_b3 --seed 0
```

Backbones: `efficientnet_b3`, `vit_b_16`, `convnextv2_tiny`.
Useful flags: `--lr` (1e-4), `--batch-size` (32), `--max-epochs` (50), `--patience` (10),
`--sample-frac` (subsample train for quick tests).

Outputs go to `outputs/classification/<backbone>/seed<seed>/`:

| file | content |
|---|---|
| `best_model.pth` | best weights (lowest val BCE) |
| `history.csv` | train / val loss and LR per epoch |
| `{val,calib,test}_probs.csv` | `id, boneage, male, label, prob` where `prob = P(label = 1)` |
| `head_metrics.csv` | AUC, log loss, Brier, and accuracy / FNR / FPR / F1 at 0.5, per split |

All calibration below works from the saved `*_probs.csv` files, so you never need to retrain to recalibrate.

### 3. CRC for FNR (repeated trials, as in the CRC paper)

```bash
python crc_fnr_trials.py --backbone efficientnet_b3 --seed 0
```

| flag | default | meaning |
|---|---|---|
| `--alphas` | `0.01 0.05 0.1 0.15 0.2 0.25` | target FNR levels |
| `--num_trials` | 1000 | random calib/val splits per α |
| `--num_calib` | 1000 | calib examples per trial (all classes) |
| `--num_lam` | 1000 | λ grid size, `linspace(0, 1, num_lam)` |
| `--pool` | `calib+test` | examples shuffled into calib/val each trial (`calib` = calib split only) |
| `--rng_seed` | 0 | seed of the splits; same seed gives identical results |
| `--run_dir` | — | read probs from another directory |

Outputs go to `outputs/classification/<backbone>/seed<seed>/crc_fnr_trials/`:

| file | content |
|---|---|
| `trials_alpha*.csv` | one row per trial: `lambda_hat`, `risk` (val FNR), `fpr`, `f1`, `n_calib_pos` |
| `summary.csv` | per α: `lower_bound`, `risk_mean/std`, `frac_risk_le_alpha`, `fpr_mean/std`, `f1_mean/std`, `lambda_hat_mean/std` |
| `histograms_alpha*.png` | per α: histograms of val FNR, val FPR, λ̂ over trials |
| `alpha_summary.png` | all α side by side: λ̂ and val FNR distributions with α and the lower bound |

## How the CRC step works

- **Rule:** predict positive iff `p >= λ`. Loss on a positive example: `L_i(λ) = 1[p_i < λ]`.
  It does not decrease as λ grows, and its mean over positives is the FNR.
- **λ̂:** `get_lhat` is copied verbatim from the original repo
  ([aangelopoulos/conformal-risk](https://github.com/aangelopoulos/conformal-risk), `core/get_lhat.py`).
  It returns the largest λ on the grid where `n/(n+1) · R̂(λ) + 1/(n+1) < α`, with
  `n` = number of positives in the calib set.
- **One trial** follows the protocol of the paper's COCO and polyp examples:
  1. Pool the examples and shuffle them.
  2. The first `num_calib` examples form the calib set; the rest form the val set.
  3. Compute λ̂ from the calib positives.
  4. Measure FNR, FPR and F1 on the val set at λ̂.

  Each α uses the same set of splits.
- **The guarantee is in expectation over the calib draw, not per trial:**

  ```
  α − 2/(n+1)  ≤  E[FNR]  ≤  α
  ```

  `risk_mean` estimates E[FNR] and should fall inside this interval. A single trial can
  exceed α, so `frac_risk_le_alpha` of roughly 0.5 to 0.6 is expected. A per-split
  high-probability guarantee needs RCPS/LTT, not CRC.
- **Pooling calib + test** assumes both are exchangeable (drawn from the same distribution).
  Use `--pool calib` to stay within the calib split.

## Files

| file | role |
|---|---|
| `config.py` | paths, image sizes, batch size |
| `make_label_csvs.py` | builds `*_cls.csv` label files |
| `data_loader.py` | splits and DataLoaders |
| `model.py` | backbone + linear head (image + sex → logit) |
| `train.py` | training, saves probabilities and metrics |
| `crc_fnr_trials.py` | CRC (FNR) with repeated trials |
| `crc_fnr_plots.py` | plots for `crc_fnr_trials.py` |