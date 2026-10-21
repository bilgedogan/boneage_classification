"""One-off: writes data/{train,val}_cls.csv with columns id, boneage, male, label.

label = 1 if boneage < 132 months (11 years), else 0.
"""

from pathlib import Path

import pandas as pd

THRESHOLD_MONTHS = 132  # 11 years
DATA_DIR = Path(__file__).resolve().parent / "data"

# raw csv -> (id col, boneage col, sex col); the two files use different headers.
SOURCES = {
    "train": ("train.csv", "id", "boneage", "male"),
    "val": ("val.csv", "Image ID", "Bone Age (months)", "male"),
}

if __name__ == "__main__":
    for split, (fname, id_col, age_col, sex_col) in SOURCES.items():
        df = pd.read_csv(DATA_DIR / fname)
        df = df.rename(columns={id_col: "id", age_col: "boneage", sex_col: "male"})
        df = df[["id", "boneage", "male"]]
        df["label"] = (df["boneage"] < THRESHOLD_MONTHS).astype(int)
        out = DATA_DIR / f"{split}_cls.csv"
        df.to_csv(out, index=False)
        print(f"{split:5s}: n={len(df):5d}  pos={df['label'].sum():5d} "
              f"({df['label'].mean():.3f})  -> {out}")
