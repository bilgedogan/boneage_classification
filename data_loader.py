import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE

class BoneAgeDataset(Dataset):
    """Returns ({'image_input', 'sex_input'}, label) with label = 1{bone age < 11y}."""
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

        self.filepaths = df['filepath'].values
        self.sex = df['male'].values.astype('float32')
        self.label = df['label'].values.astype('float32')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        img = Image.open(path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        sex = torch.tensor([self.sex[idx]], dtype=torch.float32)
        label = torch.tensor([self.label[idx]], dtype=torch.float32)

        return {'image_input': img, 'sex_input': sex}, label


def _read_source(csv_path, img_dir):
    """Read a *_cls.csv (id,boneage,male,label), add filepath; drop missing imgs."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    out = pd.read_csv(csv_path)
    out["filepath"] = out["id"].apply(
        lambda x: str(DATA_DIR / img_dir / f"{x}.png")
    )

    exists = out["filepath"].apply(lambda p: Path(p).exists())
    missing = int((~exists).sum())
    if missing:
        print(f"  warning: {missing} rows dropped (image not found) from {csv_path.name}")
    return out[exists].reset_index(drop=True)


def load_data(sample_frac=1.0, seed=None):
    """
    Loads train_cls.csv + val_cls.csv and splits the source training data
    50/25/25 into train/val/calibration. The source validation set is used as
    the held-out TEST set.

    seed controls both subsampling and the train/val/calib split so each seed
    produces a distinct but reproducible partition.
    """
    train_full = _read_source(DATA_DIR / "train_cls.csv", "train")
    test_df = _read_source(DATA_DIR / "val_cls.csv", "val")

    if sample_frac < 1.0:
        train_full = train_full.sample(frac=sample_frac, random_state=seed)

    # 50 / 25 / 25 split, stratified by sex (same partition as the regression runs).
    train_df, temp_df = train_test_split(
        train_full,
        test_size=0.5,
        random_state=seed,
        stratify=train_full["male"],
    )
    val_df, calib_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["male"],
    )

    return train_df, val_df, calib_df, test_df


def build_datasets(train_df, val_df, batch_size=BATCH_SIZE, backbone_name=None):
    """
    Converts pandas DataFrames into torch DataLoaders.
    """
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE[backbone_name]),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE[backbone_name]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = BoneAgeDataset(train_df, transform=train_transform)
    val_ds = BoneAgeDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)

    return train_loader, val_loader


def build_val_or_test_loader(df, batch_size=BATCH_SIZE, backbone_name=None):
    """
    Converts a single DataFrame (validation, calibration, or test) into a
    torch DataLoader with eval-time transforms and no shuffling.
    """
    eval_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE[backbone_name]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds = BoneAgeDataset(df, transform=eval_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)

    return loader
