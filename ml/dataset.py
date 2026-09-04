"""Dataset utilities for the sensor-first gesture model."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REQUIRED = ["user_id", "gesture", "timestamp", "flex1", "flex2", "flex3", "flex4", "flex5", "ax", "ay", "az", "gx", "gy", "gz"]


def load_csv(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df.dropna(subset=REQUIRED).reset_index(drop=True)


def split_by_user(df, test_size=0.2, random_state=42):
    """Split whole users apart to prevent leakage between train/test."""
    groups = df["user_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df["gesture"], groups))
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    return train, test


def find_csv(root="data/raw"):
    files = sorted(Path(root).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {root}")
    return files
