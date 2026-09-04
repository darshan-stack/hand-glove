"""Sensor preprocessing utilities for the smart-glove ML pipeline.

This file is hardware-independent: it operates on numpy arrays/dataframes so
we can train and test the ML system on recorded CSV data before connecting the
Raspberry Pi.
"""
from dataclasses import dataclass
import numpy as np

SENSOR_COLUMNS = [
    "flex1", "flex2", "flex3", "flex4", "flex5",
    "ax", "ay", "az", "gx", "gy", "gz",
]


@dataclass
class Calibration:
    """Per-user sensor baseline and scale."""
    center: np.ndarray
    scale: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - self.center) / np.maximum(self.scale, 1e-6)


def fit_calibration(x: np.ndarray) -> Calibration:
    """Fit robust per-sensor center/scale from a user's calibration samples."""
    x = np.asarray(x, dtype=np.float32)
    center = np.median(x, axis=0)
    q75 = np.percentile(x, 75, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    scale = q75 - q25
    # Prevent near-constant sensors from exploding during normalization.
    scale = np.where(scale < 1e-3, 1.0, scale)
    return Calibration(center=center, scale=scale)


def add_motion_features(x: np.ndarray) -> np.ndarray:
    """Append acceleration/gyro magnitudes to a [T, 11] sensor sequence."""
    x = np.asarray(x, dtype=np.float32)
    acc_mag = np.linalg.norm(x[:, 5:8], axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(x[:, 8:11], axis=1, keepdims=True)
    return np.concatenate([x, acc_mag, gyro_mag], axis=1)


def make_windows(x: np.ndarray, labels: np.ndarray, window: int = 32, stride: int = 8):
    """Create overlapping fixed-length sequences without crossing label changes."""
    x = np.asarray(x, dtype=np.float32)
    labels = np.asarray(labels)
    windows, y = [], []
    for start in range(0, len(x) - window + 1, stride):
        end = start + window
        chunk_labels = labels[start:end]
        if len(set(chunk_labels.tolist())) != 1:
            continue
        windows.append(x[start:end])
        y.append(chunk_labels[0])
    if not windows:
        return np.empty((0, window, x.shape[1]), dtype=np.float32), np.array([])
    return np.stack(windows), np.asarray(y)
