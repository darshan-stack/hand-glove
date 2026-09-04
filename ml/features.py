"""Feature extraction for the Random Forest baseline."""
import numpy as np


def extract_window_features(window):
    """Return compact statistical features from a [T, C] normalized window."""
    x = np.asarray(window, dtype=np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    minimum = x.min(axis=0)
    maximum = x.max(axis=0)
    delta = np.diff(x, axis=0) if len(x) > 1 else np.zeros_like(x)
    delta_mean = np.mean(np.abs(delta), axis=0)
    return np.concatenate([mean, std, minimum, maximum, delta_mean]).astype(np.float32)


def build_feature_matrix(windows):
    return np.stack([extract_window_features(w) for w in windows]) if len(windows) else np.empty((0, 0), dtype=np.float32)
