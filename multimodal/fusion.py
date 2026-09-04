"""Simple late-fusion utilities for sensor and vision probabilities.

The vision model itself is intentionally pluggable. This keeps the core sensor
pipeline deployable on Raspberry Pi 4 while allowing a camera model later.
"""
import numpy as np


def fuse_probabilities(sensor_probs, vision_probs=None, sensor_weight=0.75):
    sensor = np.asarray(sensor_probs, dtype=np.float32)
    if vision_probs is None:
        return sensor
    vision = np.asarray(vision_probs, dtype=np.float32)
    if sensor.shape != vision.shape:
        raise ValueError("Sensor and vision probability vectors must have the same classes")
    w = float(np.clip(sensor_weight, 0.0, 1.0))
    fused = w * sensor + (1.0 - w) * vision
    total = fused.sum()
    return fused / total if total > 0 else fused


def fused_prediction(sensor_probs, vision_probs=None, sensor_weight=0.75):
    probs = fuse_probabilities(sensor_probs, vision_probs, sensor_weight)
    idx = int(np.argmax(probs))
    return idx, float(probs[idx])
