"""User-specific adaptation primitives.

This stores calibration statistics and validated gesture prototypes. It is a
starting point for personalization experiments, not a claim of a patentable
algorithm.
"""
from dataclasses import dataclass, field
import numpy as np
from preprocessing.pipeline import fit_calibration


@dataclass
class UserProfile:
    user_id: str
    calibration_center: list = field(default_factory=list)
    calibration_scale: list = field(default_factory=list)
    prototypes: dict = field(default_factory=dict)

    def fit_calibration(self, samples):
        cal = fit_calibration(np.asarray(samples, dtype=np.float32))
        self.calibration_center = cal.center.tolist()
        self.calibration_scale = cal.scale.tolist()

    def normalize(self, samples):
        x = np.asarray(samples, dtype=np.float32)
        center = np.asarray(self.calibration_center, dtype=np.float32)
        scale = np.asarray(self.calibration_scale, dtype=np.float32)
        if center.size == 0:
            raise ValueError("Profile has no calibration")
        return (x - center) / np.maximum(scale, 1e-6)

    def add_validated_prototype(self, label, embedding):
        self.prototypes[str(label)] = np.asarray(embedding, dtype=np.float32).tolist()
