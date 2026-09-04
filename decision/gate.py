"""Safety-oriented speech decision gate.

The gate is deliberately conservative: low confidence, unstable predictions or
poor sensor quality return UNKNOWN rather than producing speech.
"""
from collections import deque, Counter
from dataclasses import dataclass


@dataclass
class GateConfig:
    confidence_threshold: float = 0.80
    agreement_ratio: float = 0.75
    history_size: int = 4


class PredictionGate:
    def __init__(self, config=None):
        self.config = config or GateConfig()
        self.history = deque(maxlen=self.config.history_size)

    def update(self, label, confidence, sensor_ok=True):
        if not sensor_ok or confidence < self.config.confidence_threshold:
            self.history.clear()
            return {"accepted": False, "label": "UNKNOWN", "reason": "low_confidence_or_sensor_quality"}
        self.history.append(str(label))
        counts = Counter(self.history)
        best, count = counts.most_common(1)[0]
        stable = count / len(self.history) >= self.config.agreement_ratio
        if stable and len(self.history) >= 2:
            return {"accepted": True, "label": best, "reason": "stable_high_confidence"}
        return {"accepted": False, "label": "UNKNOWN", "reason": "waiting_for_temporal_agreement"}
