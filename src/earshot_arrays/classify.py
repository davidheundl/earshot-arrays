"""Sound class of a block. EfficientAT comes later (`uv sync --group ml`); until it is wired in, a
spectral stand-in answers and says so on screen. Never raises: a run does not stop for a model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASSES = ("drone", "plane", "car", "person", "fan")


@dataclass(frozen=True)
class Result:
    probs: dict[str, float]
    source: str  # which classifier answered, shown next to the label

    @property
    def label(self) -> str:
        return max(self.probs, key=self.probs.get)


class StandIn:
    """Crude spectral rules, only so the pipeline and plot run end to end. Not a classifier."""

    name = "STAND-IN (not a model)"

    def __call__(self, mono: np.ndarray, fs: int) -> Result:
        try:
            return Result(self._probs(mono, fs), self.name)
        except Exception as e:  # noqa: BLE001 - a bad block must not stop the run
            return Result({c: 1.0 / len(CLASSES) for c in CLASSES}, f"{self.name}: {e}")

    @staticmethod
    def _probs(mono: np.ndarray, fs: int) -> dict[str, float]:
        spec = np.abs(np.fft.rfft(mono)) ** 2
        f = np.fft.rfftfreq(mono.size, 1.0 / fs)
        total = spec.sum() or 1.0
        low = spec[f < 300].sum() / total
        speech = spec[(f >= 300) & (f < 3000)].sum() / total
        mid = spec[(f >= 100) & (f < 4000)]
        tonal = mid.max() / max(mid.mean(), 1e-12) if mid.size else 0.0
        if tonal > 30:
            p = [0.6, 0.1, 0.05, 0.05, 0.2]
        elif low > 0.6:
            p = [0.1, 0.4, 0.4, 0.05, 0.05]
        elif speech > 0.7:
            p = [0.1, 0.05, 0.05, 0.6, 0.2]
        else:
            p = [0.2] * 5
        return dict(zip(CLASSES, p))


def load_classifier():
    """The best classifier available. For now always the stand-in (EfficientAT is not wired in yet)."""
    return StandIn()
