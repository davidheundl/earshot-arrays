"""Sound class of a block: EfficientAT (AudioSet-pretrained MobileNet) with a trained head, if torch
and the weights load (`uv sync --group ml`, weights in data/models/); else a spectral stand-in that
says so on screen. Never raises: a run does not stop for a model.

The EfficientAT wrapper (Backbone, Head, features) is copied from the pre-built `earshot` repo
(earshot/audio/classifier.py), with its weights: mn10_as_mAP_471.pt (EfficientAT v0.0.1 release)
and sound_head.pt (trained there on ESC-50 plus drone recordings). The head's fine classes are
folded into the five shown here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CLASSES = ("drone", "plane", "car", "fan", "other")
# head's fine class -> shown class; speech, music and animals are in the head's "other"
FOLD = {"drone": "drone", "plane": "plane", "helicopter": "plane", "vehicle": "car", "train": "car",
        "machine": "fan", "insect": "other", "ambient": "other", "other": "other"}

MODELS = Path(__file__).resolve().parents[2] / "data" / "models"
BACKBONE = MODELS / "mn10_as_mAP_471.pt"
HEAD = MODELS / "sound_head.pt"
INPUT_RMS = 0.05  # the net wants ordinary audio levels
# The AudioSet models were trained on 10 s clips: on 1 s their logits blow up (edge effects
# dominate the pooled features). Tiling the window to CONTEXT_S brings them back in range.
CONTEXT_S = 5.0


@dataclass(frozen=True)
class Result:
    probs: dict[str, float]
    source: str  # which classifier answered, shown next to the label
    fine: dict[str, float] = field(default_factory=dict)  # the head's own classes, if any

    @property
    def label(self) -> str:
        return max(self.probs, key=self.probs.get)


def _uniform(source: str) -> Result:
    return Result({c: 1.0 / len(CLASSES) for c in CLASSES}, source)


class StandIn:
    """Crude spectral rules, only so the pipeline and plot run end to end. Not a classifier."""

    def __init__(self, reason: str = ""):
        self.reason = reason
        self.name = "STAND-IN (not a model)"

    def __call__(self, mono: np.ndarray, fs: int) -> Result:
        try:
            return Result(self._probs(mono, fs), self.name)
        except Exception as e:  # noqa: BLE001 - a bad block must not stop the run
            return _uniform(f"{self.name}: {e}")

    @staticmethod
    def _probs(mono: np.ndarray, fs: int) -> dict[str, float]:
        spec = np.abs(np.fft.rfft(mono)) ** 2
        f = np.fft.rfftfreq(mono.size, 1.0 / fs)
        total = spec.sum() or 1.0
        low = spec[f < 300].sum() / total
        mid = spec[(f >= 100) & (f < 4000)]
        tonal = mid.max() / max(mid.mean(), 1e-12) if mid.size else 0.0
        if tonal > 30:
            p = [0.6, 0.1, 0.05, 0.15, 0.1]
        elif low > 0.6:
            p = [0.1, 0.4, 0.4, 0.05, 0.05]
        else:
            p = [0.2] * 5
        return dict(zip(CLASSES, p))


class Backbone:
    """EfficientAT mn10_as: audio -> (AudioSet logits (527,), embedding (960,)), on the CPU."""

    def __init__(self, weights: Path = BACKBONE):
        import torch

        from earshot_arrays import efficientat

        self.torch = torch
        self.net = efficientat.load(weights)
        self.mel = efficientat.MelFrontend().eval()
        self.sr = efficientat.SR

    def prepare(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Mono at 32 kHz, RMS INPUT_RMS, tiled to CONTEXT_S."""
        from scipy.signal import resample_poly

        x = np.asarray(audio, np.float64).ravel()
        if sr != self.sr:
            g = math.gcd(sr, self.sr)
            x = resample_poly(x, self.sr // g, sr // g)
        x = x - x.mean()
        rms = math.sqrt(float(np.mean(x * x)))
        x = x * (INPUT_RMS / rms) if rms > 0 else x
        n = int(CONTEXT_S * self.sr)
        return np.tile(x, -(-n // len(x)))[:n].astype(np.float32)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Head features (1, 960 + 527): embedding plus AudioSet label probabilities."""
        torch = self.torch
        with torch.inference_mode():
            logits, emb = self.net(self.mel(torch.from_numpy(self.prepare(audio, sr)[None])))
        logits, emb = logits.numpy().astype(np.float64), emb.numpy().astype(np.float64)
        return np.concatenate([emb, 1 / (1 + np.exp(-logits))], axis=1)


class EfficientAT:
    """Frozen EfficientAT features -> the trained head's fine classes -> CLASSES."""

    name = "EfficientAT"

    def __init__(self, backbone: Path = BACKBONE, head: Path = HEAD):
        import torch
        from torch import nn

        for p in (backbone, head):
            if not p.exists():
                raise FileNotFoundError(f"{p} missing (copy it from earshot's data/models/)")
        ck = torch.load(str(head), map_location="cpu", weights_only=False)
        self.fine_names = tuple(ck["fine"])
        unknown = set(self.fine_names) - set(FOLD)
        if unknown:
            raise ValueError(f"{head}: unknown classes {sorted(unknown)}")
        self.torch = torch
        self.backbone = Backbone(backbone)
        self.mean, self.std = np.asarray(ck["mean"]), np.asarray(ck["std"])
        h = ck["hidden"]
        self.head = nn.Sequential(nn.Linear(len(self.mean), h), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(h, len(self.fine_names)))
        self.head.load_state_dict(ck["state"])
        self.head.eval()

    def __call__(self, mono: np.ndarray, fs: int) -> Result:
        if np.size(mono) == 0:
            return _uniform(f"{self.name}: empty block")
        try:
            f = self.backbone(mono, fs)
            z = self.torch.from_numpy(((f - self.mean) / self.std).astype(np.float32))
            with self.torch.inference_mode():
                fine = self.torch.softmax(self.head(z), dim=1).numpy()[0].astype(float)
        except Exception as e:  # noqa: BLE001 - a bad block must not stop the run
            return _uniform(f"{self.name}: {e}")
        probs = dict.fromkeys(CLASSES, 0.0)
        for name, p in zip(self.fine_names, fine):
            probs[FOLD[name]] += float(p)
        return Result(probs, self.name, dict(zip(self.fine_names, fine.tolist())))


def load_classifier():
    """EfficientAT if torch and the weights load, else the stand-in (with the reason in .reason)."""
    try:
        return EfficientAT()
    except Exception as e:  # noqa: BLE001 - missing torch or weights: fall back, say why
        return StandIn(f"{type(e).__name__}: {e}")
