"""Synthetic recordings: a source at room positions, heard by each array with the exact (spherical)
delays to every mic. For tests and for rehearsing `locate --replay` without hardware."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from earshot_arrays.config import Pose, Profile, Setup, write_setup
from earshot_arrays.doa import BAND, C
from earshot_arrays.replay import write_wav


def mics_in_room(pose: Pose, mic_xy: np.ndarray) -> np.ndarray:
    """(2, n_mics) room positions of an array's mics."""
    r = math.radians(pose.rotation_deg)
    fwd = np.array([math.sin(r), math.cos(r)])
    right = np.array([math.cos(r), -math.sin(r)])
    return np.array([pose.x, pose.y])[:, None] + right[:, None] * mic_xy[0] + fwd[:, None] * mic_xy[1]


def drone_like(n: int, fs: int, rng: np.random.Generator, f0: float = 180.0) -> np.ndarray:
    """Band-limited noise plus a rotor-like harmonic series, unit RMS."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    spec[(f < BAND[0]) | (f > BAND[1])] *= 0.05
    sig = np.fft.irfft(spec, n)
    t = np.arange(n) / fs
    sig += sum(0.5 / k * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi)) for k in range(1, 20))
    return sig / np.sqrt(np.mean(sig**2))


def delayed(sig: np.ndarray, delays_s: np.ndarray, fs: int) -> np.ndarray:
    """(n_delays, n) copies of sig, each delayed by a fractional number of samples (circular FFT shift)."""
    n = sig.size
    spec = np.fft.rfft(sig)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return np.fft.irfft(spec[None, :] * np.exp(-2j * np.pi * f[None, :] * delays_s[:, None]), n)


def array_block(source_xy: tuple[float, float], pose: Pose, profile: Profile, sig: np.ndarray,
                rng: np.random.Generator, snr_db: float = 20.0) -> np.ndarray:
    """Raw device block (n_channels, n) for one array: delayed source on the mic channels, noise elsewhere."""
    room = mics_in_room(pose, profile.mic_xy)
    dist = np.hypot(room[0] - source_xy[0], room[1] - source_xy[1])
    mic_sig = delayed(sig, dist / C, profile.fs)
    noise = 10 ** (-snr_db / 20.0)
    block = noise * rng.standard_normal((profile.n_channels, sig.size))
    block[list(profile.mic_channels)] += mic_sig
    return (0.1 * block).astype(np.float32)


def synth_session(setup: Setup, path: list[tuple[float, float]], out_dir: str | Path,
                  seconds_per_point: int = 2, seed: int = 0) -> Path:
    """Write a recording directory (as `record` does) with the source visiting each point in turn."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    blocks: dict[str, list[np.ndarray]] = {a.name: [] for a in setup.arrays}
    for xy in path:
        for _ in range(seconds_per_point):
            for a in setup.arrays:
                sig = drone_like(a.profile.fs, a.profile.fs, rng)
                blocks[a.name].append(array_block(xy, a.pose, a.profile, sig, rng))
    for a in setup.arrays:
        write_wav(out / f"{a.name}.wav", blocks[a.name], a.profile.fs)
    write_setup(setup, out)
    return out
