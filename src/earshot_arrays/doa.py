"""Bearing of the loudest source from one array's block. Pure, no audio device.

    bearing, power = estimate_bearing(block, fs, mic_xy, rotation_deg)   # room compass degrees

SRP-PHAT copied from `earshot/audio/doa.py` (the Earshot swarm simulation, pre-built), with the
same compass convention: mic_xy in metres in the array's frame (x right, y forward); bearings
clockwise from the room's north; `rotation_deg` is the room bearing of the array's forward axis.
"""

from __future__ import annotations

import numpy as np
import pyroomacoustics as pra

C = 343.0  # m/s
BAND = (500.0, 4000.0)  # Hz; below 500 Hz a few-cm array cannot resolve direction
NFFT = 512
SCAN_BINS = 72  # 5 deg


def estimate_bearing(block: np.ndarray, fs: int, mic_xy: np.ndarray, rotation_deg: float = 0.0,
                     band: tuple[float, float] = BAND, scan: list | None = None) -> tuple[float, float]:
    """(room bearing in degrees, relative peak power) of the loudest source in the block.

    block: (n_mics, n_samples) float, channels in the order of mic_xy (2, n_mics). Power is
    peak / mean of the SRP map (1 = no direction stands out). If `scan` is a list, it receives
    the normalised SRP power over SCAN_BINS room bearings (every 5 deg, from north)."""
    if mic_xy.shape[0] != 2 or mic_xy.shape[1] != block.shape[0]:
        raise ValueError(f"mic_xy must be (2, {block.shape[0]}), got {mic_xy.shape}")
    hi = min(band[1], 0.45 * fs)
    stft = np.stack([pra.transform.stft.analysis(ch, NFFT, NFFT // 2).T for ch in block])
    doa = pra.doa.algorithms["SRP"](mic_xy, fs, NFFT, c=C, num_src=1)
    doa.locate_sources(stft, freq_range=[band[0], hi])
    az = float(np.degrees(doa.azimuth_recon[0]))  # counter-clockwise from the array's +x axis
    grid = np.asarray(doa.grid.values, dtype=float)
    power = float(grid.max() / max(grid.mean(), 1e-12))
    if scan is not None:
        compass = (90.0 - np.degrees(np.asarray(doa.grid.azimuth)) + rotation_deg) % 360.0
        bins = np.zeros(SCAN_BINS)
        idx = (compass // (360.0 / SCAN_BINS)).astype(int) % SCAN_BINS
        np.maximum.at(bins, idx, grid / max(grid.max(), 1e-12))
        scan[:] = [float(v) for v in bins]
    return array_to_room(az, rotation_deg), power


def array_to_room(azimuth_deg: float, rotation_deg: float) -> float:
    """Math azimuth in the array frame (CCW from +x) → room compass bearing."""
    return (90.0 - azimuth_deg + rotation_deg) % 360.0


def angle_diff(a: float, b: float) -> float:
    """Signed smallest difference a - b in degrees, in (-180, 180]."""
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d
