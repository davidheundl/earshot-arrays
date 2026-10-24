"""Live per-channel level bars in the terminal. Tap each mic and watch which channel jumps."""

from __future__ import annotations

import sys

import numpy as np

FLOOR_DB = -80.0
WIDTH = 50


def channel_db(block: np.ndarray) -> np.ndarray:
    """RMS level per channel in dBFS, (n_channels,)."""
    rms = np.sqrt(np.mean(np.square(block, dtype=np.float64), axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-12))


def bars(levels_db: np.ndarray, mic_channels: tuple[int, ...] = ()) -> list[str]:
    """One line per channel; mapped mic channels carry their mic index."""
    loudest = int(np.argmax(levels_db))
    lines = []
    for ch, db in enumerate(levels_db):
        n = int(WIDTH * np.clip((db - FLOOR_DB) / -FLOOR_DB, 0.0, 1.0) + 0.5)
        tag = f"mic {mic_channels.index(ch)}" if ch in mic_channels else "     "
        mark = " <" if ch == loudest else ""
        lines.append(f"ch {ch:2d} {tag} |{'#' * n}{' ' * (WIDTH - n)}| {db:6.1f} dBFS{mark}")
    return lines


def run(source, mic_channels: tuple[int, ...] = (), out=sys.stdout) -> None:
    first = True
    while (block := source.read(latest=True)) is not None:
        lines = bars(channel_db(block), mic_channels)
        if not first:
            out.write(f"\x1b[{len(lines)}F")  # back to the first bar line
        out.write("\n".join(lines) + "\n")
        out.flush()
        first = False


def tap_channel(block: np.ndarray, noise_rms: np.ndarray, min_snr_db: float = 30.0,
                min_margin_db: float = 6.0) -> int | None:
    """The channel a tap landed on: the one whose peak stands out from its own noise floor by
    min_snr_db and beats the runner-up by min_margin_db. None if no clear tap in the block."""
    peak = np.abs(block).max(axis=1)
    snr = 20.0 * np.log10(np.maximum(peak, 1e-12) / np.maximum(noise_rms, 1e-12))
    order = np.argsort(snr)[::-1]
    best, second = snr[order[0]], snr[order[1]] if len(order) > 1 else -np.inf
    if best < min_snr_db or best - second < min_margin_db:
        return None
    return int(order[0])


def map_channels(source, n_mics: int, settle_s: float = 2.0, out=sys.stdout) -> list[int]:
    """Prompt for a tap on each mic position 0..n_mics-1 and return the device channel of each.
    `source` must deliver short blocks (~0.1 s)."""
    block_s = None
    quiet = []
    t = 0.0
    out.write(f"measuring the noise floor for {settle_s:.0f} s, keep quiet...\n")
    while t < settle_s:
        b = source.read()
        block_s = b.shape[1] / source.fs
        quiet.append(b)
        t += block_s
    noise_rms = np.sqrt(np.mean(np.square(np.concatenate(quiet, axis=1), dtype=np.float64), axis=1))
    channels: list[int] = []
    while len(channels) < n_mics:
        k = len(channels)
        out.write(f"tap mic {k:2d} ... ")
        out.flush()
        source.drain()
        while (ch := tap_channel(source.read(), noise_rms)) is None:
            pass
        if ch in channels:
            out.write(f"channel {ch} is already mic {channels.index(ch)}; tap mic {k} again\n")
        else:
            out.write(f"channel {ch}\n")
            channels.append(ch)
        _skip(source, 0.8 / block_s)  # let the tap ring out
    return channels


def _skip(source, n_blocks: float) -> None:
    for _ in range(int(n_blocks + 0.5)):
        source.read()
