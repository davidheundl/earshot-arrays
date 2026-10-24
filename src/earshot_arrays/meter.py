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
