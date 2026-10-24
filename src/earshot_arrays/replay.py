"""Recordings: one WAV per array with all raw device channels, plus a copy of the setup.

    recordings/<session>/setup.yaml, profiles/<name>.yaml, <name>.wav

`record` writes this directory; every live command's `--replay` reads it (or plain WAV files).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

from earshot_arrays.devices import BLOCK_S


class ReplaySource:
    """A multichannel WAV in place of a device: same blocks, same shape (n_channels, n)."""

    def __init__(self, path: str | Path, block_s: float = BLOCK_S, realtime: bool = False):
        data, self.fs = sf.read(str(path), dtype="float32", always_2d=True)
        self.n_channels = data.shape[1]
        self._data = data.T
        self._n = int(self.fs * block_s)
        self._pos = 0
        self._realtime = realtime
        self._block_s = block_s

    def read(self, latest: bool = False) -> np.ndarray | None:
        """Next full block, or None at the end of the file."""
        if self._pos + self._n > self._data.shape[1]:
            return None
        block = self._data[:, self._pos:self._pos + self._n]
        self._pos += self._n
        if self._realtime:
            time.sleep(self._block_s)
        return block

    def close(self) -> None:
        pass


def write_wav(path: str | Path, blocks: list[np.ndarray] | np.ndarray, fs: int) -> None:
    """(n_channels, n) blocks → one float WAV, frames × channels."""
    data = np.concatenate(blocks, axis=1) if isinstance(blocks, list) else blocks
    sf.write(str(path), data.T.astype(np.float32), fs, subtype="FLOAT")


def replay_files(replay: list[str], names: list[str]) -> list[Path]:
    """Map --replay arguments to one WAV per array: a recording directory (<name>.wav for each
    array) or WAV files in setup order."""
    if len(replay) == 1 and Path(replay[0]).is_dir():
        files = [Path(replay[0]) / f"{n}.wav" for n in names]
    else:
        files = [Path(p) for p in replay]
    if len(files) < len(names):
        raise SystemExit(f"--replay needs {len(names)} WAV files (one per array), got {len(files)}")
    for f in files:
        if not f.exists():
            raise SystemExit(f"replay file not found: {f}")
    return files[:len(names)]
