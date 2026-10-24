"""Audio inputs: list devices with 2+ channels, open one stream per array from its profile.

Every source (device or replay) yields raw device blocks shaped (n_channels, n_samples), float32;
the profile's channel map is applied later with `mics()`.
"""

from __future__ import annotations

import queue

import numpy as np

from earshot_arrays.config import Profile

BLOCK_S = 1.0


def list_inputs() -> list[tuple[int, int, float, str]]:
    import sounddevice as sd

    return [(i, d["max_input_channels"], d["default_samplerate"], d["name"])
            for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] >= 2]


def resolve_device(device: int | str | None) -> int | str | None:
    """An index, a digit string, or a name substring (as in `arrays list`)."""
    if device is None or isinstance(device, int):
        return device
    if str(device).isdigit():
        return int(device)
    matches = [i for i, _, _, name in list_inputs() if str(device).lower() in name.lower()]
    if len(matches) != 1:
        raise SystemExit(f"device {device!r} matches {len(matches)} inputs; use an index from `arrays list`")
    return matches[0]


def mics(block: np.ndarray, profile: Profile) -> np.ndarray:
    """(n_mics, n) in geometry order from a raw (n_channels, n) device block."""
    return block[list(profile.mic_channels)]


class DeviceSource:
    """One input stream at the profile's rate and channel count, delivering BLOCK_S blocks."""

    def __init__(self, device: int | str | None, profile: Profile, block_s: float = BLOCK_S):
        import sounddevice as sd

        self.fs = profile.fs
        self.n_channels = profile.n_channels
        self._q: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, _frames, _time, status):
            if status:
                print(f"[audio] {status}")
            self._q.put(indata.T.copy())  # PortAudio reuses the buffer

        self._stream = sd.InputStream(device=resolve_device(device), channels=profile.n_channels,
                                      samplerate=profile.fs, blocksize=int(profile.fs * block_s),
                                      dtype="float32", callback=callback)
        self._stream.start()

    def read(self, latest: bool = False, timeout: float | None = None) -> np.ndarray | None:
        """Next block (waits for it, or raises queue.Empty after `timeout` seconds). latest=True drops
        older queued blocks, so a slow consumer always sees the newest second instead of falling behind."""
        block = self._q.get(timeout=timeout)
        while latest and not self._q.empty():
            block = self._q.get_nowait()
        return block

    def drain(self) -> None:
        """Drop everything queued so far."""
        while not self._q.empty():
            self._q.get_nowait()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
