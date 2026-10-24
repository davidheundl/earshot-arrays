import argparse
from pathlib import Path

import numpy as np
import pytest

from earshot_arrays import livemap

N_CH, FS = 16, 48000


def test_block_problem():
    ok = np.random.default_rng(0).standard_normal((N_CH, FS)).astype(np.float32)
    assert livemap.block_problem(ok, N_CH, FS) == ""
    assert "shape" in livemap.block_problem(ok[:, : FS // 2], N_CH, FS)
    bad = ok.copy()
    bad[3, 10] = np.nan
    assert "non-finite" in livemap.block_problem(bad, N_CH, FS)
    assert "constant" in livemap.block_problem(np.full((N_CH, FS), 0.01, np.float32), N_CH, FS)


class FakeDevice:
    """Good blocks, then broken ones (first instance) or a Ctrl-C (later instances)."""

    opened = 0

    def __init__(self, device, profile, block_s=1.0):
        FakeDevice.opened += 1
        self.n, self.rng, self.first = 0, np.random.default_rng(FakeDevice.opened), FakeDevice.opened == 1
        self.fs, self.n_channels = profile.fs, profile.n_channels

    def read(self, latest=False, timeout=None):
        self.n += 1
        if self.n > 3 and not self.first:
            raise KeyboardInterrupt
        block = 0.01 * self.rng.standard_normal((N_CH, FS)).astype(np.float32)
        if self.n > 3:
            block[:, ::2] = np.nan  # the stream went bad
        return block

    def close(self):
        pass


def test_live_map_reopens_a_device_that_starts_delivering_garbage(tmp_path, monkeypatch):
    profile = Path("profiles/uma16.yaml").resolve()
    (tmp_path / "setup.yaml").write_text(f"arrays:\n  - {{name: A, profile: {profile}, device: UMA16, pose: [0, 0, 0]}}\n")
    from earshot_arrays import cli, devices

    monkeypatch.setattr(devices, "DeviceSource", FakeDevice)
    monkeypatch.setattr(cli, "open_sources", lambda arrays, a: [FakeDevice(x.device, x.profile) for x in arrays])
    monkeypatch.setattr(livemap.LiveMapServer, "__init__", lambda self, config, port=0: setattr(self, "url", "x"))
    monkeypatch.setattr(livemap.LiveMapServer, "publish", lambda self, state: None)
    monkeypatch.setattr(livemap.LiveMapServer, "close", lambda self: None)
    FakeDevice.opened = 0
    a = argparse.Namespace(setup=str(tmp_path / "setup.yaml"), replay=None, realtime=False, radius=4.0, port=0,
                           mix=False, fast=False, no_browser=True, doa_s=2.0, gate=None, min_prob=None, loudest=0.3)
    livemap.cmd_live(a)  # returns when the reopened fake device sends Ctrl-C
    assert FakeDevice.opened == 2


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
