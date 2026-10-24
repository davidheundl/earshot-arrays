import dataclasses
import math
from pathlib import Path

import numpy as np

from earshot_arrays.belief import bearing_bins
from earshot_arrays.cli import Tracker, office_arrays
from earshot_arrays.config import ArraySetup, Pose, Setup, load_profile, load_setup
from earshot_arrays.locate import bearing_to
from earshot_arrays.office import load_office
from earshot_arrays.replay import ReplaySource, replay_files
from earshot_arrays.synth import synth_session

ROOT = Path(__file__).resolve().parents[1]
UMA16 = load_profile(ROOT / "profiles/uma16.yaml")
SECS = 2


def run_session(setup_path: Path, office, out: Path) -> list:
    """Replay a session through the twin's per-second step; the estimate after every block."""
    s = load_setup(setup_path)
    arrays = office_arrays(s, office)
    tracker = Tracker(arrays, office.grid())
    sources = [ReplaySource(f) for f in replay_files([str(out)], [a.name for a in arrays])]
    ests = []
    while (blocks := [src.read() for src in sources]) and all(b is not None for b in blocks):
        ests.append(tracker.step(blocks)[1])
    return ests


def test_twin_two_arrays_office_scale(tmp_path):
    office = load_office(ROOT / "office.example.yaml")
    s = Setup((1.6, 0.8), tuple(ArraySetup(n, UMA16, None, Pose(0.0, 0.0, 0.0)) for n in ("A", "B")))
    s = dataclasses.replace(s, arrays=office_arrays(s, office))
    # a walk around the long table, off the arrays' baseline (y = 2.8), 1.5-1.8 m per step: the belief
    # moves about 0.5 m/s (motion blur), so a 3 m jump would need 3-4 s to catch up, not 2
    path = [(5.0, 6.0), (6.5, 7.0), (8.0, 6.5), (9.5, 5.5)]
    out = synth_session(s, path, tmp_path / "office", SECS)
    ests = run_session(out / "setup.yaml", office, out)
    assert len(ests) == len(path) * SECS
    for i, (x, y) in enumerate(path):
        est = ests[(i + 1) * SECS - 1]  # end of the point's dwell
        assert math.hypot(est.x - x, est.y - y) < 1.0, (i, (x, y), est)


def test_twin_one_array_fan_along_bearing(tmp_path):
    office = load_office(ROOT / "office.example.yaml")
    s = Setup((1.6, 0.8), (ArraySetup("A", UMA16, None, Pose(*office.arrays["A"], 0.0)),))
    src = (8.0, 7.0)
    out = synth_session(s, [src], tmp_path / "one", 3)
    s = load_setup(out / "setup.yaml")
    tracker = Tracker(s.arrays, office.grid())
    for f in replay_files([str(out)], ["A"]):
        source = ReplaySource(f)
        while (block := source.read()) is not None:
            rays, _, _ = tracker.step([block])
    pose = s.arrays[0].pose
    assert rays[0][1] > 1.3
    bins = bearing_bins(tracker.belief.grid, pose)
    k = int(bearing_to(pose, *src) // 5)
    along = np.isin(bins, [(k + d) % 72 for d in (-1, 0, 1)])
    assert tracker.belief.p[along].sum() > 0.5


def test_mocked_array_crosses_the_real_ray_and_stays_silent_when_quiet():
    from earshot_arrays.belief import Grid
    from earshot_arrays.synth import array_block, drone_like

    real = ArraySetup("B", UMA16, None, Pose(16.05, 6.74, 270.0))
    mock = ArraySetup("A", UMA16, None, Pose(16.05, 9.46, 270.0))
    src = (12.0, 8.0)
    tr = Tracker([real, mock], Grid.from_bounds(8.0, 3.0, 17.0, 13.0, 0.25), mock={1}, mock_source=src)
    rng = np.random.default_rng(1)
    for _ in range(3):
        block = array_block(src, real.pose, UMA16, drone_like(UMA16.fs, UMA16.fs, rng), rng)
        rays, est, _ = tr.step([block, None])
    assert abs((rays[1][0] - bearing_to(mock.pose, *src) + 180) % 360 - 180) < 10
    assert rays[1][1] == rays[0][1]  # as clear as the real array
    assert math.hypot(est.x - src[0], est.y - src[1]) < 0.6
    quiet = (0.001 * rng.standard_normal((UMA16.n_channels, UMA16.fs))).astype(np.float32)
    rays, _, _ = tr.step([quiet, None])
    assert rays[1][1] == rays[0][1] < 1.3  # nothing clear heard: the mock says nothing either
