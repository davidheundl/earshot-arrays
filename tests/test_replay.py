import numpy as np
import pytest

from earshot_arrays.classify import CLASSES, StandIn, load_classifier
from earshot_arrays.config import load_setup
from earshot_arrays.devices import mics
from earshot_arrays.doa import estimate_bearing
from earshot_arrays.locate import intersect
from earshot_arrays.replay import ReplaySource, replay_files, write_wav
from earshot_arrays.synth import synth_session


def test_wav_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    blocks = [rng.uniform(-0.5, 0.5, (6, 16000)).astype(np.float32) for _ in range(3)]
    write_wav(tmp_path / "A.wav", blocks, 16000)
    src = ReplaySource(tmp_path / "A.wav")
    assert (src.fs, src.n_channels) == (16000, 6)
    for b in blocks:
        np.testing.assert_array_equal(src.read(), b)
    assert src.read() is None


def test_synthetic_session_replays_to_the_source(tmp_path):
    base = load_setup("setup.example.yaml")
    out = synth_session(base, [(0.8, 0.6)], tmp_path / "s", seconds_per_point=2)
    s = load_setup(out / "setup.yaml")  # the copy inside the recording, with its own profiles
    assert [a.name for a in s.arrays] == ["A", "B"]
    files = replay_files([str(out)], [a.name for a in s.arrays])
    sources = [ReplaySource(f) for f in files]
    n = 0
    while True:
        blocks = [src.read() for src in sources]
        if any(blk is None for blk in blocks):
            break
        b = [estimate_bearing(mics(blk, a.profile), a.profile.fs, a.profile.mic_xy, a.pose.rotation_deg)[0]
             for a, blk in zip(s.arrays, blocks)]
        fix = intersect(s.arrays[0].pose, b[0], s.arrays[1].pose, b[1])
        assert fix is not None
        assert (fix.x, fix.y) == pytest.approx((0.8, 0.6), abs=0.1)
        n += 1
    assert n == 2


def test_replay_files_needs_one_per_array(tmp_path):
    with pytest.raises(SystemExit):
        replay_files([str(tmp_path / "only.wav")], ["A", "B"])


def test_stand_in_says_so_and_never_raises():
    clf = StandIn()
    r = clf(np.zeros(16000, dtype=np.float32), 16000)
    assert "STAND-IN" in r.source
    assert set(r.probs) == set(CLASSES)
    assert clf(np.array([]), 16000).label in CLASSES


def test_loaded_classifier_never_raises():
    clf = load_classifier()  # EfficientAT if torch + weights are there, else the stand-in
    rng = np.random.default_rng(0)
    for x in (0.1 * rng.standard_normal(48000), np.zeros(48000), np.array([])):
        r = clf(x.astype(np.float32), 48000)
        assert set(r.probs) == set(CLASSES)
        assert sum(r.probs.values()) == pytest.approx(1.0, abs=1e-6)


def test_train_label_of_recording_dir():
    from earshot_arrays.train import label_of

    assert label_of("drone12") == "drone"
    assert label_of("quiet1") == "quiet"
