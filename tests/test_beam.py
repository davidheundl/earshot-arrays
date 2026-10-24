import math

import numpy as np

from earshot_arrays.config import Pose, load_profile
from earshot_arrays.devices import mics
from earshot_arrays.doa import beam
from earshot_arrays.synth import array_block, drone_like


def test_beam_toward_the_source_keeps_more_of_it_than_away():
    prof = load_profile("profiles/uma16.yaml")
    rng = np.random.default_rng(0)
    pose = Pose(0.0, 0.0, 30.0)
    src_b = 120.0
    src = (3 * math.sin(math.radians(src_b)), 3 * math.cos(math.radians(src_b)))
    m = mics(array_block(src, pose, prof, drone_like(prof.fs, prof.fs, rng), rng, snr_db=60), prof)
    toward = beam(m, prof.fs, prof.mic_xy, src_b, pose.rotation_deg)
    away = beam(m, prof.fs, prof.mic_xy, src_b + 180, pose.rotation_deg)
    assert toward.shape == (prof.fs,)
    assert np.mean(toward**2) > 1.5 * np.mean(away**2)


def test_loudest_frames_find_a_talker_with_pauses():
    """A source that is on for 30% of the block: on all frames room noise dilutes it; on the loudest
    frames its direction stands out more."""
    from earshot_arrays.doa import angle_diff, estimate_bearing

    prof = load_profile("profiles/uma16.yaml")
    rng = np.random.default_rng(1)
    pose = Pose(0.0, 0.0, 0.0)
    src_b = 250.0
    src = (2 * math.sin(math.radians(src_b)), 2 * math.cos(math.radians(src_b)))
    sig = drone_like(prof.fs, prof.fs, rng)
    sig[: int(0.7 * prof.fs)] = 0.0  # silent for the first 70%
    m = mics(array_block(src, pose, prof, sig, rng, snr_db=-6), prof)
    _, p_all = estimate_bearing(m, prof.fs, prof.mic_xy)
    b_loud, p_loud = estimate_bearing(m, prof.fs, prof.mic_xy, loudest=0.3)
    assert abs(angle_diff(b_loud, src_b)) < 10
    assert p_loud > p_all
