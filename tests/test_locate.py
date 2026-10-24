import math

import numpy as np
import pytest

from earshot_arrays.config import Pose, load_profile
from earshot_arrays.devices import mics
from earshot_arrays.doa import estimate_bearing
from earshot_arrays.locate import bearing_to, intersect
from earshot_arrays.synth import array_block, drone_like

A = Pose(0.1, 0.1, 0.0)
B = Pose(1.5, 0.1, 0.0)


def test_bearing_to_compass():
    o = Pose(0.0, 0.0, 0.0)
    assert bearing_to(o, 0.0, 1.0) == pytest.approx(0.0)
    assert bearing_to(o, 1.0, 0.0) == pytest.approx(90.0)
    assert bearing_to(o, 0.0, -1.0) == pytest.approx(180.0)
    assert bearing_to(o, -1.0, 0.0) == pytest.approx(270.0)


@pytest.mark.parametrize("x, y", [(0.8, 0.6), (0.3, 0.7), (1.4, 0.5), (0.8, 2.0)])
def test_round_trip(x, y):
    fix = intersect(A, bearing_to(A, x, y), B, bearing_to(B, x, y))
    assert fix is not None
    assert (fix.x, fix.y) == pytest.approx((x, y), abs=1e-9)
    assert fix.sigma > 0


def test_near_parallel_rays_give_none():
    # source almost on the baseline extension: rays nearly parallel
    assert intersect(A, bearing_to(A, 5.0, 0.12), B, bearing_to(B, 5.0, 0.12)) is None
    assert intersect(A, 0.0, B, 0.0) is None


def test_crossing_behind_an_array_gives_none():
    # rays point away from each other; the lines cross south of the arrays, behind both
    assert intersect(A, 45.0, B, 315.0) is not None
    assert intersect(A, 315.0, B, 45.0) is None


def test_uncertainty_grows_with_range():
    near = intersect(A, bearing_to(A, 0.8, 0.5), B, bearing_to(B, 0.8, 0.5))
    far = intersect(A, bearing_to(A, 0.8, 3.0), B, bearing_to(B, 0.8, 3.0))
    assert far.sigma > near.sigma


def test_synthetic_blocks_through_doa_and_locate():
    prof = load_profile("profiles/respeaker_usb_v2.yaml")
    rng = np.random.default_rng(7)
    rot_b = 270.0  # array B turned to face west: the pose must be applied
    pb = Pose(B.x, B.y, rot_b)
    for src in [(0.8, 0.6), (0.4, 0.5), (1.2, 0.4)]:
        bearings = []
        for pose in (A, pb):
            block = array_block(src, pose, prof, drone_like(prof.fs, prof.fs, rng), rng)
            bearings.append(estimate_bearing(mics(block, prof), prof.fs, prof.mic_xy, pose.rotation_deg)[0])
        fix = intersect(A, bearings[0], pb, bearings[1])
        assert fix is not None
        assert math.hypot(fix.x - src[0], fix.y - src[1]) < 0.3
