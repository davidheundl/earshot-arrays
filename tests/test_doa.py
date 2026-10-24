import math

import numpy as np
import pytest

from earshot_arrays.config import Pose, load_profile, mic_xy_from
from earshot_arrays.devices import mics
from earshot_arrays.doa import SCAN_BINS, angle_diff, estimate_bearing
from earshot_arrays.synth import array_block, drone_like

PROFILE = load_profile("profiles/respeaker_usb_v2.yaml")


def test_respeaker_profile_geometry():
    assert PROFILE.mic_xy.shape == (2, 4)
    assert PROFILE.mic_channels == (1, 2, 3, 4)
    ang = np.degrees(np.arctan2(PROFILE.mic_xy[1], PROFILE.mic_xy[0])) % 360
    np.testing.assert_allclose(ang, [45, 135, 225, 315])
    np.testing.assert_allclose(np.hypot(*PROFILE.mic_xy), 0.032)


def test_mics_list_is_transposed_to_2_by_n():
    xy = mic_xy_from({"mics": [[0.0, 0.0], [0.05, 0.0], [0.0, 0.05]]})
    assert xy.shape == (2, 3)
    np.testing.assert_allclose(xy[:, 1], [0.05, 0.0])


@pytest.mark.parametrize("rotation", [0.0, 90.0, 200.0])
@pytest.mark.parametrize("bearing", [0.0, 10.0, 90.0, 180.0, 270.0, 350.0])
def test_bearing_of_synthetic_source(bearing, rotation):
    rng = np.random.default_rng(int(bearing + rotation))
    pose = Pose(0.0, 0.0, rotation)
    src = (2.0 * math.sin(math.radians(bearing)), 2.0 * math.cos(math.radians(bearing)))
    block = array_block(src, pose, PROFILE, drone_like(PROFILE.fs, PROFILE.fs, rng), rng)
    est, power = estimate_bearing(mics(block, PROFILE), PROFILE.fs, PROFILE.mic_xy, rotation)
    assert abs(angle_diff(est, bearing)) < 10.0
    assert power > 1.3


def test_noise_has_no_clear_direction():
    rng = np.random.default_rng(3)
    block = rng.standard_normal((4, PROFILE.fs)).astype(np.float32)
    _, power = estimate_bearing(block, PROFILE.fs, PROFILE.mic_xy)
    assert power < 1.3


def test_scan_peaks_at_bearing():
    rng = np.random.default_rng(4)
    block = array_block((1.0, 0.0), Pose(0.0, 0.0, 0.0), PROFILE, drone_like(PROFILE.fs, PROFILE.fs, rng), rng)
    scan: list = []
    estimate_bearing(mics(block, PROFILE), PROFILE.fs, PROFILE.mic_xy, 0.0, scan=scan)
    assert len(scan) == 72
    assert abs(angle_diff(5.0 * int(np.argmax(scan)) + 2.5, 90.0)) <= 7.5


def test_angle_diff_wraps():
    assert angle_diff(350.0, 10.0) == pytest.approx(-20.0)
    assert angle_diff(10.0, 350.0) == pytest.approx(20.0)


def test_grid_geometry_rows_front_to_back_left_to_right():
    xy = mic_xy_from({"grid": {"rows": 2, "cols": 3, "spacing": 0.1}})
    np.testing.assert_allclose(xy.T, [[-0.1, 0.05], [0, 0.05], [0.1, 0.05],
                                      [-0.1, -0.05], [0, -0.05], [0.1, -0.05]], atol=1e-12)


@pytest.mark.parametrize("bearing", [0.0, 33.0, 200.0, 350.0])
def test_uma16_grid_bearing(bearing):
    prof = load_profile("profiles/uma16.yaml")
    assert prof.mic_xy.shape == (2, 16)
    rng = np.random.default_rng(int(bearing))
    pose = Pose(0.0, 0.0, 135.0)
    src = (math.sin(math.radians(bearing)), math.cos(math.radians(bearing)))
    block = array_block(src, pose, prof, drone_like(prof.fs, prof.fs, rng), rng)
    est, _ = estimate_bearing(mics(block, prof), prof.fs, prof.mic_xy, pose.rotation_deg)
    assert abs(angle_diff(est, bearing)) < 5.0


def test_silent_block_has_no_direction():
    scan: list = []
    _, power = estimate_bearing(np.zeros((4, 16000)), 16000, PROFILE.mic_xy, scan=scan)
    assert power == 0.0
    assert scan == [0.0] * SCAN_BINS
