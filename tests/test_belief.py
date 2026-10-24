import math

import numpy as np
import pytest

from earshot_arrays.belief import (
    Belief,
    Grid,
    angular_likelihood,
    bearing_bins,
    bearing_grid,
    peak_likelihood,
)
from earshot_arrays.config import Pose, load_profile
from earshot_arrays.devices import mics
from earshot_arrays.doa import SCAN_BINS, estimate_bearing
from earshot_arrays.locate import bearing_to
from earshot_arrays.synth import array_block, drone_like

UMA16 = load_profile("profiles/uma16.yaml")


def peaked_scan(bearing_deg: float, width_deg: float = 10.0) -> list[float]:
    """Synthetic normalised SRP scan peaking at bearing_deg (bin centres every 5 deg)."""
    centres = 5.0 * np.arange(SCAN_BINS) + 2.5
    d = (centres - bearing_deg + 180.0) % 360.0 - 180.0
    return list(np.exp(-0.5 * (d / width_deg) ** 2))


def observe(grid: Grid, pose: Pose, x: float, y: float) -> np.ndarray:
    return angular_likelihood(peaked_scan(bearing_to(pose, x, y)), 2.0)[bearing_bins(grid, pose)]


def test_centers_and_cell_of_agree():
    g = Grid.from_bounds(-1.0, 2.0, 4.0, 5.0, 0.5)
    assert (g.nx, g.ny) == (10, 6)
    X, Y = g.centers()
    assert X.shape == Y.shape == (6, 10)
    for iy in range(g.ny):
        for ix in range(g.nx):
            assert g.cell_of(X[iy, ix], Y[iy, ix]) == (ix, iy)
    assert X[0, 1] > X[0, 0] and Y[1, 0] > Y[0, 0]  # columns go east, rows go north
    assert g.cell_of(-1.01, 3.0) is None
    assert g.cell_of(0.0, 5.01) is None


def test_floor_polygon_masks_l_shaped_room():
    # L: 4 x 4 square with the north-east 2 x 2 quarter cut out
    floor = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
    g = Grid.from_bounds(0.0, 0.0, 4.0, 4.0, 1.0, floor=floor)
    expected = np.array([
        [1, 1, 1, 1],  # iy = 0 (south)
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0],  # iy = 3 (north)
    ], dtype=bool)
    np.testing.assert_array_equal(g.mask, expected)
    b = Belief(g)
    assert b.p[~g.mask].sum() == 0.0
    assert b.p.sum() == pytest.approx(1.0)


def test_orientation_north_east_corner():
    g = Grid.from_bounds(0.0, 0.0, 8.0, 6.0, 0.25)
    a, b = Pose(0.0, 0.0, 0.0), Pose(8.0, 0.0, 90.0)
    src = (7.0, 5.0)  # north-east
    bel = Belief(g)
    bel.update([observe(g, a, *src), observe(g, b, *src)])
    iy, ix = np.unravel_index(int(np.argmax(bel.p)), bel.p.shape)
    assert iy > g.ny // 2 and ix > g.nx // 2  # the peak index itself sits north-east
    est = bel.estimate()
    assert abs(est.x - src[0]) <= g.cell + 1e-9
    assert abs(est.y - src[1]) <= g.cell + 1e-9


def test_flat_likelihood_leaves_p_unchanged():
    g = Grid.from_bounds(0.0, 0.0, 5.0, 4.0, 0.25)
    bel = Belief(g)
    bel.update([observe(g, Pose(0.0, 0.0, 0.0), 3.0, 3.0)])
    before = bel.p.copy()
    flat = angular_likelihood(peaked_scan(45.0), power=1.1)  # below MIN_POWER
    np.testing.assert_array_equal(flat, np.ones(SCAN_BINS))
    bel.update([flat[bearing_bins(g, Pose(5.0, 0.0, 0.0))],
                angular_likelihood(None, 3.0)[bearing_bins(g, Pose(0.0, 0.0, 0.0))]])
    np.testing.assert_allclose(bel.p, before, rtol=1e-12, atol=1e-15)


def test_normalised_after_predict_and_update():
    floor = [(0, 0), (6, 0), (6, 3), (3, 3), (3, 5), (0, 5)]
    g = Grid.from_bounds(0.0, 0.0, 6.0, 5.0, 0.25, floor=floor)
    bel = Belief(g)
    pose = Pose(0.5, 0.5, 30.0)
    for _ in range(5):
        bel.predict()
        assert bel.p.sum() == pytest.approx(1.0)
        bel.update([observe(g, pose, 2.0, 4.0)])
        assert bel.p.sum() == pytest.approx(1.0)
        assert bel.p[~g.mask].sum() == 0.0
        assert np.all(bel.p >= 0.0)


def test_contradiction_keeps_previous_p():
    g = Grid.from_bounds(0.0, 0.0, 4.0, 4.0, 0.5)
    bel = Belief(g)
    before = bel.p.copy()
    bel.update([np.zeros(SCAN_BINS)[bearing_bins(g, Pose(0.0, 0.0, 0.0))]])
    np.testing.assert_array_equal(bel.p, before)


def test_one_array_gives_a_fan_along_the_bearing():
    g = Grid.from_bounds(0.0, 0.0, 8.0, 8.0, 0.25)
    pose = Pose(4.0, 4.0, 0.0)
    bel = Belief(g)
    bel.update([observe(g, pose, 7.0, 7.0)])  # bearing 45
    on, off = [], []
    for r in (1.0, 2.0, 3.0):
        for bearing, out in ((45.0, on), (135.0, off), (315.0, off)):
            x = pose.x + r * math.sin(math.radians(bearing))
            y = pose.y + r * math.cos(math.radians(bearing))
            c = g.cell_of(x, y)
            if c is not None:
                out.append(bel.p[c[1], c[0]])
    assert len(on) == 3 and len(off) == 6
    assert min(on) > 10 * max(off)


def test_predict_spreads_a_delta_and_keeps_mass():
    g = Grid.from_bounds(0.0, 0.0, 5.0, 5.0, 0.25)
    bel = Belief(g, motion_sigma_m=0.6, uniform_mix=0.0)
    bel.p = np.zeros_like(bel.p)
    bel.p[0, 0] = 1.0  # a corner: mass must not leak off the grid
    bel.predict()
    assert bel.p.sum() == pytest.approx(1.0)
    assert bel.p[0, 0] < 1.0
    assert bel.p[0, 1] > 0.0 and bel.p[1, 0] > 0.0
    assert bel.p[0, 1] == pytest.approx(bel.p[1, 0])
    assert int(np.argmax(bel.p)) == 0


def test_predict_on_a_grid_narrower_than_the_kernel():
    g = Grid.from_bounds(0.0, 0.0, 1.6, 0.8, 0.1)  # 16 x 8 cells, kernel 37 cells
    bel = Belief(g)
    assert bel.reach.shape == g.mask.shape
    bel.predict()
    assert bel.p.shape == g.mask.shape
    assert bel.p.sum() == pytest.approx(1.0)


def test_predict_keeps_uniform_uniform():
    # walls must not push mass inward: a source in a wall or corner cell stays estimated there
    floor = [(0, 0), (6, 0), (6, 3), (3, 3), (3, 5), (0, 5)]
    g = Grid.from_bounds(0.0, 0.0, 6.0, 5.0, 0.5, floor=floor)
    bel = Belief(g, uniform_mix=0.0)
    for _ in range(50):
        bel.predict()
    np.testing.assert_allclose(bel.p[g.mask], bel.uniform[g.mask], rtol=1e-9)


def test_estimate_r50_and_mass_near():
    g = Grid.from_bounds(0.0, 0.0, 4.0, 4.0, 0.5)
    bel = Belief(g)
    bel.p = np.zeros_like(bel.p)
    bel.p[2, 3] = 0.6
    bel.p[2, 4] = 0.4
    est = bel.estimate()
    assert (est.x, est.y) == pytest.approx((1.75, 1.25))
    assert est.p_max == pytest.approx(0.6)
    assert est.r50 == pytest.approx(0.0)
    assert est.mass_near == pytest.approx(1.0)


def test_end_to_end_two_uma16_arrays():
    rng = np.random.default_rng(7)
    g = Grid.from_bounds(-1.0, -1.0, 7.0, 6.0, 0.25)
    poses = [Pose(0.0, 0.0, 0.0), Pose(6.0, 0.0, 270.0)]
    src = (2.0, 3.5)  # ~4 m from the first array, ~5.3 m from the second
    n = UMA16.fs // 2
    sig = drone_like(n, UMA16.fs, rng)
    bel = Belief(g)
    obs = []
    for pose in poses:
        block = array_block(src, pose, UMA16, sig, rng)
        scan: list = []
        _, power = estimate_bearing(mics(block, UMA16), UMA16.fs, UMA16.mic_xy, pose.rotation_deg, scan=scan)
        assert power > 1.3
        obs.append(angular_likelihood(scan, power)[bearing_bins(g, pose)])
    bel.update(obs)
    est = bel.estimate()
    assert math.hypot(est.x - src[0], est.y - src[1]) < 1.0


def test_peak_likelihood_is_flat_for_a_weak_bearing():
    g = Grid.from_bounds(0.0, 0.0, 4.0, 4.0, 0.5)
    bg = bearing_grid(g, Pose(0.0, 0.0, 0.0))
    np.testing.assert_array_equal(peak_likelihood(bg, 45.0, power=1.1), np.ones_like(bg))
    like = peak_likelihood(bg, 45.0, power=2.0)
    iy, ix = g.cell_of(3.0, 3.0)[::-1]
    assert like[iy, ix] == pytest.approx(1.0, abs=0.02)  # on the bearing
    assert like.min() == pytest.approx(0.05, abs=0.01)  # far off it: the floor


def test_arrays_a_metre_apart_still_cross():
    # the office's red marks: two arrays 1.1 m apart, a source 4 m west of them; bearings differ by ~15 deg,
    # less than the SRP beam is wide, so this needs the narrow per-cell likelihood, not the 5 deg scan bins
    g = Grid.from_bounds(0.0, 0.0, 18.0, 18.0, 0.5)
    poses = [Pose(16.03, 8.70, 0.0), Pose(16.07, 7.56, 0.0)]
    src = (12.0, 9.0)
    bel = Belief(g)
    bgs = [bearing_grid(g, pose) for pose in poses]
    for _ in range(5):
        bel.predict()
        bel.update([peak_likelihood(bg, bearing_to(pose, *src), 2.0) for bg, pose in zip(bgs, poses)])
    est = bel.estimate()
    assert math.hypot(est.x - src[0], est.y - src[1]) < 1.0, est
