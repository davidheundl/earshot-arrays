"""Probability grid over the room floor, fed by each array's bearing. Pure numpy.

    grid = Grid.from_bounds(0, 0, 12, 8, 0.25, floor=[(0, 0), (12, 0), (12, 8), (0, 8)])
    bearings = {name: bearing_grid(grid, pose) for name, pose in poses}   # once per array
    belief = Belief(grid)
    every second: belief.predict(); belief.update([peak_likelihood(bearings[n], b, power) ...])

The likelihood is a narrow bump around each array's measured bearing (SIGMA_DEG), evaluated at
every cell's exact bearing: the SRP peak is accurate to a few degrees although the beam itself is
tens of degrees wide, and arrays only a metre apart differ by less than a beam width.
`angular_likelihood` keeps the whole scan (5 deg bins) for comparison.

One array gives a fan along its bearing; two or more cross into a blob. Arrays are [iy, ix]:
row iy grows with y (north), column ix with x (east). Bearings as in `locate`: compass degrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from earshot_arrays.config import Pose
from earshot_arrays.doa import SCAN_BINS

MIN_POWER = 1.3  # SRP peak / mean below which a scan says nothing (as cli.MIN_POWER)
NEAR_M = 1.0  # radius for Estimate.mass_near
# Spread of a strong bearing around the true one: stationary sources in our UMA-16 recordings scatter
# 1-3 deg (median absolute deviation 1.0-2.5), with an occasional wild block.
SIGMA_DEG = 3.0
FLOOR = 0.05  # likelihood far from the bearing: one wild block dims a good track, it does not erase it


def inside_polygon(x: np.ndarray, y: np.ndarray, poly: list[tuple[float, float]]) -> np.ndarray:
    """Even-odd rule: True where (x, y) lies inside the polygon [(x, y), ...]."""
    inside = np.zeros(np.shape(x), dtype=bool)
    n = len(poly)
    for i in range(n):
        (xa, ya), (xb, yb) = poly[i], poly[(i + 1) % n]
        crosses = (ya > y) != (yb > y)  # the edge spans the horizontal line through the point
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = xa + (y - ya) * (xb - xa) / (yb - ya)
        inside ^= crosses & (x < x_at)
    return inside


@dataclass(frozen=True)
class Grid:
    x0: float  # room x of the west edge of column 0
    y0: float  # room y of the south edge of row 0
    cell: float  # metres
    nx: int
    ny: int
    mask: np.ndarray  # (ny, nx) bool, True = on the floor

    @classmethod
    def from_bounds(cls, xmin: float, ymin: float, xmax: float, ymax: float, cell: float,
                    floor: list[tuple[float, float]] | None = None) -> Grid:
        nx = max(1, math.ceil((xmax - xmin) / cell - 1e-9))
        ny = max(1, math.ceil((ymax - ymin) / cell - 1e-9))
        mask = np.ones((ny, nx), dtype=bool)
        g = cls(xmin, ymin, cell, nx, ny, mask)
        if floor is not None:
            mask = inside_polygon(*g.centers(), floor)
            g = cls(xmin, ymin, cell, nx, ny, mask)
        return g

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """(X, Y), each (ny, nx): room coordinates of the cell centres."""
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.cell
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.cell
        return np.meshgrid(xs, ys)

    def cell_of(self, x: float, y: float) -> tuple[int, int] | None:
        """(ix, iy) of the cell holding (x, y), or None outside the grid."""
        ix = math.floor((x - self.x0) / self.cell)
        iy = math.floor((y - self.y0) / self.cell)
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return ix, iy
        return None


@dataclass(frozen=True)
class Estimate:
    x: float  # centre of the most likely cell
    y: float
    p_max: float  # its probability
    r50: float  # metres around (x, y) holding half the mass
    mass_near: float  # mass within NEAR_M of (x, y)


def angular_likelihood(scan: list | np.ndarray | None, power: float, min_power: float = MIN_POWER,
                       sharpness: float = 8.0, floor: float = 0.02) -> np.ndarray:
    """(SCAN_BINS,) likelihood over room bearings from one array's scan; all ones if it says nothing."""
    if scan is None or len(scan) != SCAN_BINS or power < min_power:
        return np.ones(SCAN_BINS)
    s = np.clip(np.asarray(scan, dtype=float), 0.0, 1.0)
    return floor + s**sharpness  # the peak dominates, a second loud direction still shows


def bearing_grid(grid: Grid, pose: Pose) -> np.ndarray:
    """(ny, nx): room bearing in degrees from the array to each cell centre."""
    X, Y = grid.centers()
    return np.degrees(np.arctan2(X - pose.x, Y - pose.y)) % 360.0  # as locate.bearing_to


def bearing_bins(grid: Grid, pose: Pose) -> np.ndarray:
    """(ny, nx) int: the scan bin of the room bearing from the array to each cell centre."""
    return (bearing_grid(grid, pose) // (360.0 / SCAN_BINS)).astype(int) % SCAN_BINS


def peak_likelihood(bearings: np.ndarray, peak_deg: float, power: float, min_power: float = MIN_POWER,
                    sigma_deg: float = SIGMA_DEG, floor: float = FLOOR) -> np.ndarray:
    """(ny, nx) likelihood of each cell given one array's measured bearing; all ones if it says nothing."""
    if power < min_power:
        return np.ones_like(bearings)
    d = (bearings - peak_deg + 180.0) % 360.0 - 180.0
    return floor + (1.0 - floor) * np.exp(-0.5 * (d / sigma_deg) ** 2)


def gaussian_kernel(sigma_cells: float) -> np.ndarray:
    r = max(1, math.ceil(3.0 * sigma_cells))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / max(sigma_cells, 1e-9)) ** 2)
    return k / k.sum()


def _convolve_same(v: np.ndarray, k: np.ndarray) -> np.ndarray:
    r = (len(k) - 1) // 2  # centre slice of "full": len(v) even when the kernel is longer (mode="same" is not)
    return np.convolve(v, k, mode="full")[r:r + len(v)]


def blur(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Separable 2D convolution, same size, zero outside."""
    a = np.apply_along_axis(_convolve_same, 0, a, k)
    return np.apply_along_axis(_convolve_same, 1, a, k)


class Belief:
    """Where the source probably is: p over the grid, zero off the floor, sums to 1."""

    def __init__(self, grid: Grid, motion_sigma_m: float = 0.6, uniform_mix: float = 0.03):
        self.grid = grid
        self.uniform_mix = uniform_mix
        self.kernel = gaussian_kernel(motion_sigma_m / grid.cell)
        self.uniform = grid.mask / max(grid.mask.sum(), 1)
        # Mass a cell's kernel puts on the floor; the rest (a step off the floor) stays in the cell.
        self.reach = blur(grid.mask.astype(float), self.kernel)
        self.p = self.uniform.copy()

    def predict(self) -> None:
        """One second of motion: Gaussian spread, then a little of the uniform floor distribution."""
        # symmetric kernel, so walls do not push mass inward (uniform stays uniform)
        p = (blur(self.p, self.kernel) + self.p * (1.0 - self.reach)) * self.grid.mask
        p = (1.0 - self.uniform_mix) * p / max(p.sum(), 1e-300) + self.uniform_mix * self.uniform
        self.p = p / p.sum()

    def update(self, likelihoods: list[np.ndarray]) -> None:
        """Multiply in each array's (ny, nx) likelihood (peak_likelihood, or angular_likelihood[bins])."""
        p = self.p.copy()
        for like in likelihoods:
            p *= like
        p *= self.grid.mask
        total = p.sum()
        if not np.isfinite(total) or total < 1e-300:
            return  # the observations contradict everything: keep what we had
        self.p = p / total

    def estimate(self) -> Estimate:
        iy, ix = np.unravel_index(int(np.argmax(self.p)), self.p.shape)
        X, Y = self.grid.centers()
        x, y = float(X[iy, ix]), float(Y[iy, ix])
        d = np.hypot(X - x, Y - y).ravel()
        order = np.argsort(d, kind="stable")
        cum = np.cumsum(self.p.ravel()[order])
        r50 = float(d[order][min(int(np.searchsorted(cum, 0.5)), d.size - 1)])
        near = float(self.p.ravel()[d <= NEAR_M].sum())
        return Estimate(x, y, float(self.p[iy, ix]), r50, near)
