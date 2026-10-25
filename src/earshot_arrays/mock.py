"""A mocked array for one-board demos: no audio, a bearing toward a virtual source. Pure numpy.

    zone = zone_from(MOCK_ZONE, office)                 # where a mocked crossing may lie
    src = zone_point(pose_a, bearing_a, zone, ref)      # on A's ray, inside the zone (or None)
    bearing, scan = mock_bearing(pose_b, src, rng)      # what the mock at pose_b "hears"

The mock is only as clear as the real array it follows, so it adds a crossing ray when something is
heard and stays silent otherwise. Used by `arrays twin` and `arrays live` (--mock, --mock-follow).
"""

from __future__ import annotations

import math

import numpy as np

from earshot_arrays.config import Pose
from earshot_arrays.doa import SCAN_BINS
from earshot_arrays.locate import bearing_to, direction

SIGMA_DEG = 3.0  # bearing noise: what a UMA-16 scatters
SCAN_WIDTH_DEG = 15.0  # width of the mock's scan peak (wedges only)
# The demo office: the gap between the arrays table and the table west of it, and a strip just behind
# (west of) that table, along the tables' length
MOCK_ZONE = "14.05,6.2,15.15,10.9;11.2,6.2,12.38,10.9"


def zone_from(spec: str, office=None):
    """(x, y) -> bool: inside one of the rectangles "x0,y0,x1,y1;..." and not on the office's furniture."""
    from earshot_arrays.belief import inside_polygon

    rects = [tuple(float(v) for v in r.split(",")) for r in spec.split(";") if r.strip()]
    furniture = [] if office is None else [list(ar.polygon) for ar in office.areas if ar.kind == "furniture"]

    def inside(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.zeros(x.shape, bool)
        for x0, y0, x1, y1 in rects:
            ok |= (x >= min(x0, x1)) & (x <= max(x0, x1)) & (y >= min(y0, y1)) & (y <= max(y0, y1))
        for poly in furniture:
            ok &= ~inside_polygon(x, y, poly)
        return ok

    return inside


def zone_point(pose: Pose, bearing: float, zone, ref: tuple[float, float] | None = None,
               max_m: float = 25.0) -> tuple[float, float] | None:
    """Middle of the stretch of the ray from `pose` along `bearing` that lies in the zone (of several,
    the one nearest `ref`, else the first), or None if the ray misses the zone."""
    dx, dy = direction(bearing)
    t = np.arange(0.3, max_m, 0.05)
    x, y = pose.x + t * dx, pose.y + t * dy
    inside = np.asarray(zone(x, y), dtype=bool)
    if not inside.any():
        return None
    edges = np.flatnonzero(np.diff(np.concatenate([[0], inside.astype(int), [0]])))
    mids = [((x[a] + x[e - 1]) / 2, (y[a] + y[e - 1]) / 2) for a, e in zip(edges[::2], edges[1::2])]
    if ref is None:
        return mids[0]
    return min(mids, key=lambda m: math.hypot(m[0] - ref[0], m[1] - ref[1]))


def mock_bearing(pose: Pose, src: tuple[float, float], rng: np.random.Generator,
                 sigma_deg: float = SIGMA_DEG) -> tuple[float, list[float]]:
    """(room bearing with noise, normalised scan peaked there) of a mock at `pose` hearing `src`."""
    b = (bearing_to(pose, *src) + rng.normal(0.0, sigma_deg)) % 360.0
    d = (360.0 / SCAN_BINS * (np.arange(SCAN_BINS) + 0.5) - b + 180.0) % 360.0 - 180.0
    return b, [float(v) for v in np.exp(-0.5 * (d / SCAN_WIDTH_DEG) ** 2)]
