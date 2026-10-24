"""Two array poses + two bearings → source position. Pure.

Rays start at each array's (x, y) and point along the room bearing b: direction (sin b, cos b).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from earshot_arrays.config import Pose

MIN_CROSS_DEG = 10.0  # rays closer to parallel than this give no usable fix
BEARING_SIGMA_DEG = 5.0  # assumed bearing error per array, for the uncertainty


@dataclass(frozen=True)
class Fix:
    x: float
    y: float
    sigma: float  # metres, 1-sigma radius from the bearing error
    crossing_deg: float  # angle between the rays


def direction(bearing_deg: float) -> tuple[float, float]:
    b = math.radians(bearing_deg)
    return math.sin(b), math.cos(b)


def bearing_to(pose: Pose, x: float, y: float) -> float:
    """Room bearing from the array at `pose` to the point (x, y)."""
    return math.degrees(math.atan2(x - pose.x, y - pose.y)) % 360.0


def intersect(pa: Pose, ba: float, pb: Pose, bb: float, sigma_deg: float = BEARING_SIGMA_DEG,
              min_cross_deg: float = MIN_CROSS_DEG) -> Fix | None:
    """Crossing of the two bearing rays, or None if they are near-parallel or cross behind an array."""
    da, db = direction(ba), direction(bb)
    cross = da[0] * db[1] - da[1] * db[0]
    if abs(cross) < math.sin(math.radians(min_cross_deg)):
        return None
    # pa + ta*da = pb + tb*db  →  solve for ta, tb by Cramer's rule
    rx, ry = pb.x - pa.x, pb.y - pa.y
    ta = (rx * db[1] - ry * db[0]) / cross
    tb = (rx * da[1] - ry * da[0]) / cross
    if ta <= 0 or tb <= 0:
        return None
    # A bearing error δ on one ray moves the crossing along the other ray by range·δ / sin(θ).
    s = math.radians(sigma_deg) / abs(cross)
    sigma = math.hypot(ta * s, tb * s)
    crossing = math.degrees(math.asin(min(1.0, abs(cross))))
    return Fix(pa.x + ta * da[0], pa.y + ta * da[1], sigma, crossing)
