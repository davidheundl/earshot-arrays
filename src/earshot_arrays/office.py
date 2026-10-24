"""The office the live demo runs in (`office.yaml`): floor, walls, furniture, where the arrays stand.

Room frame as everywhere: metres, x east, y north (up on the map). See office.example.yaml for the schema.
An optional floor-plan bitmap is north-up and placed by metres per pixel and the pixel at room (0, 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from earshot_arrays.config import Pose

AREA_KINDS = ("room", "furniture", "glass")
REF_SCALE_TOL = 0.05  # image_ref: x and y scale may disagree by this much


@dataclass(frozen=True)
class Area:
    name: str
    polygon: tuple[tuple[float, float], ...]
    kind: str = "room"  # room | furniture | glass


@dataclass(frozen=True)
class Label:
    text: str
    at: tuple[float, float]


@dataclass(frozen=True)
class Office:
    name: str
    bounds: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax in metres
    cell: float = 0.5  # grid cell size, metres
    floor: tuple[tuple[float, float], ...] | None = None  # polygon; cells outside are not the floor
    walls: tuple[tuple[tuple[float, float], ...], ...] = ()  # polylines
    areas: tuple[Area, ...] = ()
    labels: tuple[Label, ...] = ()
    image: Path | None = None  # floor-plan bitmap (PNG/JPG), north-up
    image_scale: float | None = None  # metres per pixel
    image_origin_px: tuple[float, float] | None = None  # pixel (u right, v down) at room (0, 0)
    arrays: dict[str, tuple[float, float]] = field(default_factory=dict)  # marks (x, y): replace setup x, y by name

    def image_extent(self, width_px: int, height_px: int) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) of the image in room metres, for imshow(origin='upper')."""
        if self.image_scale is None or self.image_origin_px is None:
            raise ValueError(f"office {self.name!r} has no image_scale / image_origin_px")
        s, (u, v) = self.image_scale, self.image_origin_px
        return (-u * s, (width_px - u) * s, -(height_px - v) * s, v * s)

    def grid(self):
        """The probability grid over the bounds, cells off the floor masked out."""
        from earshot_arrays.belief import Grid

        return Grid.from_bounds(*self.bounds, self.cell, self.floor)

    def poses(self, setup_arrays) -> list[tuple[str, Pose]]:
        """(name, pose) per array of a Setup: x, y from this office's marks by name, rotation from the setup."""
        return [(a.name, mark_pose(a.pose, self.arrays.get(a.name))) for a in setup_arrays]


def mark_pose(pose: Pose, mark: tuple[float, float] | None) -> Pose:
    """The pose moved onto an office mark; rotation_deg stays the one calibrated in setup.yaml (`arrays bearing`)."""
    return pose if mark is None else Pose(mark[0], mark[1], pose.rotation_deg)


def _pts(v) -> tuple[tuple[float, float], ...]:
    if any(len(p) != 2 for p in v):
        raise ValueError(f"expected a list of [x, y] pairs, got {v!r}")
    return tuple((float(p[0]), float(p[1])) for p in v)


def solve_image_ref(ref) -> tuple[float, tuple[float, float]]:
    """(metres per pixel, origin pixel) from two {px: [u, v], m: [x, y]} pairs, north-up, no rotation."""
    if len(ref) != 2:
        raise ValueError("image_ref needs exactly two {px, m} pairs")
    (u1, v1), (u2, v2) = (tuple(float(c) for c in r["px"]) for r in ref)
    (x1, y1), (x2, y2) = (tuple(float(c) for c in r["m"]) for r in ref)
    scales = []
    if u1 != u2:
        scales.append((x2 - x1) / (u2 - u1))
    if v1 != v2:
        scales.append(-(y2 - y1) / (v2 - v1))  # v runs down, y runs up
    if not scales:
        raise ValueError("image_ref points share the same pixel")
    if any(sc <= 0 for sc in scales):
        raise ValueError("image_ref implies a mirrored or rotated image; the plan must be north-up")
    s = sum(scales) / len(scales)
    if len(scales) == 2 and abs(scales[0] - scales[1]) / s > REF_SCALE_TOL:
        raise ValueError(f"image_ref x scale {scales[0]:.4g} and y scale {scales[1]:.4g} m/px disagree by "
                         f"more than {REF_SCALE_TOL:.0%}")
    return s, (u1 - x1 / s, v1 + y1 / s)


def load_office(path: str | Path) -> Office:
    path = Path(path)
    d = yaml.safe_load(path.read_text())
    b = tuple(float(v) for v in d["bounds"])
    if len(b) != 4 or b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError(f"{path}: bounds must be [xmin, ymin, xmax, ymax]")
    areas = []
    for a in d.get("areas") or []:
        kind = str(a.get("kind", "room"))
        if kind not in AREA_KINDS:
            raise ValueError(f"{path}: area kind {kind!r} not in {AREA_KINDS}")
        areas.append(Area(str(a.get("name", "")), _pts(a["polygon"]), kind))
    labels = tuple(Label(str(lb["text"]), (float(lb["at"][0]), float(lb["at"][1]))) for lb in d.get("labels") or [])
    image = scale = origin = None
    if d.get("image"):
        image = Path(d["image"])
        if not image.is_absolute():
            image = path.parent / image  # relative to the office file
        if "image_ref" in d:
            scale, origin = solve_image_ref(d["image_ref"])
        else:
            scale = float(d["image_scale"])
            origin = (float(d["image_origin_px"][0]), float(d["image_origin_px"][1]))
    arrays = {}
    for k, v in (d.get("arrays") or {}).items():  # [x, y]; a third value (old form) is ignored, see mark_pose
        if len(v) not in (2, 3):
            raise ValueError(f"{path}: array {k}: expected [x, y], got {v!r}")
        arrays[str(k)] = (float(v[0]), float(v[1]))
    return Office(
        name=str(d.get("name", path.stem)),
        bounds=b,
        cell=float(d.get("cell", 0.5)),
        floor=_pts(d["floor"]) if d.get("floor") else None,
        walls=tuple(_pts(w) for w in d.get("walls") or []),
        areas=tuple(areas),
        labels=labels,
        image=image,
        image_scale=scale,
        image_origin_px=origin,
        arrays=arrays,
    )
