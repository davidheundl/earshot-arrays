"""Array profiles (`profiles/*.yaml`) and the table setup (`setup.yaml`). See the YAML files for the schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Profile:
    name: str
    n_channels: int
    fs: int
    mic_channels: tuple[int, ...]
    mic_xy: np.ndarray  # (2, n_mics), metres, array frame (x right, y forward)


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    rotation_deg: float  # room bearing the array's forward axis points to


@dataclass(frozen=True)
class ArraySetup:
    name: str
    profile: Profile
    device: int | str | None
    pose: Pose


@dataclass(frozen=True)
class Setup:
    table: tuple[float, float]  # width (x), depth (y) in metres
    arrays: tuple[ArraySetup, ...]
    path: Path | None = None


def mic_xy_from(d: dict) -> np.ndarray:
    """(2, n_mics) from a profile's `mics` list or `circle` block."""
    if "mics" in d:
        xy = np.asarray(d["mics"], dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("profile `mics` must be a list of [x, y] pairs")
        return xy.T.copy()
    if "circle" in d:
        c = d["circle"]
        n = int(c["n"])
        step = float(c.get("step_deg", 360.0 / n))
        ang = np.radians(float(c.get("start_deg", 0.0)) + step * np.arange(n))
        r = float(c["radius"])
        return np.stack([r * np.cos(ang), r * np.sin(ang)])
    raise ValueError("profile needs `mics` or `circle`")


def load_profile(path: str | Path) -> Profile:
    d = yaml.safe_load(Path(path).read_text())
    mic_xy = mic_xy_from(d)
    chans = tuple(int(c) for c in d["mic_channels"])
    n_ch = int(d["n_channels"])
    if len(chans) != mic_xy.shape[1]:
        raise ValueError(f"{path}: {len(chans)} mic_channels but {mic_xy.shape[1]} mic positions")
    if any(not 0 <= c < n_ch for c in chans):
        raise ValueError(f"{path}: mic_channels must be 0-based and below n_channels={n_ch}")
    return Profile(str(d.get("name", Path(path).stem)), n_ch, int(d["fs"]), chans, mic_xy)


def load_setup(path: str | Path) -> Setup:
    path = Path(path)
    d = yaml.safe_load(path.read_text())
    arrays = []
    for a in d["arrays"]:
        prof = Path(a["profile"])
        if not prof.is_absolute() and (path.parent / prof).exists():
            prof = path.parent / prof  # relative to the setup file first (e.g. a copy inside a recording)
        x, y, rot = (float(v) for v in a["pose"])
        arrays.append(ArraySetup(str(a["name"]), load_profile(prof), a.get("device"), Pose(x, y, rot)))
    t = d.get("table", {})
    return Setup((float(t.get("width", 1.6)), float(t.get("depth", 0.8))), tuple(arrays), path)


def setup_dict(s: Setup, profile_dir: str = "profiles") -> dict:
    """Plain dict of a setup, with profiles written next to it (used by `record` to copy the setup)."""
    return {
        "table": {"width": s.table[0], "depth": s.table[1]},
        "arrays": [
            {"name": a.name, "profile": f"{profile_dir}/{a.name}.yaml", "device": a.device,
             "pose": [a.pose.x, a.pose.y, a.pose.rotation_deg]}
            for a in s.arrays
        ],
    }


def profile_dict(p: Profile) -> dict:
    return {"name": p.name, "n_channels": p.n_channels, "fs": p.fs, "mic_channels": list(p.mic_channels),
            "mics": [[round(float(x), 6), round(float(y), 6)] for x, y in p.mic_xy.T]}


def write_setup(s: Setup, out_dir: Path) -> Path:
    """Write the setup and its profiles into out_dir, so the directory replays on its own."""
    (out_dir / "profiles").mkdir(parents=True, exist_ok=True)
    for a in s.arrays:
        (out_dir / "profiles" / f"{a.name}.yaml").write_text(yaml.safe_dump(profile_dict(a.profile)))
    p = out_dir / "setup.yaml"
    p.write_text(yaml.safe_dump(setup_dict(s), sort_keys=False))
    return p
