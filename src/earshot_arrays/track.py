"""What the live map shows, second by second. Pure: no audio, no UI.

    tracker = Tracker({"A": pose_a, "B": pose_b})
    state = tracker.update(t, {"A": Obs(bearing, power, scan, level_db), ...}, probs)

The "clear direction" gate: a second counts only if some array's SRP power is >= GATE (a sound
stands out from one direction). Class probabilities are smoothed over gated seconds (each new
second weighs ALPHA, so a single interrupting second does not flip the label), and the label is
dropped ("–") after HOLD_S seconds without a gated second. A new label needs a smoothed probability
of MIN_PROB (below: "–", the model is unsure, e.g. a fan it never heard); a label already shown
stays while it is still the top class and above KEEP_PROB, so one interrupting second neither
flips nor blanks it. With two gated arrays
the bearings are crossed into a fix (locate.intersect). The trail keeps the last TRAIL gated seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from earshot_arrays.config import Pose
from earshot_arrays.locate import Fix, intersect

GATE = 1.3  # SRP peak / mean: a direction stands out (as cli.MIN_POWER)
ALPHA = 0.4  # weight of the newest gated second in the smoothed class probabilities (~2-3 s memory)
HOLD_S = 3.0  # seconds without a gated second before the label is dropped
MIN_PROB = 0.6  # smoothed top class below this: shown as "–" (unsure)
KEEP_PROB = 0.4  # ... unless it is the label already shown and stays above this
TRAIL = 10
NONE = "–"
# the room head's own names, as shown on the map
DISPLAY = {"speech": "person"}


@dataclass(frozen=True)
class Obs:
    bearing: float  # room compass degrees
    power: float  # SRP peak / mean
    scan: tuple[float, ...] = ()  # normalised SRP over SCAN_BINS room bearings from north
    level_db: float = -120.0
    gate: float = GATE

    @property
    def clear(self) -> bool:
        return self.power >= self.gate


@dataclass(frozen=True)
class TrailPoint:
    t: float
    label: str
    bearings: dict[str, float]  # array name -> bearing, only the arrays that were clear
    fix: tuple[float, float] | None


@dataclass
class State:
    t: float
    obs: dict[str, Obs]
    label: str  # smoothed class, or NONE
    prob: float
    why: str  # why the label is NONE: "no clear direction" / "unsure"; "" when there is a label
    probs: dict[str, float]  # smoothed
    raw_label: str  # this second's own top class (before smoothing and gate)
    clear: bool
    fix: Fix | None
    trail: list[TrailPoint] = field(default_factory=list)

    def to_json(self) -> dict:
        r = lambda v: round(float(v), 3)
        return {
            "t": r(self.t), "label": self.label, "prob": r(self.prob), "why": self.why, "raw_label": self.raw_label,
            "clear": self.clear, "probs": {k: r(v) for k, v in self.probs.items()},
            "arrays": {n: {"bearing": r(o.bearing), "power": r(o.power), "clear": o.clear,
                           "level_db": r(o.level_db), "scan": [r(v) for v in o.scan]}
                       for n, o in self.obs.items()},
            "fix": None if self.fix is None else {"x": r(self.fix.x), "y": r(self.fix.y), "sigma": r(self.fix.sigma)},
            "trail": [{"t": r(p.t), "label": p.label, "bearings": {n: r(b) for n, b in p.bearings.items()},
                       "fix": None if p.fix is None else [r(p.fix[0]), r(p.fix[1])]} for p in self.trail],
        }


def display_probs(probs: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in probs.items():
        out[DISPLAY.get(k, k)] = out.get(DISPLAY.get(k, k), 0.0) + float(v)
    return out


class Tracker:
    def __init__(self, poses: dict[str, Pose], alpha: float = ALPHA, hold_s: float = HOLD_S, trail: int = TRAIL,
                 min_prob: float = MIN_PROB):
        self.poses, self.alpha, self.hold_s, self.n_trail, self.min_prob = poses, alpha, hold_s, trail, min_prob
        self.smoothed: dict[str, float] = {}
        self.last_clear_t: float | None = None
        self.trail: list[TrailPoint] = []
        self.shown = NONE

    def update(self, t: float, obs: dict[str, Obs], probs: dict[str, float]) -> State:
        probs = display_probs(probs)
        raw = max(probs, key=probs.get) if probs else NONE
        clear = [n for n, o in obs.items() if o.clear]
        fix = None
        if len(clear) >= 2:
            a, b = clear[:2]
            fix = intersect(self.poses[a], obs[a].bearing, self.poses[b], obs[b].bearing)
        if clear:
            self.last_clear_t = t
            keys = sorted(set(self.smoothed) | set(probs))
            if not self.smoothed:
                self.smoothed = dict(probs)
            else:
                self.smoothed = {k: (1 - self.alpha) * self.smoothed.get(k, 0.0) + self.alpha * probs.get(k, 0.0)
                                 for k in keys}
        elif self.last_clear_t is None or t - self.last_clear_t > self.hold_s:
            self.smoothed = {}
        label, prob, why = NONE, 0.0, "no clear direction"
        if self.smoothed:
            top = max(self.smoothed, key=self.smoothed.get)
            prob = self.smoothed[top]
            keep = top == self.shown and prob >= KEEP_PROB
            label, why = (top, "") if prob >= self.min_prob or keep else (NONE, "unsure")
        self.shown = label
        if clear:
            self.trail.append(TrailPoint(t, label, {n: obs[n].bearing for n in clear},
                                         None if fix is None else (fix.x, fix.y)))
            self.trail = self.trail[-self.n_trail:]
        return State(t, obs, label, prob, why, dict(self.smoothed), raw, bool(clear), fix, list(self.trail))
