import pytest

from earshot_arrays.config import Pose
from earshot_arrays.locate import bearing_to
from earshot_arrays.track import NONE, Obs, Tracker

DRONE = {"drone": 0.9, "speech": 0.1}
SPEECH = {"drone": 0.1, "speech": 0.9}


def test_gate_hides_the_class_without_a_clear_direction():
    tr = Tracker({"A": Pose(0, 0, 0)})
    s = tr.update(0, {"A": Obs(10.0, 1.1)}, DRONE)
    assert s.label == NONE and not s.clear and s.trail == []
    s = tr.update(1, {"A": Obs(10.0, 1.6)}, DRONE)
    assert s.label == "drone" and s.clear


def test_one_interrupting_second_does_not_flip_the_label():
    tr = Tracker({"A": Pose(0, 0, 0)})
    for t in range(3):
        tr.update(t, {"A": Obs(10.0, 1.6)}, DRONE)
    s = tr.update(3, {"A": Obs(10.0, 1.6)}, SPEECH)
    assert s.raw_label == "person"  # speech is shown as person
    assert s.label == "drone"
    for t in range(4, 7):
        s = tr.update(t, {"A": Obs(10.0, 1.6)}, SPEECH)
    assert s.label == "person"


def test_label_drops_after_hold_without_clear_seconds():
    tr = Tracker({"A": Pose(0, 0, 0)}, hold_s=3)
    tr.update(0, {"A": Obs(10.0, 1.6)}, DRONE)
    assert tr.update(2, {"A": Obs(10.0, 1.0)}, DRONE).label == "drone"  # held
    assert tr.update(4, {"A": Obs(10.0, 1.0)}, DRONE).label == NONE


def test_two_clear_arrays_give_a_fix_and_trail():
    a, b = Pose(0, 0, 0), Pose(2, 0, 0)
    tr = Tracker({"A": a, "B": b}, trail=2)
    for t in range(3):
        s = tr.update(t, {"A": Obs(bearing_to(a, 1, 2), 1.6), "B": Obs(bearing_to(b, 1, 2), 1.6)}, DRONE)
    assert (s.fix.x, s.fix.y) == pytest.approx((1, 2))
    assert len(s.trail) == 2 and s.trail[-1].fix == pytest.approx((1, 2))
    js = s.to_json()
    assert js["fix"]["x"] == pytest.approx(1) and js["label"] == "drone"


def test_unsure_top_class_is_not_shown():
    tr = Tracker({"A": Pose(0, 0, 0)}, min_prob=0.6)
    s = tr.update(0, {"A": Obs(10.0, 1.6)}, {"drone": 0.4, "quiet": 0.35, "speech": 0.25})
    assert s.label == NONE and s.why == "unsure" and s.raw_label == "drone"
    s = tr.update(1, {"A": Obs(10.0, 1.6)}, {"drone": 0.95, "quiet": 0.05})
    assert s.label == "drone" and s.why == ""


def test_gate_is_adjustable_per_observation():
    assert not Obs(0.0, 1.15).clear
    assert Obs(0.0, 1.15, gate=1.1).clear


def test_quiet_never_counts_even_with_a_clear_direction():
    tr = Tracker({"A": Pose(0, 0, 0)})
    s = tr.update(0, {"A": Obs(180.0, 2.0)}, {"quiet": 0.9, "drone": 0.1})  # a directional fan
    assert s.label == NONE and s.why == "quiet" and not s.clear and s.trail == []
    s = tr.update(1, {"A": Obs(10.0, 1.6)}, DRONE)
    assert s.label == "drone"
