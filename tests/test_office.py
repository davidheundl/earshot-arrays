from pathlib import Path

import numpy as np
import pytest
import yaml

from earshot_arrays.belief import Estimate, Grid
from earshot_arrays.classify import Result
from earshot_arrays.config import ArraySetup, Pose
from earshot_arrays.office import Office, load_office, solve_image_ref

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "office.example.yaml"


def test_example_loads():
    o = load_office(EXAMPLE)
    assert o.bounds == (0.0, 0.0, 14.0, 9.0)
    assert o.cell == 0.5
    assert set(o.arrays) == {"A", "B"}
    assert o.arrays["B"][0] - o.arrays["A"][0] == pytest.approx(7.0)
    assert {a.kind for a in o.areas} == {"room", "furniture", "glass"}
    g = o.grid()
    assert (g.nx, g.ny) == (28, 18)
    assert g.mask.shape == (18, 28) and g.mask.all()


def test_image_extent():
    o = Office("t", (0, 0, 1, 1), image_scale=0.1, image_origin_px=(20, 40))
    assert o.image_extent(100, 50) == pytest.approx((-2, 8, -1, 4))


def test_image_ref_solves_scale_and_origin(tmp_path):
    s, (u0, v0) = solve_image_ref([{"px": [20, 40], "m": [0, 0]}, {"px": [120, -60], "m": [10, 10]}])
    assert s == pytest.approx(0.1)
    assert (u0, v0) == pytest.approx((20, 40))
    with pytest.raises(ValueError):
        solve_image_ref([{"px": [0, 0], "m": [0, 0]}, {"px": [100, -100], "m": [10, 12]}])  # 20% apart
    p = tmp_path / "office.yaml"
    p.write_text(yaml.safe_dump({"name": "r", "bounds": [0, 0, 4, 3], "image": "plan.png",
                                 "image_ref": [{"px": [10, 310], "m": [0, 0]}, {"px": [410, 10], "m": [4, 3]}]}))
    o = load_office(p)
    assert o.image == tmp_path / "plan.png"
    assert o.image_scale == pytest.approx(0.01)
    assert o.image_extent(420, 320) == pytest.approx((-0.1, 4.1, -0.1, 3.1))


def test_office_marks_move_setup_poses_keep_rotation():
    o = Office("t", (0, 0, 10, 10), arrays={"B": (5.0, 6.0)})
    setup = [ArraySetup("A", None, None, Pose(0.1, 0.1, 0.0)), ArraySetup("B", None, None, Pose(1.5, 0.1, 90.0))]
    assert o.poses(setup) == [("A", Pose(0.1, 0.1, 0.0)), ("B", Pose(5.0, 6.0, 90.0))]  # calibrated rotation kept


def test_twin_renders_hot_cell_in_place(tmp_path):
    from earshot_arrays.twin import TwinPlot

    o = Office("t", (-1.0, 2.0, 5.0, 6.0), cell=0.5, walls=(((-1.0, 2.0), (5.0, 2.0)),))
    grid = Grid.from_bounds(*o.bounds, o.cell)
    plot = TwinPlot(o, [("A", Pose(0.0, 2.5, 0.0)), ("B", Pose(4.0, 2.5, 0.0))], grid, show=False)
    ix, iy = 7, 3
    p = np.zeros((grid.ny, grid.nx))
    p[iy, ix] = 1.0
    cls = Result({"drone": 0.8, "plane": 0.2}, "stand-in")
    plot.update(p, [(30.0, 2.0), None], Estimate(3.75, 3.75, 1.0, 0.4, 0.9), cls, 3.0)
    assert plot.heat.get_extent() == pytest.approx([-1.0, -1.0 + grid.nx * 0.5, 2.0, 2.0 + grid.ny * 0.5])
    drawn = np.asarray(plot.heat.get_array())  # (ny, nx, 4), row 0 south (origin lower)
    assert drawn.shape == (grid.ny, grid.nx, 4)
    assert np.unravel_index(drawn[..., 3].argmax(), drawn.shape[:2]) == (iy, ix)
    from matplotlib.colors import to_rgb

    from earshot_arrays.twin import COLORS
    assert tuple(drawn[iy, ix, :3]) == pytest.approx(to_rgb(COLORS["drone"]))  # the class colour
    assert plot.label.get_text() == "DRONE  80%"
    plot.update(p * 0, [None, None], None, None, 4.0)  # one array or none, no estimate, no class
    assert "no clear direction" in plot.sub.get_text()
    out = tmp_path / "twin.png"
    plot.save(str(out))
    assert out.stat().st_size > 0


def test_twin_hides_estimate_without_position_and_keeps_flat_belief_pale():
    from earshot_arrays.twin import TwinPlot

    o = Office("t", (0.0, 0.0, 6.0, 4.0), cell=0.5)
    grid = Grid.from_bounds(*o.bounds, o.cell)
    plot = TwinPlot(o, [("A", Pose(3.0, 0.5, 0.0))], grid, show=False)
    flat = np.full((grid.ny, grid.nx), 1.0 / (grid.nx * grid.ny))
    plot.update(flat, [(10.0, 1.1)], Estimate(0.25, 0.25, flat.max(), 2.5, 0.05), None, 0.0)  # silence
    assert np.asarray(plot.heat.get_array())[..., 3].max() == 0.0  # a flat belief stays dark
    assert not plot.where.get_visible() and not plot.blob.get_visible()
    plot.update(flat, [(10.0, 2.0)], Estimate(3.25, 3.25, flat.max(), 2.5, 0.1), None, 1.0)  # a streak
    assert not plot.where.get_visible()
    assert "no position" in plot.sub.get_text()
    plot.update(flat, [(10.0, 2.0)], Estimate(5.75, 3.75, flat.max(), 0.5, 0.6), None, 2.0)  # north-east corner
    assert plot.where.get_visible()
    assert (plot.where.get_ha(), plot.where.get_va()) == ("right", "top")
