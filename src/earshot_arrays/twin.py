"""2D digital twin of the office: floor plan, arrays and their rays, the probability grid, the estimate.

    plot = TwinPlot(office, [(name, pose), ...], grid)
    plot.update(p, rays, estimate, cls, t)   # once per block; p is (ny, nx), p[iy, ix], iy north, ix east

Built once, then only artist data changes per update, so 1 update/s with a ~30 x 20 grid is cheap.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from earshot_arrays.config import Pose
from earshot_arrays.locate import direction
from earshot_arrays.office import Office

if TYPE_CHECKING:
    from earshot_arrays.belief import Estimate, Grid

MIN_POWER = 1.3  # as cli.MIN_POWER: a direction stands out
MARGIN = 0.8  # metres around the bounds
HUD_BAND = 1.4  # metres above the bounds for the class label and status
FIG_W, FIG_H = 11.0, 8.5  # inches the figure may take at most (fits a laptop screen or a 1080p projector)
HEAT_FLOOR = 8.0  # colour scale tops out at least at this many times the uniform share: a flat belief stays pale
MIN_MASS_NEAR = 0.3  # the estimate is drawn only if this much mass lies within belief.NEAR_M of it

SURROUND = "#dcd8d0"
FLOOR = "#fbfaf6"
WALL = "#3b3b3b"
ARRAY = "#1f5fa8"
HOT = "#c62828"
AREA_STYLE = {  # kind -> (face, edge)
    "room": ("#f3f1ec", "#b9b4aa"),
    "furniture": ("#ece5d8", "#9c907e"),
    "glass": ("#e3eff8", "#86acc9"),
}


def heat_cmap(n: int = 256):
    """Warm sequential map, transparent at zero and opaque at the top."""
    from matplotlib import colormaps
    from matplotlib.colors import ListedColormap

    rgba = colormaps["YlOrRd"](np.linspace(0.15, 1.0, n))
    rgba[:, 3] = np.linspace(0.0, 1.0, n) ** 0.6 * 0.85
    return ListedColormap(rgba, name="earshot_heat").with_extremes(bad=(0, 0, 0, 0))


class TwinPlot:
    """Top-down office map that a live run updates once per block."""

    def __init__(self, office: Office, arrays: list[tuple[str, Pose]], grid: Grid, show: bool = True,
                 vectors: bool = True):
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Circle, Polygon, Rectangle

        self.plt, self.office, self.arrays, self.grid, self.show = plt, office, arrays, grid, show
        if show:
            plt.ion()
        xmin, ymin, xmax, ymax = office.bounds
        w, h = xmax - xmin + 2 * MARGIN, ymax - ymin + 2 * MARGIN + HUD_BAND
        scale = min(FIG_W / w, FIG_H / h)  # inches per metre
        self.fig, self.ax = plt.subplots(figsize=(w * scale, h * scale))
        self.fig.patch.set_facecolor(SURROUND)
        ax = self.ax
        ax.set_facecolor(SURROUND)

        floor = office.floor or ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))
        floor_patch = Polygon(floor, closed=True, facecolor=FLOOR, edgecolor="none", zorder=0)
        ax.add_patch(floor_patch)  # also the clip path of the backdrop and the grid: nothing past the walls
        if office.image is not None:
            img = plt.imread(office.image)
            im = ax.imshow(img, extent=office.image_extent(img.shape[1], img.shape[0]), origin="upper", zorder=0.5)
            im.set_clip_path(floor_patch)
        if vectors:
            for a in office.areas:
                face, edge = AREA_STYLE.get(a.kind, AREA_STYLE["room"])
                ax.add_patch(Polygon(a.polygon, closed=True, facecolor=face, edgecolor=edge, lw=0.8, zorder=1))
            for lb in office.labels:
                ax.text(*lb.at, lb.text, ha="center", va="center", fontsize=8, color="0.45", zorder=6)

        # probability grid: explicit cell edges, so p[iy, ix] fills [x0 + ix*cell, +cell) x [y0 + iy*cell, +cell)
        self.xe = grid.x0 + grid.cell * np.arange(grid.nx + 1)
        self.ye = grid.y0 + grid.cell * np.arange(grid.ny + 1)
        self.off_floor = None if grid.mask is None else ~np.asarray(grid.mask, dtype=bool)
        self.mesh = ax.pcolormesh(self.xe, self.ye, self._masked(np.zeros((grid.ny, grid.nx))), shading="flat",
                                  cmap=heat_cmap(), vmin=0.0, vmax=1.0, edgecolors="none", zorder=2)
        self.mesh.set_clip_path(floor_patch)
        edges = ax.add_collection(LineCollection(self._cell_edges(), colors=[(0, 0, 0, 0.13)], linewidths=0.4,
                                                 zorder=2.1))
        edges.set_clip_path(floor_patch)
        n_floor = int(np.count_nonzero(grid.mask)) if grid.mask is not None else grid.nx * grid.ny
        self.vmax_floor = HEAT_FLOOR / max(n_floor, 1)

        if vectors and office.walls:
            ax.add_collection(LineCollection(office.walls, colors=WALL, linewidths=4.5, capstyle="projecting",
                                             joinstyle="miter", zorder=5))

        self.reach = 2.0 * math.hypot(xmax - xmin, ymax - ymin)
        self.rays = []
        clip = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, transform=ax.transData)  # rays stop at the bounds
        for name, pose in arrays:
            (line,) = ax.plot([], [], ls=(0, (6, 4)), color=ARRAY, lw=1.6, zorder=3)
            line.set_clip_path(clip)
            self.rays.append(line)
            ax.plot(pose.x, pose.y, "s", ms=11, color=ARRAY, mec="white", mew=1.5, zorder=8)
            fx, fy = direction(pose.rotation_deg)
            ax.annotate("", (pose.x + 0.7 * fx, pose.y + 0.7 * fy), (pose.x, pose.y), zorder=8,
                        arrowprops={"arrowstyle": "-|>", "color": ARRAY, "lw": 1.5, "shrinkA": 6, "shrinkB": 0})
            ax.text(pose.x + 0.25, pose.y - 0.25, name, ha="left", va="top", fontsize=11, weight="bold",
                    color=ARRAY, zorder=8)

        self.blob = Circle((0, 0), 0.0, facecolor=HOT, edgecolor=HOT, alpha=0.15, lw=1, visible=False, zorder=6)
        ax.add_patch(self.blob)
        (self.ring,) = ax.plot([], [], "o", ms=18, mfc="none", mec=HOT, mew=2.8, zorder=9)
        (self.dot,) = ax.plot([], [], "o", ms=4, color=HOT, zorder=9)
        self.where = ax.text(0, 0, "", fontsize=9, color=HOT, weight="bold", zorder=9, visible=False,
                             bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.8})

        top = ymax + MARGIN + HUD_BAND  # HUD in a band above the plan, so it never hides the floor
        hud = {"va": "top", "ha": "left", "zorder": 10}
        self.label = ax.text(xmin, top - 0.1, "", fontsize=20, weight="bold", **hud)
        self.sub = ax.text(xmin, top - 0.8, "", fontsize=9, color="0.35", linespacing=1.5, **hud)

        # north arrow (top right, in the HUD band) and a 1 m scale bar in the bottom margin
        ax.annotate("N", (xmax, top - 0.2), (xmax, top - 1.2), ha="center", va="top",
                    fontsize=11, weight="bold", color="0.25", zorder=10,
                    arrowprops={"arrowstyle": "-|>", "color": "0.25", "lw": 1.8})
        sy = ymin - 0.45 * MARGIN
        ax.add_patch(Rectangle((xmax - 1.0, sy - 0.04), 1.0, 0.08, color="0.25", zorder=10, clip_on=False))
        ax.text(xmax - 0.5, sy - 0.1, "1 m", ha="center", va="top", fontsize=8, color="0.25")

        ax.set_xlim(xmin - MARGIN, xmax + MARGIN)
        ax.set_ylim(ymin - MARGIN, top)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(math.ceil(xmin), math.floor(xmax) + 1, 2))
        ax.set_yticks(np.arange(math.ceil(ymin), math.floor(ymax) + 1, 2))
        ax.tick_params(colors="0.6", labelsize=7, length=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(office.name, fontsize=13, color="0.2", loc="left", weight="bold")
        self.fig.tight_layout()

    def _cell_edges(self) -> list:
        """Segments of the cell grid, floor cells only (a masked QuadMesh still strokes its edges)."""
        g = self.grid
        on = np.ones((g.ny, g.nx), bool) if self.off_floor is None else ~self.off_floor
        pad = np.pad(on, 1)
        segs = []
        for j in range(g.ny + 1):  # horizontal edge j lies between rows j-1 and j
            for i in np.flatnonzero(pad[j, 1:-1] | pad[j + 1, 1:-1]):
                segs.append(((self.xe[i], self.ye[j]), (self.xe[i + 1], self.ye[j])))
        for i in range(g.nx + 1):
            for j in np.flatnonzero(pad[1:-1, i] | pad[1:-1, i + 1]):
                segs.append(((self.xe[i], self.ye[j]), (self.xe[i], self.ye[j + 1])))
        return segs

    def _masked(self, p: np.ndarray):
        return np.ma.masked_array(p, mask=self.off_floor) if self.off_floor is not None else p

    def update(self, p: np.ndarray, rays, estimate: Estimate | None, cls, t: float) -> None:
        p = np.asarray(p, dtype=float)
        self.mesh.set_array(self._masked(p))
        self.mesh.set_clim(0.0, max(float(p.max()), self.vmax_floor))

        strong = False
        for line, (_, pose), ray in zip(self.rays, self.arrays, rays):
            if ray is None:
                line.set_data([], [])
                continue
            b, power = ray
            dx, dy = direction(b)
            line.set_data([pose.x, pose.x + self.reach * dx], [pose.y, pose.y + self.reach * dy])
            line.set_alpha(1.0 if power >= MIN_POWER else 0.25)
            strong |= power >= MIN_POWER

        # a streak (one array, near-parallel rays) or a flat belief has an argmax, but no position to show
        shown = estimate is not None and strong and estimate.mass_near >= MIN_MASS_NEAR
        if shown:
            self.ring.set_data([estimate.x], [estimate.y])
            self.dot.set_data([estimate.x], [estimate.y])
            self.blob.set_center((estimate.x, estimate.y))
            self.blob.set_radius(estimate.r50)
            self.blob.set_visible(True)
            xmin, ymin, xmax, ymax = self.office.bounds
            east = estimate.x > xmin + 2 / 3 * (xmax - xmin)  # label left of the ring, so it stays in the figure
            north = estimate.y > ymin + 2 / 3 * (ymax - ymin)
            d = max(estimate.r50, 0.4) * 0.75
            self.where.set_position((estimate.x - d if east else estimate.x + d, estimate.y - d if north else estimate.y + d))
            self.where.set_ha("right" if east else "left")
            self.where.set_va("top" if north else "bottom")
            self.where.set_text(f"{estimate.x:.1f}, {estimate.y:.1f} m  ±{estimate.r50:.1f}")
            self.where.set_visible(True)
        else:
            self.ring.set_data([], [])
            self.dot.set_data([], [])
            self.blob.set_visible(False)
            self.where.set_visible(False)

        lines = []
        if cls is not None:
            self.label.set_text(f"{cls.label.upper()}  {cls.probs[cls.label]:.0%}")
            self.label.set_color(HOT if cls.label == "drone" else "0.2")
            lines.append(f"classifier: {cls.source}")
        else:
            self.label.set_text("")
        status = "" if shown else "   no clear direction" if not strong else "   direction only, no position yet"
        lines.append(f"t = {t:.1f} s{status}")  # two lines at most: the HUD band is only HUD_BAND high
        self.sub.set_text("\n".join(lines))
        if self.show:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.01)

    def hold(self) -> None:
        self.plt.ioff()
        self.plt.show()

    def save(self, path: str) -> None:
        self.fig.savefig(path, dpi=120, facecolor=self.fig.get_facecolor())
