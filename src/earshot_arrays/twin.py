"""2D digital twin of the office: floor plan, arrays and their rays, the probability grid, the estimate.

    plot = TwinPlot(office, [(name, pose), ...], grid)
    plot.update(p, rays, estimate, cls, t, scans)   # once per block; p is (ny, nx), p[iy, ix], iy north

Looks like the browser live map (`arrays live`): dark, the probability map and the rays in the colour of
the class, light wedges toward the sound from each array's SRP scan, a dashed circle around the most
likely cell. Built once, then only artist data changes per update, so 1 update/s is cheap.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from earshot_arrays.config import Pose
from earshot_arrays.doa import SCAN_BINS
from earshot_arrays.locate import direction
from earshot_arrays.office import Office

if TYPE_CHECKING:
    from earshot_arrays.belief import Estimate, Grid

MIN_POWER = 1.3  # as cli.MIN_POWER: a direction stands out
MARGIN = 0.8  # metres around the bounds
HUD_BAND = 1.4  # metres above the bounds for the class label and status
FIG_W, FIG_H = 11.0, 8.5  # inches the figure may take at most (fits a laptop screen or a 1080p projector)
MIN_MASS_NEAR = 0.3  # the estimate is drawn only if this much mass lies within belief.NEAR_M of it

SCAN_R = 5.0  # metres the scan wedges reach from an array
# the live map's palette (livemap.PAGE)
SURROUND = "#0d1117"
FLOOR = "#161b22"
WALL = "#7d8590"
ARRAY = "#e6edf3"
FG, DIM = "#e6edf3", "#7d8590"
GREY = "#484f58"
COLORS = {"drone": "#ff4d4f", "plane": "#4ea1ff", "person": "#3fb950", "speech": "#3fb950", "music": "#bc8cff",
          "car": "#ff9f43", "fan": "#39c5cf", "quiet": "#7d8590", "other": "#7d8590"}
AREA_STYLE = {  # kind -> (face, edge)
    "room": ("#1b2027", "#30363d"),
    "furniture": ("#21262d", "#3d444d"),
    "glass": ("#16263a", "#2f5a85"),
}


def colour(label: str | None) -> str:
    return COLORS.get(label or "", "#c9d1d9")


def heat_rgba(p: np.ndarray, rgb: tuple[float, float, float], fade: float) -> np.ndarray:
    """(ny, nx, 4): every cell in the class colour, alpha as on the live map (share of the top cell^1.5)."""
    q = p / max(float(p.max()), 1e-300)
    out = np.empty(p.shape + (4,))
    out[..., :3] = rgb
    a = q**1.5 * 0.7 * fade
    out[..., 3] = np.where(a < 0.02, 0.0, a)
    return out


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
        self.heat = ax.imshow(np.zeros((grid.ny, grid.nx, 4)), origin="lower", interpolation="nearest",
                              extent=(self.xe[0], self.xe[-1], self.ye[0], self.ye[-1]), zorder=2.2)
        self.heat.set_clip_path(floor_patch)
        edges = ax.add_collection(LineCollection(self._cell_edges(), colors=[(1, 1, 1, 0.05)], linewidths=0.4,
                                                 zorder=2.1))
        edges.set_clip_path(floor_patch)
        self.n_floor = int(np.count_nonzero(grid.mask)) if grid.mask is not None else grid.nx * grid.ny

        if vectors and office.walls:
            ax.add_collection(LineCollection(office.walls, colors=WALL, linewidths=4.5, capstyle="projecting",
                                             joinstyle="miter", zorder=5))

        from matplotlib.collections import PolyCollection

        self.reach = 2.0 * math.hypot(xmax - xmin, ymax - ymin)
        self.rays, self.wedges, self.outlines = [], [], []
        clip = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, transform=ax.transData)  # rays stop at the bounds
        step = 360.0 / SCAN_BINS
        for name, pose in arrays:
            polys = []
            for i in range(SCAN_BINS):  # one wedge per scan bin, as the live map draws them
                arc = [direction(i * step + k * step / 4) for k in range(5)]
                polys.append([(pose.x, pose.y)] + [(pose.x + SCAN_R * dx, pose.y + SCAN_R * dy) for dx, dy in arc])
            wedges = ax.add_collection(PolyCollection(polys, facecolors=[(0, 0, 0, 0)] * SCAN_BINS,
                                                      edgecolors="none", zorder=2.5))
            wedges.set_clip_path(floor_patch)
            self.wedges.append(wedges)
            (outline,) = ax.plot([], [], color=GREY, lw=1.5, alpha=0.9, zorder=3.5)
            self.outlines.append(outline)
            (line,) = ax.plot([], [], ls=(0, (6, 6)), color=GREY, lw=1.6, zorder=4)
            line.set_clip_path(clip)
            self.rays.append(line)
            ax.plot(pose.x, pose.y, "s", ms=11, color=ARRAY, mec=SURROUND, mew=1.5, zorder=8)
            fx, fy = direction(pose.rotation_deg)
            ax.annotate("", (pose.x + 0.7 * fx, pose.y + 0.7 * fy), (pose.x, pose.y), zorder=8,
                        arrowprops={"arrowstyle": "-|>", "color": ARRAY, "lw": 1.5, "shrinkA": 6, "shrinkB": 0})
            ax.text(pose.x + 0.25, pose.y - 0.25, name, ha="left", va="top", fontsize=11, weight="bold",
                    color=ARRAY, zorder=8)

        # most likely cell: dashed circle holding half the probability, and a crosshair (as the live map)
        self.blob = Circle((0, 0), 0.0, facecolor="none", edgecolor=FG, ls=(0, (4, 4)), lw=1.8, visible=False,
                           zorder=9)
        ax.add_patch(self.blob)
        (self.dot,) = ax.plot([], [], "+", ms=16, mew=2, color=FG, zorder=9)
        self.where = ax.text(0, 0, "", fontsize=9, color=FG, weight="bold", zorder=9, visible=False,
                             bbox={"boxstyle": "round,pad=0.25", "fc": SURROUND, "ec": "none", "alpha": 0.8})

        top = ymax + MARGIN + HUD_BAND  # HUD in a band above the plan, so it never hides the floor
        hud = {"va": "top", "ha": "left", "zorder": 10}
        self.label = ax.text(xmin, top - 0.1, "", fontsize=20, weight="bold", **hud)
        self.sub = ax.text(xmin, top - 0.8, "", fontsize=9, color=DIM, linespacing=1.5, **hud)

        # north arrow (top right, in the HUD band) and a 1 m scale bar in the bottom margin
        ax.annotate("N", (xmax, top - 0.2), (xmax, top - 1.2), ha="center", va="top",
                    fontsize=11, weight="bold", color=FG, zorder=10,
                    arrowprops={"arrowstyle": "-|>", "color": FG, "lw": 1.8})
        sy = ymin - 0.45 * MARGIN
        ax.add_patch(Rectangle((xmax - 1.0, sy - 0.04), 1.0, 0.08, color=DIM, zorder=10, clip_on=False))
        ax.text(xmax - 0.5, sy - 0.1, "1 m", ha="center", va="top", fontsize=8, color=DIM)

        ax.set_xlim(xmin - MARGIN, xmax + MARGIN)
        ax.set_ylim(ymin - MARGIN, top)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(math.ceil(xmin), math.floor(xmax) + 1, 2))
        ax.set_yticks(np.arange(math.ceil(ymin), math.floor(ymax) + 1, 2))
        ax.tick_params(colors=GREY, labelsize=7, length=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(office.name, fontsize=13, color=FG, loc="left", weight="bold")
        self.fig.tight_layout()

    def set_mock_source(self, x: float, y: float) -> None:
        """Mark where a mocked array's virtual source is (a hollow star)."""
        if not hasattr(self, "mock_marker"):
            (self.mock_marker,) = self.ax.plot([], [], "*", ms=20, mfc="none", mec=FG, mew=1.8, zorder=9.5)
        self.mock_marker.set_data([x], [y])

    def on_click(self, fn) -> None:
        """Call fn(x, y) in room metres when the map is clicked."""
        def handler(ev):
            if ev.inaxes is self.ax and ev.xdata is not None:
                fn(float(ev.xdata), float(ev.ydata))

        self.fig.canvas.mpl_connect("button_press_event", handler)

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

    def update(self, p: np.ndarray, rays, estimate: Estimate | None, cls, t: float, scans=None) -> None:
        from matplotlib.colors import to_rgb

        p = np.asarray(p, dtype=float)
        label = None if cls is None else cls.label
        lc = colour(label)
        strong = any(ray is not None and ray[1] >= MIN_POWER for ray in rays)
        # a flat map stays dark: fade in once the top cell holds well over the uniform share (as the live map)
        concentration = float(p.max()) * self.n_floor
        fade = min(1.0, max(0.0, (concentration - 2.0) / 15.0))
        self.heat.set_data(heat_rgba(p, to_rgb(lc), fade))

        for i, ((_, pose), ray) in enumerate(zip(self.arrays, rays)):
            line, wedges, outline = self.rays[i], self.wedges[i], self.outlines[i]
            if ray is None:
                line.set_data([], [])
                continue
            b, power = ray
            on = power >= MIN_POWER
            c = lc if on else GREY
            dx, dy = direction(b)
            line.set_data([pose.x, pose.x + self.reach * dx], [pose.y, pose.y + self.reach * dy])
            line.set_color(c)
            line.set_linewidth(4.0 if on else 1.6)
            line.set_linestyle("-" if on else (0, (6, 6)))
            scan = None if scans is None else scans[i]
            if scan is None or len(scan) != SCAN_BINS:
                wedges.set_facecolors([(0, 0, 0, 0)] * SCAN_BINS)
                outline.set_data([], [])
                continue
            sc = np.asarray(scan, dtype=float)
            sc = (sc - sc.min()) / max(float(sc.max() - sc.min()), 1e-6)
            rgb = to_rgb(c if on else DIM)
            alpha = sc**3 * (0.6 if on else 0.2)
            wedges.set_facecolors([(*rgb, a if a >= 0.01 else 0.0) for a in alpha])
            ang = (np.arange(SCAN_BINS + 1) % SCAN_BINS) * (360.0 / SCAN_BINS) + 180.0 / SCAN_BINS
            r = SCAN_R * (0.12 + 0.45 * sc[np.arange(SCAN_BINS + 1) % SCAN_BINS])
            outline.set_data(pose.x + r * np.sin(np.radians(ang)), pose.y + r * np.cos(np.radians(ang)))
            outline.set_color(c)

        # a streak (one array, near-parallel rays) or a flat belief has an argmax, but no position to show
        shown = estimate is not None and strong and estimate.mass_near >= MIN_MASS_NEAR
        if shown:
            self.dot.set_data([estimate.x], [estimate.y])
            self.blob.set_center((estimate.x, estimate.y))
            self.blob.set_radius(max(estimate.r50, 0.3))
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
            self.dot.set_data([], [])
            self.blob.set_visible(False)
            self.where.set_visible(False)

        lines = []
        if cls is not None:
            self.label.set_text(f"{cls.label.upper()}  {cls.probs[cls.label]:.0%}")
            self.label.set_color(lc)
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
