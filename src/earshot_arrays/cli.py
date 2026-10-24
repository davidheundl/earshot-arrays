"""`arrays list | meter | bearing | locate | record | synth`. Every live command takes `--replay`."""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from earshot_arrays.config import ArraySetup, Pose, Setup, load_profile, load_setup, write_setup
from earshot_arrays.devices import DeviceSource, list_inputs, mics
from earshot_arrays.replay import ReplaySource, replay_files, write_wav

MIN_POWER = 1.3  # SRP peak/mean below this: no direction stands out, no ray is drawn


def cmd_list(_a) -> None:
    for i, ch, fs, name in list_inputs():
        print(f"{i:3d}  {ch:2d} ch  {fs:7.0f} Hz  {name}")


def single_array(a) -> ArraySetup:
    """The array for meter/bearing: --setup + --array, or --profile (+ --device, --rotation)."""
    if a.setup:
        s = load_setup(a.setup)
        if a.array is None:
            return s.arrays[0]
        for arr in s.arrays:
            if arr.name == a.array:
                return arr
        raise SystemExit(f"no array {a.array!r} in {a.setup}")
    if not a.profile:
        raise SystemExit("give --setup or --profile")
    return ArraySetup("A", load_profile(a.profile), a.device, Pose(0.0, 0.0, a.rotation))


def open_sources(arrays: tuple[ArraySetup, ...] | list[ArraySetup], a) -> list:
    if a.replay:
        files = replay_files(a.replay, [arr.name for arr in arrays])
        sources = [ReplaySource(f, realtime=a.realtime) for f in files]
        for arr, src, f in zip(arrays, sources, files):
            if src.n_channels != arr.profile.n_channels or src.fs != arr.profile.fs:
                raise SystemExit(f"{f}: {src.n_channels} ch at {src.fs} Hz, profile of {arr.name} says "
                                 f"{arr.profile.n_channels} ch at {arr.profile.fs} Hz")
        return sources
    return [DeviceSource(arr.device, arr.profile) for arr in arrays]


def cmd_meter(a) -> None:
    from earshot_arrays import meter

    a.realtime = True  # levels are for watching
    if a.setup or a.profile:
        arr = single_array(a)
        src = open_sources([arr], a)[0]
        chans = arr.profile.mic_channels
    elif a.replay:
        src, chans = ReplaySource(a.replay[0], realtime=True), ()
    else:  # before any profile exists: open the device with all its input channels
        import sounddevice as sd

        from earshot_arrays.config import Profile
        from earshot_arrays.devices import resolve_device

        info = sd.query_devices(resolve_device(a.device), "input")
        prof = Profile("raw", info["max_input_channels"], int(info["default_samplerate"]), (), np.zeros((2, 0)))
        src, chans = DeviceSource(a.device, prof), ()
    try:
        meter.run(src, chans)
    except KeyboardInterrupt:
        pass
    finally:
        src.close()


def cmd_map(a) -> None:
    import re

    from earshot_arrays.meter import map_channels

    prof = load_profile(a.profile)
    src = DeviceSource(a.device, prof, block_s=0.1)
    n = len(prof.mic_channels)
    print(f"{prof.name}: tap each mic position 0..{n - 1} once, firmly, with a fingernail. Ctrl-C aborts.")
    try:
        chans = map_channels(src, n)
    except KeyboardInterrupt:
        print("\naborted, profile unchanged")
        return
    finally:
        src.close()
    text = Path(a.profile).read_text()
    new = re.sub(r"^mic_channels:.*$", f"mic_channels: {chans}", text, count=1, flags=re.MULTILINE)
    Path(a.profile).write_text(new)
    print(f"mic_channels: {chans}\nwritten to {a.profile}")


def cmd_bearing(a) -> None:
    from earshot_arrays.doa import estimate_bearing

    arr = single_array(a)
    src = open_sources([arr], a)[0]
    print(f"{arr.name}: {arr.profile.name}, rotation {arr.pose.rotation_deg:.0f} deg. Ctrl-C to stop.")
    try:
        while (block := src.read(latest=True)) is not None:
            b, p = estimate_bearing(mics(block, arr.profile), arr.profile.fs, arr.profile.mic_xy,
                                    arr.pose.rotation_deg)
            flag = "" if p >= MIN_POWER else "  (weak, no clear direction)"
            print(f"bearing {b:6.1f} deg   power {p:5.2f}{flag}")
    except KeyboardInterrupt:
        pass
    finally:
        src.close()


def cmd_classify(a) -> None:
    from earshot_arrays.classify import load_classifier
    from earshot_arrays.doa import estimate_bearing

    arr = single_array(a)
    src = open_sources([arr], a)[0]
    clf = load_classifier()
    print(f"{arr.name}: {arr.profile.name}; classifier: {clf.name} {getattr(clf, 'reason', '')}. Ctrl-C to stop.")
    try:
        while (block := src.read(latest=True)) is not None:
            m = mics(block, arr.profile)
            r = clf(m.mean(axis=0), arr.profile.fs)
            b, p = estimate_bearing(m, arr.profile.fs, arr.profile.mic_xy, arr.pose.rotation_deg)
            top = sorted(r.fine.items(), key=lambda kv: -kv[1])[:3]
            fine = "  (" + ", ".join(f"{k} {v:.0%}" for k, v in top) + ")" if top else ""
            level = 20 * np.log10(max(float(np.sqrt(np.mean(np.square(m)))), 1e-12))
            print(f"{r.label.upper():6s} {r.probs[r.label]:4.0%}{fine}   bearing {b:5.1f}° p{p:4.2f}"
                  f"   {level:5.1f} dBFS")
    except KeyboardInterrupt:
        pass
    finally:
        src.close()


def cmd_locate(a) -> None:
    from earshot_arrays.classify import load_classifier
    from earshot_arrays.doa import estimate_bearing
    from earshot_arrays.locate import intersect

    s = load_setup(a.setup)
    if len(s.arrays) != 2:
        raise SystemExit(f"locate needs exactly two arrays in {a.setup}, found {len(s.arrays)}")
    sources = open_sources(s.arrays, a)
    classify = load_classifier()
    plot = None if a.no_plot else LivePlot(s, show=not a.save_plot)
    print(f"classifier: {classify.name} {getattr(classify, 'reason', '')}")
    t, ended = 0, False
    try:
        while True:
            blocks = [src.read(latest=True) for src in sources]
            if any(b is None for b in blocks):
                ended = True
                break
            rays = []
            for arr, block in zip(s.arrays, blocks):
                b, p = estimate_bearing(mics(block, arr.profile), arr.profile.fs, arr.profile.mic_xy,
                                        arr.pose.rotation_deg)
                rays.append((b, p))
            fix = None
            if all(p >= MIN_POWER for _, p in rays):
                fix = intersect(s.arrays[0].pose, rays[0][0], s.arrays[1].pose, rays[1][0])
            # classify the louder array's mic mix
            k = int(np.argmax([np.mean(np.square(b)) for b in blocks]))
            arr = s.arrays[k]
            cls = classify(mics(blocks[k], arr.profile).mean(axis=0), arr.profile.fs)
            where = f"fix ({fix.x:5.2f}, {fix.y:5.2f}) m ± {fix.sigma:.2f}" if fix else "no fix"
            print(f"t={t:4d}s  " + "  ".join(f"{arr.name} {b:6.1f}° p{p:4.2f}" for arr, (b, p) in
                                           zip(s.arrays, rays)) + f"  {where}  {cls.label} [{cls.source}]")
            if plot:
                plot.update(rays, fix, cls)
            t += 1
    except KeyboardInterrupt:
        pass
    finally:
        for src in sources:
            src.close()
        if plot and a.save_plot:
            plot.save(a.save_plot)
            print(f"plot saved to {a.save_plot}")
    if plot and plot.show and ended:  # end of a replay: keep the last frame up until the window is closed
        print("end of recording; close the plot window to exit")
        plot.hold()


class LivePlot:
    """Top-down table: both arrays, both rays, the fix with its uncertainty, the class label."""

    def __init__(self, s: Setup, show: bool = True):
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        self.plt, self.s, self.show = plt, s, show
        if show:
            plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        w, d = s.table
        ax = self.ax
        ax.add_patch(Rectangle((0, 0), w, d, fill=False, lw=2, color="0.3"))
        ax.text(w / 2, d + 0.03, "N", ha="center", va="bottom", fontsize=12, weight="bold")
        self.reach = 2.0 * math.hypot(w, d)
        self.rays = []
        for arr in s.arrays:
            ax.plot(arr.pose.x, arr.pose.y, "s", ms=10, color="tab:blue")
            ax.text(arr.pose.x, arr.pose.y - 0.06, arr.name, ha="center", va="top")
            fwd = math.radians(arr.pose.rotation_deg)
            ax.annotate("", (arr.pose.x + 0.08 * math.sin(fwd), arr.pose.y + 0.08 * math.cos(fwd)),
                        (arr.pose.x, arr.pose.y), arrowprops={"arrowstyle": "->", "color": "tab:blue"})
            (line,) = ax.plot([], [], "-", color="tab:orange", lw=1.5)
            self.rays.append(line)
        (self.dot,) = ax.plot([], [], "o", ms=12, color="tab:red")
        self.blob = Circle((0, 0), 0.0, color="tab:red", alpha=0.15)
        ax.add_patch(self.blob)
        self.label = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=14, weight="bold")
        self.sub = ax.text(0.02, 0.90, "", transform=ax.transAxes, va="top", fontsize=9, color="0.4")
        m = 0.3
        ax.set_xlim(-m, w + m)
        ax.set_ylim(-m, d + m)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m), north up")
        ax.set_title("Earshot Arrays")

    def update(self, rays, fix, cls) -> None:
        from earshot_arrays.locate import direction

        for line, arr, (b, p) in zip(self.rays, self.s.arrays, rays):
            dx, dy = direction(b)
            line.set_data([arr.pose.x, arr.pose.x + self.reach * dx], [arr.pose.y, arr.pose.y + self.reach * dy])
            line.set_alpha(1.0 if p >= MIN_POWER else 0.2)
        if fix:
            self.dot.set_data([fix.x], [fix.y])
            self.blob.set_center((fix.x, fix.y))
            self.blob.set_radius(fix.sigma)
        else:
            self.dot.set_data([], [])
            self.blob.set_radius(0.0)
        prob = cls.probs[cls.label]
        self.label.set_text(f"{cls.label.upper()}  {prob:.0%}")
        self.label.set_color("tab:red" if cls.label == "drone" else "0.2")
        where = f"fix ({fix.x:.2f}, {fix.y:.2f}) m ± {fix.sigma:.2f} m" if fix else "no fix"
        self.sub.set_text(f"classifier: {cls.source}\n{where}")
        if self.show:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.01)

    def hold(self) -> None:
        self.plt.ioff()
        self.plt.show()

    def save(self, path: str) -> None:
        self.fig.savefig(path, dpi=120, bbox_inches="tight")


def cmd_record(a) -> None:
    s = load_setup(a.setup)
    out = Path(a.out or f"recordings/{datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    sources = open_sources(s.arrays, a)
    blocks: list[list[np.ndarray]] = [[] for _ in s.arrays]
    print(f"recording {a.seconds:.0f} s from {', '.join(arr.name for arr in s.arrays)}. Ctrl-C ends early.")
    start = time.monotonic()
    try:
        while time.monotonic() - start < a.seconds:
            for i, src in enumerate(sources):
                blocks[i].append(src.read())  # every block, in order: no dropping while recording
    except KeyboardInterrupt:
        pass
    finally:
        for src in sources:
            src.close()
    n = min(len(b) for b in blocks)
    for arr, b in zip(s.arrays, blocks):
        write_wav(out / f"{arr.name}.wav", b[:n], arr.profile.fs)
    write_setup(s, out)
    print(f"{n} s written to {out}/\nreplay: uv run arrays locate --setup {out}/setup.yaml --replay {out}")


def cmd_train(a) -> None:
    from earshot_arrays import train
    from earshot_arrays.classify import ROOM_HEAD

    extra = []
    for e in a.extra or []:
        label, _, path = e.partition("=")
        extra.append((label, Path(path)))
    train.run(Path(a.recordings), Path(a.out) if a.out else ROOM_HEAD, extra)


def cmd_synth(a) -> None:
    from earshot_arrays.synth import synth_session

    s = load_setup(a.setup)
    pts = [tuple(float(v) for v in p.split(",")) for p in a.path]
    out = synth_session(s, pts, a.out, a.seconds_per_point)
    print(f"synthetic session in {out}/\nreplay: uv run arrays locate --setup {out}/setup.yaml --replay {out}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="arrays")
    sub = ap.add_subparsers(required=True)

    def live(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--replay", nargs="+", metavar="WAV|DIR",
                       help="recording directory or one WAV per array, in place of the devices")
        p.add_argument("--realtime", action="store_true", help="pace replay at 1 s per block")
        return p

    def one_array(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--setup", help="setup YAML (with --array NAME, default the first)")
        p.add_argument("--array")
        p.add_argument("--profile", help="array profile YAML, if no setup")
        p.add_argument("--device", help="device index or name substring (see `arrays list`)")
        p.add_argument("--rotation", type=float, default=0.0, help="room bearing of the array's forward axis")
        return live(p)

    sub.add_parser("list", help="audio inputs with 2+ channels").set_defaults(fn=cmd_list)
    one_array(sub.add_parser("meter", help="live per-channel levels (find the channel order)")).set_defaults(
        fn=cmd_meter)
    p = sub.add_parser("map", help="tap mics 0..n-1 in turn; writes the channel order into the profile")
    p.add_argument("--profile", required=True)
    p.add_argument("--device", required=True, help="device index or name substring")
    p.set_defaults(fn=cmd_map)
    one_array(sub.add_parser("classify", help="live sound class (+ bearing) of one array")).set_defaults(
        fn=cmd_classify)
    one_array(sub.add_parser("bearing", help="live bearing of one array")).set_defaults(fn=cmd_bearing)

    p = live(sub.add_parser("locate", help="two arrays → position + class, live plot"))
    p.add_argument("--setup", default="setup.yaml")
    p.add_argument("--no-plot", action="store_true", help="print fixes only")
    p.add_argument("--save-plot", metavar="PNG", help="no window; save the last frame here")
    p.set_defaults(fn=cmd_locate)

    from earshot_arrays.livemap import add_parser as add_live_map
    add_live_map(sub, live)

    p = sub.add_parser("record", help="record both arrays for later --replay")
    p.add_argument("--setup", default="setup.yaml")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--out", help="directory (default recordings/<timestamp>)")
    p.set_defaults(fn=cmd_record, replay=None, realtime=False)

    p = sub.add_parser("train", help="train the room head on recordings/<label><n>/ (EfficientAT stays frozen)")
    p.add_argument("--recordings", default="recordings")
    p.add_argument("--extra", nargs="*", metavar="LABEL=WAV", help="more clips, e.g. drone=clean.wav")
    p.add_argument("--out", help="default data/models/room_head.pt")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("synth", help="write a synthetic two-array session (rehearsal without hardware)")
    p.add_argument("--setup", default="setup.example.yaml")
    p.add_argument("--path", nargs="+", default=["0.8,0.6", "0.4,0.5", "1.2,0.4"], metavar="X,Y",
                   help="source positions in metres, visited in turn")
    p.add_argument("--seconds-per-point", type=int, default=3)
    p.add_argument("--out", default="recordings/synthetic")
    p.set_defaults(fn=cmd_synth)

    a = ap.parse_args()
    a.fn(a)
