"""Train a room head: frozen EfficientAT features -> our own classes, from our own recordings.

    uv run arrays train                        # recordings/<label><n>[_<tag>]/ -> data/models/room_head.pt

Every recording directory named <label><n>, optionally with a tag such as the distance
(drone7_3m), is labelled by its leading letters: drone1, drone7_3m -> "drone"; quiet2 -> "quiet".
Other directories (demo_room1, synthetic) are skipped. Each recording is cut into 1 s windows
(0.5 s hop), and each window becomes the delay-and-sum beam toward the bearing measured in that
window, as `arrays live` classifies (--mix: the plain mic mean instead). In a recording of a sound,
windows more than LOUD_DB below the recording's median level are dropped (the sound had stopped).
The last TEST_SHARE of every recording is held out to report how well the head does on audio it
did not see; the saved head is then trained on everything.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import soundfile as sf

from earshot_arrays.config import load_setup

WIN_S, HOP_S = 1.0, 0.5
LOUD_DB = 3.0
TEST_SHARE = 0.25
HIDDEN = 128
QUIET = "quiet"


NAME = re.compile(r"([a-z]+)\d+(_[a-z0-9]+)?")


def label_of(name: str) -> str | None:
    """drone7_3m -> "drone"; None for directories that are not training data."""
    m = NAME.fullmatch(name)
    return m.group(1) if m else None


def windows(path: Path, profile=None, mix: bool = False) -> tuple[list[np.ndarray], np.ndarray, int]:
    """1 s mono windows of a WAV, their levels in dBFS, and the rate. With an array profile each
    window is the beam toward its own measured bearing (or the mic mean with mix); without one
    (a plain clip) the channels are averaged."""
    from earshot_arrays.doa import beam, estimate_bearing

    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.T[list(profile.mic_channels)] if profile else x.T
    n, h = int(WIN_S * fs), int(HOP_S * fs)
    wins = []
    for i in range(0, x.shape[1] - n + 1, h):
        m = x[:, i:i + n]
        if profile is None or mix:
            wins.append(m.mean(axis=0))
        else:
            b, _ = estimate_bearing(m, fs, profile.mic_xy)
            wins.append(beam(m, fs, profile.mic_xy, b))
    levels = np.array([20 * np.log10(max(float(np.sqrt(np.mean(w**2))), 1e-12)) for w in wins])
    return wins, levels, fs


def collect(recordings: Path, extra: list[tuple[str, Path]] = (), mix: bool = False) -> list[tuple[str, str, list, int]]:
    """(source name, label, windows, fs) per recording, with quiet tails dropped."""
    out = []
    for d in sorted(p for p in recordings.iterdir() if p.is_dir() and label_of(p.name)):
        if not (d / "setup.yaml").exists():
            continue  # still recording
        setup = load_setup(d / "setup.yaml")
        for arr in setup.arrays:
            wav = d / f"{arr.name}.wav"
            if wav.exists():
                out.append((f"{d.name}/{arr.name}", label_of(d.name), wav, arr.profile))
    for label, wav in extra:
        out.append((wav.name, label, wav, None))
    result = []
    for name, label, wav, profile in out:
        wins, levels, fs = windows(wav, profile, mix)
        if label != QUIET:
            keep = levels >= np.median(levels) - LOUD_DB
            wins = [w for w, k in zip(wins, keep) if k]
        result.append((name, label, wins, fs))
    return result


def fit(X: np.ndarray, y: np.ndarray, n_classes: int, epochs: int = 300, seed: int = 0):
    """MLP head (as in classify.EfficientAT) on standardised features, class-weighted."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
    Xt = torch.from_numpy(((X - mean) / std).astype(np.float32))
    yt = torch.from_numpy(y.astype(np.int64))
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weight = torch.from_numpy(counts.sum() / np.maximum(counts, 1) / n_classes)
    net = nn.Sequential(nn.Linear(X.shape[1], HIDDEN), nn.ReLU(), nn.Dropout(0.3), nn.Linear(HIDDEN, n_classes))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(net(Xt), yt).backward()
        opt.step()
    net.eval()
    return net, mean, std


def predict(net, mean, std, X: np.ndarray) -> np.ndarray:
    import torch

    with torch.inference_mode():
        z = torch.from_numpy(((X - mean) / std).astype(np.float32))
        return torch.softmax(net(z), dim=1).numpy()


def confusion(y: np.ndarray, p: np.ndarray, names: list[str]) -> str:
    k = len(names)
    m = np.zeros((k, k), int)
    for t, q in zip(y, p):
        m[t, q] += 1
    w = max(len(n) for n in names)
    lines = [" " * (w + 2) + " ".join(f"{n[:6]:>6}" for n in names) + "  recall"]
    for i, n in enumerate(names):
        r = m[i, i] / m[i].sum() if m[i].sum() else float("nan")
        lines.append(f"{n:>{w}}  " + " ".join(f"{v:6d}" for v in m[i]) + f"  {r:6.0%}")
    return "\n".join(lines)


def run(recordings: Path, out: Path, extra: list[tuple[str, Path]] = (), mix: bool = False) -> None:
    import torch

    from earshot_arrays.classify import Backbone

    data = collect(recordings, extra, mix)
    names = sorted({label for _, label, _, _ in data})
    if len(names) < 2:
        raise SystemExit(f"need at least two labels in {recordings}, found {names}")
    bb = Backbone()
    rows = []  # (source, label index, is_test, features)
    for src, label, wins, fs in data:
        n_test = max(1, round(TEST_SHARE * len(wins))) if len(wins) > 3 else 0
        for i, w in enumerate(wins):
            rows.append((src, names.index(label), i >= len(wins) - n_test, bb(w, fs)[0]))
        print(f"{src:28s} {label:8s} {len(wins):3d} windows")
    X = np.stack([r[3] for r in rows])
    y = np.array([r[1] for r in rows])
    test = np.array([r[2] for r in rows])
    # hold-out check: train on the first part of every recording, test on its end
    net, mean, std = fit(X[~test], y[~test], len(names))
    pred = predict(net, mean, std, X[test]).argmax(axis=1)
    print(f"\nheld-out windows (last {TEST_SHARE:.0%} of each recording), rows = truth:")
    print(confusion(y[test], pred, names))
    for src in dict.fromkeys(r[0] for r in rows):
        sel = np.array([r[0] == src for r in rows]) & test
        if sel.any():
            got = [names[i] for i in pred[sel[test]]]
            print(f"  {src:28s} -> {', '.join(f'{n} {got.count(n)}' for n in dict.fromkeys(got))}")
    # final head on everything
    net, mean, std = fit(X, y, len(names))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fine": names, "mean": mean, "std": std, "hidden": HIDDEN, "state": net.state_dict(),
                "input": "mix" if mix else "beam"}, out)
    print(f"\nroom head ({', '.join(names)}) trained on {len(y)} windows -> {out}")
