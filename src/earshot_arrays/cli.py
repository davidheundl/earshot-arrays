"""`arrays list`: audio inputs with at least two channels (more commands to come, see CLAUDE.md)."""

from __future__ import annotations

import argparse


def cmd_list(_a) -> None:
    import sounddevice as sd

    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] >= 2:
            print(f"{i:3d}  {d['max_input_channels']:2d} ch  {d['default_samplerate']:7.0f} Hz  {d['name']}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="arrays")
    sub = ap.add_subparsers(required=True)
    sub.add_parser("list", help="audio inputs with 2+ channels").set_defaults(fn=cmd_list)
    a = ap.parse_args()
    a.fn(a)
