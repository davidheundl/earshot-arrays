# Earshot Arrays

Hackathon side project (Oct 24, 2026): two real microphone arrays on one laptop locate a sound source in the room and say whether it is a drone. It is the "we can do this with real hardware today" part of the demo; the big picture (the perching swarm over Munich Airport) is the separate `earshot` repo and its simulation. Two-person team: David and Nikolas.

The demo in one sentence: two arrays on the table each give a bearing, the bearings cross at the source, a sound classifier says "drone" (and not plane, car, person, fan), and a live top-down plot shows the position and the class.

## Scope

- Standalone. Do not import from `earshot`, do not share its config or scenarios. Copying a pure algorithm from there (e.g. SRP-PHAT in `earshot/audio/doa.py`, the EfficientAT wrapper in `earshot/audio/classifier.py`) is fine; say so in the commit and in DECLARATION.md.
- No swarm, no radio, no agents, no simulator. Room coordinates in metres, nothing else.
- Hardware is unknown until the morning. Nothing may assume a specific array: geometry, channel map, sample rate and channel count come from an array profile, never from constants in the code.

## Pipeline

```
array A ─┐  sounddevice, 1 s blocks, native rate
array B ─┤
         ├─ per array: channel map → SRP-PHAT bearing (+ peak power) → room frame via the array's pose
         ├─ both:     intersect the two bearing rays → position + uncertainty (reject near-parallel rays)
         ├─ classifier on one beam / the mix → class probabilities
         └─ live plot: table outline, both arrays, both rays, the fix, the class label
```

Planned layout (create as needed, keep it small):

- `profiles/*.yaml`: one per array model. `n_channels`, `fs`, `mic_channels` (raw mics in order), mic positions in metres in the array's own frame (x right, y forward), or `circle: {n, radius, start_deg}`.
- `src/earshot_arrays/devices.py`: list inputs with ≥ 2 channels; open a stream from a profile.
- `src/earshot_arrays/meter.py`: live per-channel level bars. Used to find the channel order (tap each mic).
- `src/earshot_arrays/doa.py`: pure: `(n_mics, n_samples) block, fs, mic_xy → bearing, power, scan`.
- `src/earshot_arrays/locate.py`: pure: two array poses + two bearings → position, or None.
- `src/earshot_arrays/classify.py`: EfficientAT if torch and weights load; else a stand-in that says so on screen. A run never stops because a model is missing.
- `src/earshot_arrays/replay.py`: feed a multichannel WAV in place of a device, so everything can be rehearsed without hardware.
- `src/earshot_arrays/belief.py`: pure: probability grid over the floor; each array's bearing multiplies in a narrow bump (3 deg) at every cell's exact bearing, one array gives a fan, two or more a blob.
- `src/earshot_arrays/office.py`: `office.yaml` (floor, walls, furniture, the arrays' marks, optional floor-plan image) → `Office`.
- `src/earshot_arrays/twin.py`: the 2D office twin plot: map, arrays, rays, probability grid, estimate, class.
- `src/earshot_arrays/cli.py`: `arrays list | meter | bearing | locate | twin | record`.

## Rules

- Pure functions for the maths (`doa`, `locate`), tested with synthetic delayed signals; devices only at the edges.
- Every live command also accepts `--replay file.wav`, and `record` writes exactly that format, so a session on the day can be replayed later.
- Bearings: compass-style, clockwise from the room's +y ("north" = a chosen table edge, written down at setup). An array's pose is `(x, y, rotation_deg)`, where `rotation_deg` is the room bearing its forward axis points to.
- Both arrays on one laptop (USB hub if needed); each gets its own stream. Clock drift between them does not matter, because each bearing comes from one array only.
- Anything built before Oct 24 gets a line in DECLARATION.md.

## First hour with the hardware

1. Which model? Check it is a USB audio device (a Raspberry Pi HAT, e.g. ReSpeaker 4/6-mic HAT or Matrix Voice, is not; then the Pi records and streams to the laptop). Find its datasheet: mic positions, channel layout, firmware (ReSpeaker USB v2.0 needs the 6-channel firmware for raw mics).
2. `arrays list`: device index and channel count. Write a profile.
3. `arrays meter`: tap each mic and fix `mic_channels` so the order matches the geometry.
4. `arrays bearing`: clap from the "north" edge and adjust `rotation_deg` until it reads about 0; check 90 too.
5. Put both arrays as far apart as the table allows (bearings cross badly when the source is near the line between them), measure their poses, then `arrays locate`.
6. Record a few sessions with `record` (drone, phone playing a plane, speech) for rehearsal and as a fallback if the hardware fails during the pitch.

## When the second array arrives

Two UMA-16s, 4 m apart, both facing the demo area: `setup.pair.example.yaml` has the poses and the checks.

1. `arrays list`, then `arrays meter --profile profiles/uma16.yaml --device N` per board: which index is A, which B. Write both into `setup.yaml`.
2. Same board edge toward the demo area on both. Per board `arrays bearing --setup setup.yaml --array A` (then B): a clap 2 m in front reads ~0, else fix that board's `rotation_deg`.
3. Pose check with both: a sound at (2, 2) reads A ~45, B ~315.
4. `arrays live`: fix and probability map appear once both arrays are clear. Keep the source 1.5-4 m in front, not on the line through both boards.
5. `arrays record` a few sessions with both, for `--replay` rehearsal.

## Verify

`uv run pytest` and `uv run ruff check src tests`. Before the demo also a short live `arrays locate` with the real arrays, and a `--replay` run of a recorded session.
