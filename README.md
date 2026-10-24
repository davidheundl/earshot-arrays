# Earshot Arrays

Two microphone arrays on one laptop: each array gives a bearing, the bearings cross at the sound source, and a classifier says whether it is a drone. The live part of the Earshot demo; the swarm simulation lives in the `earshot` repo.

```bash
uv sync
uv run arrays list
cp setup.example.yaml setup.yaml      # fill in devices and poses
uv run arrays meter --profile profiles/respeaker_usb_v2.yaml --device 3   # tap each mic
uv run arrays bearing --setup setup.yaml --array A
uv run arrays locate                  # live plot
uv run arrays record --seconds 30     # → recordings/<timestamp>/, replays with --replay

uv run arrays classify --profile profiles/uma16.yaml --device UMA16   # live class + bearing
uv run arrays live                    # live map in the browser: direction, 360° scan, class (+ fix with 2 arrays)
uv run arrays live --replay recordings/demo_room1   # rehearse the map on a recording (real time; --fast)
# classifier weights (not in git): copy mn10_as_mAP_471.pt and sound_head.pt from earshot's data/models/ to data/models/

# rehearse without hardware
uv run arrays synth --out recordings/synthetic
uv run arrays locate --setup recordings/synthetic/setup.yaml --replay recordings/synthetic --realtime

# office twin: probability grid over the office map, works with 1, 2 or more arrays
# office.yaml: the demo office from plans/office_sketch.png; its `arrays` marks move the setup's arrays
# (rotation stays from setup.yaml). office.example.yaml is a placeholder to copy for another room.
uv run arrays twin --setup setup.yaml # live; --office FILE, --cell 0.25, --no-plot, --save-plot PNG
uv run arrays synth --office office.example.yaml --path 5,6 6.5,7 8,6.5 9.5,5.5 --out recordings/office_synth
uv run arrays twin --setup recordings/office_synth/setup.yaml --office office.example.yaml \
  --replay recordings/office_synth --realtime
```

See [CLAUDE.md](CLAUDE.md) for the pipeline, the planned modules and the setup checklist for the day.
