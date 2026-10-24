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
# classifier weights (not in git): copy mn10_as_mAP_471.pt and sound_head.pt from earshot's data/models/ to data/models/

# rehearse without hardware
uv run arrays synth --out recordings/synthetic
uv run arrays locate --setup recordings/synthetic/setup.yaml --replay recordings/synthetic --realtime
```

See [CLAUDE.md](CLAUDE.md) for the pipeline, the planned modules and the setup checklist for the day.
