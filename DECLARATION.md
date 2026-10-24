# Pre-built declaration

Everything below existed before the hackathon day (Oct 24, 2026). Everything else was built on the day. Add a line whenever pre-event work lands.

- Repository skeleton: project file, README, CLAUDE.md with the planned pipeline (Oct 23).
- Pipeline before the hardware arrives (Oct 23): array profiles + `respeaker_usb_v2.yaml` example, setup YAML, devices/meter/replay/record, SRP-PHAT bearing (`doa.py`, copied from the pre-built `earshot` repo's `earshot/audio/doa.py`, same compass convention), two-ray intersection with uncertainty (`locate.py`), synthetic two-array sessions (`synth.py`), a stand-in classifier that labels itself as such, the live top-down plot, the `arrays` CLI and tests.
- Sound classifier (Oct 24, on the day, from pre-built parts): the EfficientAT wrapper and vendored model code (`classify.py`, `efficientat/`, MIT) are copied from the pre-built `earshot` repo (`earshot/audio/classifier.py`, `earshot/audio/efficientat/`), together with its weights `mn10_as_mAP_471.pt` (EfficientAT release) and `sound_head.pt` (head trained before the event on ESC-50 plus drone recordings). Not committed; copied into `data/models/`.
