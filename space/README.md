---
title: Public Keyword Spotting Demo
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
short_description: Free CPU browser-mic KWS demo; separate from local v3.
---

# Public Keyword Spotting Demo

This Hugging Face Space exposes a free browser-microphone keyword spotting demo for public testing.

It is intentionally **not** the same operating mode as the local `v3` desktop demo:

- local flagship demo: `accuracy-first`, Apple `MPS`, external `AST + SUPERB` ensemble
- this hosted demo: browser mic, hosted `CPU`, public lightweight checkpoint

## What this demo does
- lets users stream speech continuously from the browser microphone
- runs CPU inference with the current public lightweight checkpoint
- shows live listening / match state, confidence, and hard-word confusion hints
- visualizes the keyword wheel used in the desktop demo

## Important notes
- This Space is the free public browser path, not the highest-accuracy local `v3` path.
- This hosted demo follows the local realtime flow more closely, but still runs inside a browser rather than the desktop `sounddevice` loop.
- Cloud CPU latency may differ from local Apple Silicon latency.
- Hard words such as `left`, `on`, and `down` use stricter guardrails to reduce confusion.

## Model source
The app downloads public checkpoints from the GitHub release assets of the main repository instead of bundling weights into the Space.

Repository: [bozliu/E2E-Keyword-Spotting](https://github.com/bozliu/E2E-Keyword-Spotting)
