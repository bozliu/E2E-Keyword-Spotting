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
short_description: Public browser-mic demo for lightweight keyword spotting.
---

# Public Keyword Spotting Demo

This Hugging Face Space exposes the latest lightweight keyword spotting model through a browser microphone interface.

## What this demo does
- lets users stream speech continuously from the browser microphone
- runs CPU inference with the current public lightweight checkpoint
- shows live listening / match state, confidence, and hard-word confusion hints
- visualizes the keyword wheel used in the desktop demo

## Important notes
- This hosted demo follows the local realtime flow more closely, but still runs inside a browser rather than the desktop `sounddevice` loop.
- Cloud CPU latency may differ from local Apple Silicon latency.
- Hard words such as `left`, `on`, and `down` use stricter guardrails to reduce confusion.

## Model source
The app downloads public checkpoints from the GitHub release assets of the main repository instead of bundling weights into the Space.

Repository: [bozliu/E2E-Keyword-Spotting](https://github.com/bozliu/E2E-Keyword-Spotting)
