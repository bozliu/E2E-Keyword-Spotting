from __future__ import annotations

import queue
import threading
import time

import numpy as np
import torch

from kws.constants import CLIP_SAMPLES, COMMAND31_LABELS
from kws.demo.realtime import AdaptiveGateConfig, DemoSnapshot, GateStateMachine, InferenceWorker, MIC_RUNTIME_ERROR
from kws.models.common import DualTaskOutput


class _FakeFrontend:
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return torch.zeros(80, 126, dtype=torch.float32)


class _FakeModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> DualTaskOutput:  # noqa: ARG002
        command_logits = torch.zeros(1, len(COMMAND31_LABELS), dtype=torch.float32)
        command_logits[0, COMMAND31_LABELS.index("yes")] = 10.0
        wake_logits = torch.tensor([10.0], dtype=torch.float32)
        embedding = torch.zeros(1, 8, dtype=torch.float32)
        return DualTaskOutput(command_logits=command_logits, wake_logits=wake_logits, embedding=embedding)


class _ExplodingModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> DualTaskOutput:  # noqa: ARG002
        raise RuntimeError("boom")


def test_inference_worker_emits_snapshot() -> None:
    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=16)
    stop = threading.Event()
    lock = threading.Lock()
    snapshots: list[DemoSnapshot | None] = [None]

    gate = GateStateMachine(
        mode="fixed",
        open_threshold=0.5,
        close_threshold=0.4,
        cmd_conf_threshold=0.3,
        hold_seconds=0.2,
        adaptive=AdaptiveGateConfig(calibration_seconds=0.0),
    )

    worker = InferenceWorker(
        q=q,
        stop=stop,
        lock=lock,
        snapshot_ref=snapshots,
        model=_FakeModel(),
        frontend=_FakeFrontend(),
        device=torch.device("cpu"),
        command31_labels=COMMAND31_LABELS,
        wheel="command31",
        gate=gate,
        hop_seconds=0.0,
        ema_alpha=0.3,
        hold_ms=200.0,
        selected_device_label="auto->cpu",
        input_device_name="Test Mic",
        stream_sample_rate=16000.0,
        model_sample_rate=16000,
        audio_seconds=1.0,
        mic_precheck_seconds=0.0,
        mic_min_rms=0.0,
        auto_gain=True,
        target_rms=0.05,
        max_gain_db=18.0,
        display_conf_thr=0.0,
        display_wake_thr=0.0,
        vote_window=1,
        vote_min_count=1,
        reset_precheck_event=threading.Event(),
        passive_profile=None,
        keyword_calibration=None,
    )
    assert worker.engine.segment_runtime_enabled is False
    assert worker.engine.segment_decoder is None
    assert worker.engine.segment_decoder_disabled is True
    assert worker.engine.realtime_specialist is None
    worker.start()

    q.put(np.full(CLIP_SAMPLES, 0.01, dtype=np.float32))

    deadline = time.time() + 2.0
    snap = None
    while time.time() < deadline:
        with lock:
            snap = snapshots[0]
        if snap is not None:
            break
        time.sleep(0.02)

    stop.set()
    worker.join(timeout=2.0)

    assert snap is not None
    assert snap.gate_open is True
    assert snap.gate_state in {"open", "hold"}
    assert snap.command_label == "yes"
    assert snap.active_label == "yes"
    assert snap.highlight_label == "yes"
    assert snap.command_conf > 0.0
    assert snap.selected_device == "auto->cpu"
    assert snap.prompt_status in {"MATCH", "LISTENING"}
    assert snap.precheck_passed is True
    assert snap.input_device_name == "Test Mic"
    assert snap.stream_sample_rate == 16000.0


def test_inference_worker_publishes_runtime_error_snapshot() -> None:
    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=16)
    stop = threading.Event()
    lock = threading.Lock()
    snapshots: list[DemoSnapshot | None] = [None]

    gate = GateStateMachine(
        mode="fixed",
        open_threshold=0.5,
        close_threshold=0.4,
        cmd_conf_threshold=0.3,
        hold_seconds=0.2,
        adaptive=AdaptiveGateConfig(calibration_seconds=0.0),
    )

    worker = InferenceWorker(
        q=q,
        stop=stop,
        lock=lock,
        snapshot_ref=snapshots,
        model=_ExplodingModel(),
        frontend=_FakeFrontend(),
        device=torch.device("cpu"),
        command31_labels=COMMAND31_LABELS,
        wheel="command31",
        gate=gate,
        hop_seconds=0.0,
        ema_alpha=0.3,
        hold_ms=200.0,
        selected_device_label="auto->cpu",
        input_device_name="Test Mic",
        stream_sample_rate=16000.0,
        model_sample_rate=16000,
        audio_seconds=1.0,
        mic_precheck_seconds=0.0,
        mic_min_rms=0.0,
        auto_gain=True,
        target_rms=0.05,
        max_gain_db=18.0,
        display_conf_thr=0.0,
        display_wake_thr=0.0,
        vote_window=1,
        vote_min_count=1,
        reset_precheck_event=threading.Event(),
        passive_profile=None,
        keyword_calibration=None,
    )
    worker.start()
    q.put(np.full(CLIP_SAMPLES, 0.01, dtype=np.float32))

    deadline = time.time() + 2.0
    snap = None
    while time.time() < deadline:
        with lock:
            snap = snapshots[0]
        if snap is not None and snap.prompt_status == MIC_RUNTIME_ERROR:
            break
        time.sleep(0.02)

    worker.join(timeout=2.0)

    assert snap is not None
    assert stop.is_set() is True
    assert snap.prompt_status == MIC_RUNTIME_ERROR
    assert snap.display_label == "ERROR"
    assert snap.highlight_label is None
    assert "boom" in snap.error_message
