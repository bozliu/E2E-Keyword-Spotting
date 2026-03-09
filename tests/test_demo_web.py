from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
import wave

import numpy as np
import torch

from kws.constants import KWS12_LABELS
from kws.demo import web
from kws.demo.web_runtime import HighlightPreviewState, TemporalLabelSmoother, resolve_runtime_decision


class _DummyModel(torch.nn.Module):
    def forward(self, x):  # pragma: no cover - should not be called in no-speech test
        raise AssertionError("model inference should not run for no-speech clips")


class _StaticWebModel(torch.nn.Module):
    def __init__(self, command_logits: list[float], wake_logit: float) -> None:
        super().__init__()
        self._command_logits = torch.tensor([command_logits], dtype=torch.float32)
        self._wake_logit = torch.tensor([wake_logit], dtype=torch.float32)

    def forward(self, x):
        return type(
            "WebOut",
            (),
            {
                "command_logits": self._command_logits,
                "wake_logits": self._wake_logit,
                "embedding": torch.zeros((1, 4), dtype=torch.float32),
            },
        )()


class _BoomWebModel(torch.nn.Module):
    def forward(self, x):
        raise RuntimeError("boom from test model")


class _IdentityFrontend:
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        base = waveform[: min(64, waveform.numel())].float()
        if base.numel() == 0:
            base = torch.zeros(64, dtype=torch.float32)
        return base.unsqueeze(0).repeat(8, 1)


def test_normalize_audio_input_downmixes_and_scales_int16() -> None:
    sample_rate = 16_000
    stereo = np.array([[1000, -1000], [2000, -2000], [0, 0]], dtype=np.int16)
    sr, waveform = web._normalize_audio_input((sample_rate, stereo))
    assert sr == sample_rate
    assert waveform.ndim == 1
    assert waveform.dtype == np.float32
    assert np.max(np.abs(waveform)) <= 1.0


def test_normalize_audio_input_accepts_file_payload(tmp_path: Path) -> None:
    sample_rate = 16_000
    t = np.linspace(0.0, 0.10, int(sample_rate * 0.10), endpoint=False)
    waveform = 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
    wav_path = tmp_path / "speech.wav"
    pcm = np.clip(waveform * np.iinfo(np.int16).max, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)

    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())

    sr, loaded = web._normalize_audio_input({"path": str(wav_path)})
    assert sr == sample_rate
    assert loaded.dtype == np.float32
    assert loaded.ndim == 1
    assert loaded.size == waveform.size


def test_public_checkpoint_urls_prefer_canonical_release_tag() -> None:
    urls = web._checkpoint_urls("demo_mhatt_small_focus")
    assert urls == ["https://github.com/bozliu/E2E-Keyword-Spotting/releases/download/v3.0.0/best_kws12.pt"]


def test_predict_web_clip_returns_friendly_unknown_for_no_signal() -> None:
    bundle = web.LoadedWebDemo(
        checkpoint_path=Path("dummy.pt"),
        runtime_device=torch.device("cpu"),
        checkpoint_name="dummy",
        model=_DummyModel(),
        frontend=None,
        command31_labels=["silence", "yes", "no"],
        wheel="kws12",
        keyword_calibration={},
        sample_rate=16_000,
        clip_samples=16_000,
        verifier=None,
        display_conf_thr=0.35,
        display_wake_thr=0.45,
        default_vote_window=4,
        default_vote_min_count=2,
    )
    audio = (16_000, np.zeros(16_000, dtype=np.float32))
    result = web.predict_web_clip(bundle, audio)
    assert result.label == "UNKNOWN"
    assert result.gate_open is False
    assert "No clear command detected" in result.status_message
    assert set(result.keyword_scores) == set(KWS12_LABELS)


def test_new_stream_state_allows_preview_before_full_buffer() -> None:
    bundle = web.LoadedWebDemo(
        checkpoint_path=Path("dummy.pt"),
        runtime_device=torch.device("cpu"),
        checkpoint_name="dummy",
        model=_DummyModel(),
        frontend=None,
        command31_labels=["silence", "yes", "no"],
        wheel="kws12",
        keyword_calibration={},
        sample_rate=16_000,
        clip_samples=16_000,
        verifier=None,
        display_conf_thr=0.35,
        display_wake_thr=0.45,
        default_vote_window=4,
        default_vote_min_count=2,
    )

    state = web.PublicBrowserDemo()._new_stream_state(bundle, sample_rate=48_000)

    assert state.buffer.capacity == 48_000
    assert state.min_buffer_samples == int(round(48_000 * web.WEB_MIN_BUFFER_SECONDS))
    assert state.min_buffer_samples < state.buffer.capacity


def test_stream_emits_preview_before_precheck_passes(monkeypatch) -> None:
    bundle = web.LoadedWebDemo(
        checkpoint_path=Path("dummy.pt"),
        runtime_device=torch.device("cpu"),
        checkpoint_name="dummy",
        model=_StaticWebModel(command_logits=[-2.0, 4.0, -2.0], wake_logit=-2.2),
        frontend=_IdentityFrontend(),
        command31_labels=["silence", "yes", "no"],
        wheel="kws12",
        keyword_calibration={},
        sample_rate=16_000,
        clip_samples=16_000,
        verifier=None,
        display_conf_thr=0.35,
        display_wake_thr=0.45,
        default_vote_window=4,
        default_vote_min_count=2,
    )
    demo = web.PublicBrowserDemo()
    monkeypatch.setattr(demo, "_bundle", lambda: bundle)

    clock = iter([0.0, 0.15, 0.30, 0.45])
    monkeypatch.setattr(web.time, "monotonic", lambda: next(clock))

    chunk = (16_000, (0.02 * np.sin(np.linspace(0.0, 12.0 * np.pi, 4000, endpoint=False))).astype(np.float32))
    _summary1, status1, _conf1, _fig1, state = demo.stream(chunk, None)
    summary2, status2, _conf2, _fig2, _state2 = demo.stream(chunk, state)

    assert "Mic state: MIC_CHECK" in status1
    assert "Preview: yes" in status1
    assert "Prompt: LISTENING" in summary2
    assert "Preview: yes" in status2
    assert "Preview reason: preview-ok" in status2


def test_stream_accepts_filepath_audio_payload(monkeypatch, tmp_path: Path) -> None:
    bundle = web.LoadedWebDemo(
        checkpoint_path=Path("dummy.pt"),
        runtime_device=torch.device("cpu"),
        checkpoint_name="dummy",
        model=_StaticWebModel(command_logits=[-2.0, 4.0, -2.0], wake_logit=-2.2),
        frontend=_IdentityFrontend(),
        command31_labels=["silence", "yes", "no"],
        wheel="kws12",
        keyword_calibration={},
        sample_rate=16_000,
        clip_samples=16_000,
        verifier=None,
        display_conf_thr=0.35,
        display_wake_thr=0.45,
        default_vote_window=4,
        default_vote_min_count=2,
    )
    demo = web.PublicBrowserDemo()
    monkeypatch.setattr(demo, "_bundle", lambda: bundle)

    wav_path = tmp_path / "chunk.wav"
    pcm = (0.25 * np.sin(np.linspace(0.0, 20.0 * np.pi, 4000, endpoint=False)) * np.iinfo(np.int16).max).astype(np.int16)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(pcm.tobytes())

    clock = iter([0.0, 0.15, 0.30, 0.45])
    monkeypatch.setattr(web.time, "monotonic", lambda: next(clock))

    _summary1, _status1, _conf1, _fig1, state = demo.stream({"path": str(wav_path)}, None)
    summary2, status2, _conf2, figure2, _state2 = demo.stream({"path": str(wav_path)}, state)

    assert "Prompt: LISTENING" in summary2
    assert "Preview: yes" in status2
    assert figure2 is not None


def test_stream_reports_runtime_failures_without_raising(monkeypatch) -> None:
    bundle = web.LoadedWebDemo(
        checkpoint_path=Path("dummy.pt"),
        runtime_device=torch.device("cpu"),
        checkpoint_name="dummy",
        model=_BoomWebModel(),
        frontend=_IdentityFrontend(),
        command31_labels=["silence", "yes", "no"],
        wheel="kws12",
        keyword_calibration={},
        sample_rate=16_000,
        clip_samples=16_000,
        verifier=None,
        display_conf_thr=0.35,
        display_wake_thr=0.45,
        default_vote_window=4,
        default_vote_min_count=2,
    )
    demo = web.PublicBrowserDemo()
    monkeypatch.setattr(demo, "_bundle", lambda: bundle)

    clock = iter([0.0, 0.20])
    monkeypatch.setattr(web.time, "monotonic", lambda: next(clock))

    chunk = (16_000, (0.02 * np.sin(np.linspace(0.0, 12.0 * np.pi, 4000, endpoint=False))).astype(np.float32))
    summary, status, _conf, _fig, _state = demo.stream(chunk, None)

    assert "Prompt: ERROR" in summary
    assert "Failure stage: model-inference" in status
    assert "RuntimeError: boom from test model" in status


def test_debug_stream_file_reuses_live_runtime_and_returns_trace(monkeypatch, tmp_path: Path) -> None:
    bundle = web.LoadedWebDemo(
        checkpoint_path=Path("dummy.pt"),
        runtime_device=torch.device("cpu"),
        checkpoint_name="dummy",
        model=_StaticWebModel(command_logits=[-2.0, 4.0, -2.0], wake_logit=-2.2),
        frontend=_IdentityFrontend(),
        command31_labels=["silence", "yes", "no"],
        wheel="kws12",
        keyword_calibration={},
        sample_rate=16_000,
        clip_samples=16_000,
        verifier=None,
        display_conf_thr=0.35,
        display_wake_thr=0.45,
        default_vote_window=4,
        default_vote_min_count=2,
    )
    demo = web.PublicBrowserDemo()
    monkeypatch.setattr(demo, "_bundle", lambda: bundle)

    wav_path = tmp_path / "debug.wav"
    pcm = (0.25 * np.sin(np.linspace(0.0, 20.0 * np.pi, 6400, endpoint=False)) * np.iinfo(np.int16).max).astype(np.int16)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(pcm.tobytes())

    summary, status, _conf, figure, trace = demo.debug_stream_file(str(wav_path))

    assert "Prompt: LISTENING" in summary
    assert "Preview: yes" in status
    assert figure is not None
    assert "preview=yes" in trace


def test_format_top_confusions_lists_scores() -> None:
    result = web.WebDemoResult(
        label="ON",
        confidence=0.91,
        wake_prob=0.88,
        gate_open=True,
        status_message="ok",
        keyword_scores={label: 0.0 for label in KWS12_LABELS},
        top_confusions=[{"label": "off", "score": 0.31}, {"label": "one", "score": 0.11}],
        latency_ms=5.0,
        wheel_active_label="on",
    )
    rendered = web.format_top_confusions(result)
    assert "off" in rendered
    assert "0.310" in rendered


def test_build_space_bundle_stages_required_files(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "deploy_hf_space.py"
    spec = importlib.util.spec_from_file_location("deploy_hf_space", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    module.build_space_bundle(tmp_path)

    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "app.py").exists()
    assert (tmp_path / "requirements.txt").exists()
    assert (tmp_path / "src" / "kws" / "demo" / "web.py").exists()
    assert not list((tmp_path / "src").rglob("__pycache__"))
    assert not list((tmp_path / "src").rglob("*.pyc"))
    assert not list((tmp_path / "src").rglob("*.egg-info"))


def test_create_gradio_app_does_not_load_model_on_construction(monkeypatch) -> None:
    calls = {"count": 0}

    def _boom(_checkpoint: str = "auto"):
        calls["count"] += 1
        raise AssertionError("model bundle should not load during app construction")

    monkeypatch.setattr(web, "load_cached_web_demo", _boom)
    app = web.create_gradio_app(checkpoint="auto")
    assert app is not None
    assert calls["count"] == 0


def test_create_gradio_app_only_exposes_microphone_input() -> None:
    app = web.create_gradio_app(checkpoint="auto")
    components = app.config.get("components", [])
    audio_components = [component for component in components if component.get("type") == "audio"]
    assert len(audio_components) == 1
    assert audio_components[0]["props"]["sources"] == ["microphone"]
    assert audio_components[0]["props"]["streaming"] is True


def test_create_gradio_app_exposes_debug_stream_file_api() -> None:
    app = web.create_gradio_app(checkpoint="auto")
    dependencies = app.config.get("dependencies", [])
    api_names = {
        dep.get("api_name")
        for dep in dependencies
        if dep.get("api_name") is not False and dep.get("api_name") is not None
    }
    assert "debug_stream_file" in api_names


def test_resolve_checkpoint_for_web_uses_public_fallback(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_ensure(name: str, cache_dir=web.DEFAULT_SPACE_CACHE_DIR):
        calls.append(name)
        if name == web.DEFAULT_PUBLIC_CHECKPOINT:
            raise RuntimeError("primary missing")
        return Path(f"/tmp/{name}.pt")

    monkeypatch.setattr(web, "ensure_public_checkpoint", _fake_ensure)

    path, name = web._resolve_checkpoint_for_web("auto")

    assert path == Path(f"/tmp/{web.DEFAULT_FALLBACK_PUBLIC_CHECKPOINT}.pt")
    assert name == web.DEFAULT_FALLBACK_PUBLIC_CHECKPOINT
    assert calls == [web.DEFAULT_PUBLIC_CHECKPOINT, web.DEFAULT_FALLBACK_PUBLIC_CHECKPOINT]


def test_resolve_runtime_decision_keeps_listening_when_gate_closed() -> None:
    smoother = TemporalLabelSmoother(window_size=4, min_count=2, hold_seconds=0.3)
    preview = HighlightPreviewState()
    probs = np.array([0.05, 0.90, 0.05], dtype=np.float32)

    decision = resolve_runtime_decision(
        now=1.0,
        wheel="command31",
        command31_labels=["silence", "yes", "no"],
        command_probs=probs,
        gate_open=False,
        gate_state="closed",
        wake_prob=0.20,
        display_wake_thr=0.45,
        calibration={},
        default_conf_thr=0.35,
        default_vote_window=4,
        default_vote_min_count=2,
        smoother=smoother,
        preview=preview,
    )

    assert decision.prompt_label == "LISTENING"
    assert decision.final_label == "LISTENING"
    assert decision.active_label is None


def test_resolve_runtime_decision_emits_preview_when_gate_closed_for_web_mode() -> None:
    smoother = TemporalLabelSmoother(window_size=4, min_count=2, hold_seconds=0.3)
    preview = HighlightPreviewState()
    probs = np.array([0.05, 0.90, 0.05], dtype=np.float32)

    decision = resolve_runtime_decision(
        now=1.0,
        wheel="command31",
        command31_labels=["silence", "yes", "no"],
        command_probs=probs,
        gate_open=False,
        gate_state="closed",
        wake_prob=0.35,
        display_wake_thr=0.45,
        calibration={},
        default_conf_thr=0.35,
        default_vote_window=4,
        default_vote_min_count=2,
        smoother=smoother,
        preview=preview,
        preview_requires_gate=False,
        preview_min_command_conf=0.15,
        preview_min_wake_prob=0.10,
    )

    assert decision.prompt_label == "LISTENING"
    assert decision.final_label == "LISTENING"
    assert decision.active_label is None
    assert decision.preview_label == "yes"
    assert decision.highlight_label == "yes"
    assert decision.preview_reason == "preview-ok"


def test_resolve_runtime_decision_reports_preview_rejection_reason() -> None:
    smoother = TemporalLabelSmoother(window_size=4, min_count=2, hold_seconds=0.3)
    preview = HighlightPreviewState()
    probs = np.array([0.05, 0.90, 0.05], dtype=np.float32)

    decision = resolve_runtime_decision(
        now=1.0,
        wheel="command31",
        command31_labels=["silence", "yes", "no"],
        command_probs=probs,
        gate_open=False,
        gate_state="closed",
        wake_prob=0.05,
        display_wake_thr=0.45,
        calibration={},
        default_conf_thr=0.35,
        default_vote_window=4,
        default_vote_min_count=2,
        smoother=smoother,
        preview=preview,
        preview_requires_gate=False,
        preview_min_command_conf=0.15,
        preview_min_wake_prob=0.10,
    )

    assert decision.preview_label is None
    assert decision.highlight_label is None
    assert decision.preview_reason == "wake-low"


def test_resolve_runtime_decision_promotes_match_after_repeated_frames() -> None:
    smoother = TemporalLabelSmoother(window_size=4, min_count=2, hold_seconds=0.3)
    preview = HighlightPreviewState()
    probs = np.array([0.05, 0.92, 0.03], dtype=np.float32)

    first = resolve_runtime_decision(
        now=1.0,
        wheel="command31",
        command31_labels=["silence", "yes", "no"],
        command_probs=probs,
        gate_open=True,
        gate_state="open",
        wake_prob=0.90,
        display_wake_thr=0.45,
        calibration={},
        default_conf_thr=0.35,
        default_vote_window=4,
        default_vote_min_count=2,
        smoother=smoother,
        preview=preview,
    )
    second = resolve_runtime_decision(
        now=1.1,
        wheel="command31",
        command31_labels=["silence", "yes", "no"],
        command_probs=probs,
        gate_open=True,
        gate_state="open",
        wake_prob=0.91,
        display_wake_thr=0.45,
        calibration={},
        default_conf_thr=0.35,
        default_vote_window=4,
        default_vote_min_count=2,
        smoother=smoother,
        preview=preview,
    )

    assert first.active_label is None
    assert first.final_label == "YES"
    assert first.highlight_label == "yes"
    assert second.active_label == "yes"
    assert second.prompt_label == "MATCH"
    assert second.final_label == "YES"


def test_app_import_succeeds_with_proxy_environment(monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:10818")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:10818")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:10818")
    sys.modules.pop("kws.demo.rank_checkpoints", None)
    sys.modules.pop("kws.train.engine", None)

    start = time.perf_counter()
    spec = importlib.util.spec_from_file_location(
        "public_demo_app",
        Path(__file__).resolve().parents[1] / "app.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    elapsed = time.perf_counter() - start

    assert hasattr(module, "app")
    assert elapsed < 5.0
    assert "kws.demo.rank_checkpoints" not in sys.modules
    assert "kws.train.engine" not in sys.modules
