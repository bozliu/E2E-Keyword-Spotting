from __future__ import annotations

from collections import deque
import hashlib
import os
import shutil
import tempfile
import time
import traceback
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from kws.constants import KWS12_LABELS
from kws.data.audio import MelFrontend, pad_or_trim
from kws.demo.user_profile import blend_keyword_score
from kws.demo.verifier_runtime import LoadedVerifier, load_runtime_verifier, verify_keyword
from kws.demo.visuals import apply_theme, build_wheel, create_hud, resolve_active_index, update_wheel
from kws.eval.fusion import select_verifier_candidate
from kws.demo.web_runtime import (
    AdaptiveGateConfig,
    AudioRingBuffer,
    GateStateMachine,
    HighlightPreviewState,
    TemporalLabelSmoother,
    accept_display_candidate,
    aggregate_command_probs_to_kws12,
    classify_precheck_signal,
    compute_auto_gain,
    compute_rms,
    ema_update,
    ema_update_scalar,
    get_sensitivity_tuning,
    keyword_runtime_params,
    passes_confusion_guardrail,
    resolve_runtime_decision,
    resolve_display_candidate,
    wheel_labels,
)
from kws.models import create_model
from kws.utils.keyword_focus import DEFAULT_RUNTIME_CONFUSION_GROUPS, load_keyword_calibration

PUBLIC_RELEASE_BASE = "https://github.com/bozliu/E2E-Keyword-Spotting/releases/download/v2.0-public"
PUBLIC_CHECKPOINTS: Dict[str, Dict[str, str]] = {
    "demo_mhatt_small_focus_lod": {
        "filename": "demo_mhatt_small_focus_lod_best_kws12.pt",
        "sha256": "2263a6bab3c0b7d6015d076c094af5b90a1efbb3c18c054e41ed4202b4c9a615",
    },
    "demo_mhatt_small_focus": {
        "filename": "demo_mhatt_small_focus_best_kws12.pt",
        "sha256": "220c61110adf2447884852d879eb05f3fd4df1907bd1bc3485f2d468add0f4f0",
    },
    "quick_mhatt": {
        "filename": "quick_mhatt_best_kws12.pt",
        "sha256": "452b00f1733d8a33333b12a8b2fa412061c3aa6ecdd23f65fd7f4c4960f5160e",
    },
}
DEFAULT_PUBLIC_CHECKPOINT = "demo_mhatt_small_focus"
DEFAULT_PUBLIC_CHECKPOINTS_BY_PROFILE: Dict[str, str] = {
    "stable": "demo_mhatt_small_focus",
    "balanced": "demo_mhatt_small_focus",
    "fast": "demo_mhatt_small_focus_lod",
}
DEFAULT_FALLBACK_PUBLIC_CHECKPOINT = "quick_mhatt"
DEFAULT_SPACE_CACHE_DIR = Path(os.getenv("KWS_SPACE_CACHE_DIR", ".cache/public_checkpoints")).expanduser()
NO_SPEECH_RMS = 0.003
NO_SPEECH_PEAK = 0.02
STREAM_PRECHECK_SECONDS = 1.2
STREAM_MIN_RMS = 0.001
STREAM_TARGET_RMS = 0.05
STREAM_MAX_GAIN_DB = 18.0
STREAM_EMA_ALPHA = 0.35
STREAM_HOP_SECONDS = 0.10
STREAM_HOLD_MS = 300.0
WEB_MIN_BUFFER_SECONDS = 0.20
WEB_PREVIEW_MIN_COMMAND_CONF = 0.12
WEB_PREVIEW_MIN_WAKE_PROB = 0.05
WEB_PREVIEW_HOLD_SECONDS = 0.45
WEB_GATE_CMD_CONF_THR = 0.14
WEB_GATE_CALIBRATION_SECONDS = 1.0
WEB_GATE_OPEN_OFFSET = 0.02
WEB_GATE_CLOSE_OFFSET = 0.01
WEB_GATE_OPEN_FLOOR = 0.28
WEB_GATE_CLOSE_FLOOR = 0.16
WEB_GATE_CALIBRATION_WAKE_CAP = 0.55
WEB_GATE_MAX_OPEN_THR = 0.72
WEB_SPEECH_ESCAPE_MIN_FRAMES = 2
WEB_SPEECH_ESCAPE_RMS = 0.0006
WEB_SPEECH_ESCAPE_PEAK = 0.008
WEB_SPEECH_ESCAPE_BOOSTED_RMS = 0.010
WEB_SPEECH_ESCAPE_BOOSTED_PEAK = 0.050
MIC_CHECK = "MIC_CHECK"
MIC_NO_SIGNAL = "NO_SIGNAL"
MIC_RUNNING = "RUNNING"
MIC_PERMISSION_DENIED = "PERMISSION_DENIED"


@dataclass(frozen=True)
class WebDemoResult:
    label: str
    confidence: float
    wake_prob: float
    gate_open: bool
    status_message: str
    keyword_scores: Dict[str, float]
    top_confusions: list[dict[str, float | str]]
    latency_ms: float
    wheel_active_label: str | None
    preview_label: str | None = None
    preview_reason: str = ""
    raw_command_confidence: float = 0.0
    raw_wake_probability: float = 0.0


@dataclass(frozen=True)
class LoadedWebDemo:
    checkpoint_path: Path
    runtime_device: torch.device
    checkpoint_name: str
    model: torch.nn.Module
    frontend: MelFrontend
    command31_labels: list[str]
    wheel: str
    keyword_calibration: Dict[str, object]
    sample_rate: int
    clip_samples: int
    verifier: LoadedVerifier | None
    display_conf_thr: float
    display_wake_thr: float
    default_vote_window: int
    default_vote_min_count: int


@dataclass
class PublicBrowserStreamState:
    sample_rate: int
    buffer: AudioRingBuffer
    min_buffer_samples: int
    gate: GateStateMachine
    smoother: TemporalLabelSmoother
    preview: HighlightPreviewState
    precheck_started_at: float
    precheck_rms_hist: deque[float] = field(default_factory=lambda: deque(maxlen=256))
    precheck_peak_hist: deque[float] = field(default_factory=lambda: deque(maxlen=256))
    precheck_passed: bool = False
    mic_state: str = MIC_CHECK
    precheck_threshold: float = STREAM_MIN_RMS
    ema_probs: np.ndarray | None = None
    ema_wake: float | None = None
    input_gain_db: float = 0.0
    is_clipping: bool = False
    last_result: WebDemoResult | None = None
    last_infer_at: float = 0.0
    last_gate_state: str = "calibrating"
    last_open_thr: float = 0.0
    last_close_thr: float = 0.0
    speech_like_frames: int = 0
    speech_like_active: bool = False


@dataclass(frozen=True)
class StreamAdvanceStep:
    state: PublicBrowserStreamState
    result: WebDemoResult
    prompt_label: str
    extra_status: str
    trace_line: str


class StreamAdvanceError(RuntimeError):
    def __init__(self, stage: str, exc: Exception) -> None:
        super().__init__(f"{stage}: {type(exc).__name__}: {exc}")
        self.stage = stage
        self.original = exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_url(name: str) -> str:
    meta = PUBLIC_CHECKPOINTS[name]
    return f"{PUBLIC_RELEASE_BASE}/{meta['filename']}"


def ensure_public_checkpoint(name: str = DEFAULT_PUBLIC_CHECKPOINT, cache_dir: str | Path = DEFAULT_SPACE_CACHE_DIR) -> Path:
    if name not in PUBLIC_CHECKPOINTS:
        raise KeyError(f"Unknown public checkpoint: {name}")
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    meta = PUBLIC_CHECKPOINTS[name]
    target = cache_root / meta["filename"]
    if target.exists() and _sha256(target) == meta["sha256"]:
        return target

    url = _checkpoint_url(name)
    fd, raw_tmp = tempfile.mkstemp(prefix=f"{target.stem}.", suffix=".download", dir=str(cache_root))
    os.close(fd)
    tmp_path = Path(raw_tmp)
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        if _sha256(tmp_path) != meta["sha256"]:
            raise RuntimeError(f"Checksum mismatch while downloading {name} from {url}")
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
    return target


def _resolve_checkpoint_for_web(checkpoint: str = "auto", selection_profile: str = "stable") -> tuple[Path, str]:
    requested = str(checkpoint).strip().lower()
    if requested == "auto":
        preferred_name = DEFAULT_PUBLIC_CHECKPOINTS_BY_PROFILE.get(str(selection_profile).lower().strip(), DEFAULT_PUBLIC_CHECKPOINT)
        try:
            return ensure_public_checkpoint(preferred_name), preferred_name
        except Exception:
            return ensure_public_checkpoint(DEFAULT_FALLBACK_PUBLIC_CHECKPOINT), DEFAULT_FALLBACK_PUBLIC_CHECKPOINT

    explicit = Path(checkpoint).expanduser().resolve()
    if explicit.exists():
        return explicit, explicit.stem
    if requested in PUBLIC_CHECKPOINTS:
        return ensure_public_checkpoint(requested), requested
    raise FileNotFoundError(f"Checkpoint not found or unsupported: {checkpoint}")


def _normalize_waveform(sample_rate: int, waveform: np.ndarray | torch.Tensor) -> tuple[int, np.ndarray]:
    waveform = np.asarray(waveform)
    if waveform.ndim == 2:
        if waveform.shape[0] <= 8 and waveform.shape[1] > waveform.shape[0]:
            waveform = waveform.mean(axis=0)
        else:
            waveform = waveform.mean(axis=1)
    waveform = waveform.reshape(-1)
    if waveform.size == 0:
        raise ValueError("Received empty audio clip.")
    if np.issubdtype(waveform.dtype, np.integer):
        max_val = max(1.0, float(np.iinfo(waveform.dtype).max))
        waveform = waveform.astype(np.float32) / max_val
    else:
        waveform = waveform.astype(np.float32)
    np.clip(waveform, -1.0, 1.0, out=waveform)
    return int(sample_rate), waveform


def _audio_path_from_payload(audio: object) -> str | None:
    if isinstance(audio, (str, os.PathLike)):
        return os.fspath(audio)
    if isinstance(audio, dict):
        path = audio.get("path") or audio.get("url")
        return os.fspath(path) if path else None
    path = getattr(audio, "path", None)
    if path:
        return os.fspath(path)
    return None


def _load_audio_path(audio_path: str | os.PathLike[str]) -> tuple[int, np.ndarray]:
    import torchaudio

    waveform, sample_rate = torchaudio.load(os.fspath(audio_path))
    return _normalize_waveform(int(sample_rate), waveform.detach().cpu().numpy())


def _normalize_audio_input(audio: tuple[int, np.ndarray] | list | None | str | os.PathLike[str] | dict | object) -> tuple[int, np.ndarray]:
    if audio is None:
        raise ValueError("No audio received.")

    audio_path = _audio_path_from_payload(audio)
    if audio_path:
        return _load_audio_path(audio_path)

    if not isinstance(audio, (tuple, list)) or len(audio) != 2:
        raise ValueError("Expected Gradio audio input as (sample_rate, waveform) or a filepath/FileData payload.")
    return _normalize_waveform(int(audio[0]), audio[1])


def _resample_waveform(waveform: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return waveform
    return torch.nn.functional.interpolate(
        waveform.view(1, 1, -1),
        size=int(round(waveform.numel() * float(dst_sr) / float(src_sr))),
        mode="linear",
        align_corners=False,
    ).view(-1)


def _compute_rms_and_peak(waveform: np.ndarray) -> tuple[float, float]:
    if waveform.size == 0:
        return 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)) + 1e-12))
    peak = float(np.max(np.abs(waveform)))
    return rms, peak


def _update_speech_activity(
    state: PublicBrowserStreamState,
    *,
    chunk_rms: float,
    chunk_peak: float,
) -> tuple[bool, float, float]:
    gain_lin, _gain_db = compute_auto_gain(chunk_rms, STREAM_TARGET_RMS, STREAM_MAX_GAIN_DB, enabled=True)
    boosted_rms = float(chunk_rms * gain_lin)
    boosted_peak = float(min(1.0, chunk_peak * gain_lin))
    rms_floor = max(WEB_SPEECH_ESCAPE_RMS, state.precheck_threshold * 0.65)
    peak_floor = max(WEB_SPEECH_ESCAPE_PEAK, state.precheck_threshold * 4.0)
    speech_like = bool(
        chunk_rms >= rms_floor
        or chunk_peak >= peak_floor
        or boosted_rms >= WEB_SPEECH_ESCAPE_BOOSTED_RMS
        or boosted_peak >= WEB_SPEECH_ESCAPE_BOOSTED_PEAK
    )
    if speech_like:
        state.speech_like_frames = min(state.speech_like_frames + 1, WEB_SPEECH_ESCAPE_MIN_FRAMES + 4)
    else:
        state.speech_like_frames = max(0, state.speech_like_frames - 1)
    state.speech_like_active = state.speech_like_frames >= WEB_SPEECH_ESCAPE_MIN_FRAMES
    return state.speech_like_active, boosted_rms, boosted_peak


def load_web_demo(checkpoint: str = "auto", wheel: str = "kws12", selection_profile: str = "stable", sensitivity_profile: str = "strict") -> LoadedWebDemo:
    ckpt_path, ckpt_name = _resolve_checkpoint_for_web(checkpoint, selection_profile=selection_profile)
    checkpoint_payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = checkpoint_payload["config"]
    features = cfg["features"]
    command31_labels = list(checkpoint_payload.get("label_set", []))
    if not command31_labels:
        raise RuntimeError(f"Checkpoint at {ckpt_path} is missing label_set.")

    device = torch.device("cpu")
    verifier = load_runtime_verifier(ckpt_path, device=device)
    model = create_model(cfg["model"], n_mels=int(features["n_mels"]), num_commands=len(command31_labels))
    model.load_state_dict(checkpoint_payload["model_state"])
    model.to(device)
    model.eval()

    frontend = MelFrontend(
        sample_rate=int(features["sample_rate"]),
        n_fft=int(features.get("n_fft", 1024)),
        hop_length=int(features.get("hop_length", 128)),
        n_mels=int(features.get("n_mels", 80)),
        f_min=float(features.get("f_min", 20.0)),
        f_max=float(features.get("f_max", 7600.0)),
    )
    keyword_calibration = load_keyword_calibration(ckpt_path.parent / "keyword_calibration.json")
    if not keyword_calibration and isinstance(checkpoint_payload.get("keyword_calibration"), dict):
        keyword_calibration = dict(checkpoint_payload["keyword_calibration"])

    tuning = get_sensitivity_tuning(sensitivity_profile)
    sample_rate = int(features.get("sample_rate", 16000))
    audio_seconds = float(features.get("audio_seconds", 1.0))
    clip_samples = int(round(sample_rate * audio_seconds))
    print(
        f"[web-demo] loaded checkpoint='{ckpt_name}' path='{ckpt_path}' wheel='{wheel}' device='{device.type}'",
        flush=True,
    )
    if str(checkpoint).strip().lower() == "auto":
        try:
            from kws.demo.rank_checkpoints import select_best_checkpoint

            outputs_root = Path("outputs")
            if outputs_root.exists():
                local_auto_path, _local_device = select_best_checkpoint(
                    outputs_root=outputs_root,
                    device="cpu",
                    use_cache=True,
                    rebuild=False,
                    benchmark_iters=1,
                    selection_profile=selection_profile,
                )
                if local_auto_path.resolve() != ckpt_path.resolve():
                    print(
                        "[web-demo] warning: local auto checkpoint differs from public web checkpoint "
                        f"(local='{local_auto_path}' public='{ckpt_path}')",
                        flush=True,
                    )
        except Exception as exc:
            print(f"[web-demo] checkpoint alignment check skipped: {type(exc).__name__}: {exc}", flush=True)

    return LoadedWebDemo(
        checkpoint_path=ckpt_path,
        runtime_device=device,
        checkpoint_name=ckpt_name,
        model=model,
        frontend=frontend,
        command31_labels=command31_labels,
        wheel=wheel,
        keyword_calibration=keyword_calibration,
        sample_rate=sample_rate,
        clip_samples=clip_samples,
        verifier=verifier,
        display_conf_thr=tuning.display_conf_thr,
        display_wake_thr=tuning.display_wake_thr,
        default_vote_window=tuning.vote_window,
        default_vote_min_count=tuning.vote_min_count,
    )


def _top_confusions(label: str | None, command_probs: np.ndarray, command31_labels: Sequence[str]) -> list[dict[str, float | str]]:
    if label is None:
        return []
    confusable = DEFAULT_RUNTIME_CONFUSION_GROUPS.get(str(label), ())
    items: list[dict[str, float | str]] = []
    labels = list(command31_labels)
    for rival in confusable:
        if rival not in labels:
            continue
        rival_idx = labels.index(rival)
        items.append({"label": rival, "score": float(command_probs[rival_idx])})
    items.sort(key=lambda item: float(item["score"]), reverse=True)
    return items


def predict_web_clip(bundle: LoadedWebDemo, audio: object | None) -> WebDemoResult:
    sample_rate, waveform_np = _normalize_audio_input(audio)
    rms, peak = _compute_rms_and_peak(waveform_np)
    waveform = torch.from_numpy(waveform_np)
    waveform = _resample_waveform(waveform, sample_rate, bundle.sample_rate)
    waveform = pad_or_trim(waveform, target_samples=bundle.clip_samples)

    if rms < NO_SPEECH_RMS and peak < NO_SPEECH_PEAK:
        empty_scores = {label: 0.0 for label in wheel_labels(bundle.wheel, bundle.command31_labels)}
        return WebDemoResult(
            label="UNKNOWN",
            confidence=0.0,
            wake_prob=0.0,
            gate_open=False,
            status_message="No clear command detected. Try speaking a little louder or closer to the mic.",
            keyword_scores=empty_scores,
            top_confusions=[],
            latency_ms=0.0,
            wheel_active_label=None,
            preview_label=None,
            preview_reason="no-speech",
            raw_command_confidence=0.0,
            raw_wake_probability=0.0,
        )

    feature = bundle.frontend(waveform)
    mean = feature.mean()
    std = feature.std().clamp(min=1e-5)
    x = ((feature - mean) / std).unsqueeze(0).to(bundle.runtime_device)

    with torch.no_grad():
        start = time.perf_counter()
        out = bundle.model(x)
        if bundle.runtime_device.type == "cuda":
            torch.cuda.synchronize()
        elif bundle.runtime_device.type == "mps":
            torch.mps.synchronize()
        latency_ms = float((time.perf_counter() - start) * 1000.0)

        command_probs = torch.softmax(out.command_logits, dim=-1).squeeze(0).detach().cpu().numpy()
        wake_prob = float(torch.sigmoid(out.wake_logits).squeeze(0).item())

    kws12_probs = aggregate_command_probs_to_kws12(command_probs, bundle.command31_labels)
    detector_idx = int(np.argmax(kws12_probs))
    detector_label = KWS12_LABELS[detector_idx]
    if kws12_probs.size > 1:
        top2 = np.partition(kws12_probs, -2)[-2:]
        detector_margin = float(top2[-1] - top2[-2])
    else:
        detector_margin = float(kws12_probs[detector_idx])
    gate_open = wake_prob >= bundle.display_wake_thr
    raw_active_label, fallback_display_label, decision_conf = resolve_display_candidate(
        bundle.wheel,
        bundle.command31_labels,
        command_probs,
        gate_open,
    )
    keyword_conf_thr, vote_window, vote_min_count, prototype_bonus_cap, min_margin, _highlight_hold_ms = keyword_runtime_params(
        bundle.keyword_calibration,
        raw_active_label,
        default_conf_thr=bundle.display_conf_thr,
        default_vote_window=bundle.default_vote_window,
        default_vote_min_count=bundle.default_vote_min_count,
    )
    _ = (vote_window, vote_min_count)
    decision_conf = blend_keyword_score(decision_conf, wake_prob, 0.0, prototype_bonus_cap=prototype_bonus_cap)
    passes_margin, _margin = passes_confusion_guardrail(
        candidate_label=raw_active_label,
        command_probs=command_probs,
        command31_labels=bundle.command31_labels,
        calibration=bundle.keyword_calibration,
        min_margin=min_margin,
    )
    accepted_label = accept_display_candidate(
        gate_open=gate_open,
        candidate_label=raw_active_label if passes_margin else None,
        command_conf=decision_conf,
        wake_prob=wake_prob,
        min_command_conf=keyword_conf_thr,
        min_wake_prob=bundle.display_wake_thr,
    )
    verifier_decision = None
    verifier_candidate = accepted_label or select_verifier_candidate(
        detector_label=detector_label,
        detector_margin=detector_margin,
        detector_probs_kws12=kws12_probs,
        decision_profile="stable",
        margin_trigger=0.20,
    )
    if verifier_candidate in KWS12_LABELS[2:]:
        verifier_decision = verify_keyword(
            bundle.verifier,
            x,
            candidate_label=verifier_candidate,
        )
        if verifier_decision is not None and not verifier_decision.accepted:
            accepted_label = None
        elif accepted_label is None and verifier_decision is not None and verifier_decision.accepted:
            accepted_label = verifier_candidate
    display_label = accepted_label if accepted_label is not None else fallback_display_label
    keyword_scores = {label: float(kws12_probs[idx]) for idx, label in enumerate(KWS12_LABELS)}

    if accepted_label is not None:
        status_message = f"Detected '{accepted_label}' with confidence {decision_conf:.2f}."
        final_label = accepted_label.upper()
        wheel_active = accepted_label
    elif display_label == "SILENCE":
        status_message = "No speech command detected in this clip."
        final_label = "SILENCE"
        wheel_active = None
    else:
        if verifier_decision is not None and not verifier_decision.accepted:
            status_message = (
                f"Verifier rejected the initial candidate and preferred '{verifier_decision.top_label}'."
            )
        else:
            status_message = "No clear command detected. The model thinks this clip is closer to unknown speech."
        final_label = display_label.upper()
        wheel_active = None

    return WebDemoResult(
        label=final_label,
        confidence=float(decision_conf if accepted_label is not None else max(kws12_probs)),
        wake_prob=float(wake_prob),
        gate_open=bool(gate_open),
        status_message=status_message,
        keyword_scores=keyword_scores,
        top_confusions=_top_confusions(raw_active_label, command_probs, bundle.command31_labels),
        latency_ms=float(latency_ms),
        wheel_active_label=wheel_active,
        preview_label=accepted_label,
        preview_reason="single-clip" if accepted_label is not None else "clip-no-match",
        raw_command_confidence=float(np.max(kws12_probs)),
        raw_wake_probability=float(wake_prob),
    )


def render_keyword_wheel(
    result: WebDemoResult,
    *,
    title: str = "Google Speech Dataset Demo",
    prompt_label: str | None = None,
    status_override: str | None = None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(KWS12_LABELS)
    fig, ax = plt.subplots(figsize=(6.6, 7.0), dpi=140)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.06, right=0.94)
    ax.set_position([0.10, 0.14, 0.80, 0.66])
    apply_theme(fig, ax)
    artists = build_wheel(ax, labels, fontsize=11, labeldistance=0.80, radius=1.0, ring_width=0.40, abbreviate=False)
    hud = create_hud(fig, ax, title_text=title)
    active_idx = resolve_active_index(labels, result.wheel_active_label)
    update_wheel(artists, active_idx)
    hud.prompt.set_text(f"[{prompt_label or ('MATCH' if result.wheel_active_label else 'LISTENING')}]")
    hud.center.set_text(result.label)
    if status_override is not None:
        status = status_override
    else:
        top_scores = sorted(result.keyword_scores.items(), key=lambda item: item[1], reverse=True)[:4]
        status = f"wake={result.wake_prob:.2f} conf={result.confidence:.2f} top=" + ", ".join(
            f"{label}:{score:.2f}" for label, score in top_scores
        )
    hud.status.set_text(status)
    return fig


def format_top_confusions(result: WebDemoResult) -> str:
    if not result.top_confusions:
        return "No strong hard-word confusion competitors for this clip."
    return "\n".join(f"- {item['label']}: {float(item['score']):.3f}" for item in result.top_confusions)


@lru_cache(maxsize=1)
def load_cached_web_demo(
    checkpoint: str = "auto",
    selection_profile: str = "stable",
    sensitivity_profile: str = "strict",
) -> LoadedWebDemo:
    return load_web_demo(
        checkpoint=checkpoint,
        selection_profile=selection_profile,
        sensitivity_profile=sensitivity_profile,
    )


def _empty_keyword_scores() -> Dict[str, float]:
    return {label: 0.0 for label in KWS12_LABELS}


def _blank_result(
    *,
    label: str,
    status_message: str,
    gate_open: bool = False,
    confidence: float = 0.0,
    wake_prob: float = 0.0,
    latency_ms: float = 0.0,
        wheel_active_label: str | None = None,
) -> WebDemoResult:
    return WebDemoResult(
        label=label,
        confidence=confidence,
        wake_prob=wake_prob,
        gate_open=gate_open,
        status_message=status_message,
        keyword_scores=_empty_keyword_scores(),
        top_confusions=[],
        latency_ms=latency_ms,
        wheel_active_label=wheel_active_label,
        preview_label=None,
        preview_reason="blank",
        raw_command_confidence=0.0,
        raw_wake_probability=wake_prob,
    )


class PublicBrowserDemo:
    def __init__(self, checkpoint: str = "auto", *, selection_profile: str = "stable", sensitivity_profile: str = "strict") -> None:
        self.checkpoint = checkpoint
        self.selection_profile = selection_profile
        self.sensitivity_profile = sensitivity_profile

    def _bundle(self) -> LoadedWebDemo:
        return load_cached_web_demo(
            self.checkpoint,
            selection_profile=self.selection_profile,
            sensitivity_profile=self.sensitivity_profile,
        )

    def _new_stream_state(self, bundle: LoadedWebDemo, sample_rate: int) -> PublicBrowserStreamState:
        started_at = time.monotonic()
        gate = GateStateMachine(
            mode="adaptive",
            open_threshold=WEB_GATE_OPEN_FLOOR,
            close_threshold=WEB_GATE_CLOSE_FLOOR,
            cmd_conf_threshold=WEB_GATE_CMD_CONF_THR,
            hold_seconds=STREAM_HOLD_MS / 1000.0,
            adaptive=AdaptiveGateConfig(
                calibration_seconds=WEB_GATE_CALIBRATION_SECONDS,
                open_offset=WEB_GATE_OPEN_OFFSET,
                close_offset=WEB_GATE_CLOSE_OFFSET,
                open_floor=WEB_GATE_OPEN_FLOOR,
                close_floor=WEB_GATE_CLOSE_FLOOR,
                calibration_score_cap=WEB_GATE_CALIBRATION_WAKE_CAP,
                max_open_threshold=WEB_GATE_MAX_OPEN_THR,
            ),
        )
        gate.reset(now=started_at)
        clip_seconds = bundle.clip_samples / float(bundle.sample_rate)
        stream_clip_samples = int(round(max(1.0, sample_rate * clip_seconds)))
        min_buffer_samples = int(round(max(1.0, sample_rate * min(clip_seconds, WEB_MIN_BUFFER_SECONDS))))
        return PublicBrowserStreamState(
            sample_rate=sample_rate,
            buffer=AudioRingBuffer(stream_clip_samples),
            min_buffer_samples=min_buffer_samples,
            gate=gate,
            smoother=TemporalLabelSmoother(
                window_size=bundle.default_vote_window,
                min_count=bundle.default_vote_min_count,
                hold_seconds=STREAM_HOLD_MS / 1000.0,
                max_window_size=max(bundle.default_vote_window, 6),
            ),
            preview=HighlightPreviewState(),
            precheck_started_at=started_at,
            last_open_thr=WEB_GATE_OPEN_FLOOR,
            last_close_thr=WEB_GATE_CLOSE_FLOOR,
        )

    def _ensure_stream_state(
        self,
        bundle: LoadedWebDemo,
        audio: object | None,
        state: PublicBrowserStreamState | None,
    ) -> tuple[PublicBrowserStreamState, int, np.ndarray]:
        sample_rate, waveform_np = _normalize_audio_input(audio)
        if state is None or state.sample_rate != sample_rate:
            state = self._new_stream_state(bundle, sample_rate)
        return state, sample_rate, waveform_np

    def _render_stream_outputs(
        self,
        bundle: LoadedWebDemo,
        state: PublicBrowserStreamState | None,
        result: WebDemoResult,
        *,
        prompt_label: str,
        extra_status: str,
    ):
        mic_state = state.mic_state if state is not None else MIC_CHECK
        gate_state = state.last_gate_state if state is not None else "calibrating"
        open_thr = state.last_open_thr if state is not None else bundle.display_wake_thr
        close_thr = state.last_close_thr if state is not None else max(0.0, bundle.display_wake_thr - 0.05)
        gain_db = state.input_gain_db if state is not None else 0.0
        clip_flag = state.is_clipping if state is not None else False
        preview_label = result.preview_label or "none"
        preview_reason = result.preview_reason or "n/a"
        summary = (
            f"Prompt: {prompt_label}\n"
            f"Label: {result.label}\n"
            f"Confidence: {result.confidence:.3f}\n"
            f"Wake probability: {result.wake_prob:.3f}\n"
            f"Gate: {gate_state}\n"
            f"Latency: {result.latency_ms:.2f} ms"
        )
        status = (
            f"{result.status_message}\n"
            f"Mic state: {mic_state}\n"
            f"Wake open/close: {open_thr:.2f} / {close_thr:.2f}\n"
            f"Preview: {preview_label}\n"
            f"Preview reason: {preview_reason}\n"
            f"Raw top conf: {result.raw_command_confidence:.3f}\n"
            f"Raw wake: {result.raw_wake_probability:.3f}\n"
            f"Input gain: {gain_db:.1f} dB\n"
            f"Clipping: {'yes' if clip_flag else 'no'}\n"
            f"{extra_status}"
        )
        wheel_status = (
            f"mic={mic_state} gate={gate_state} wake={result.wake_prob:.2f} conf={result.confidence:.2f} "
            f"preview={preview_label} "
            f"open={open_thr:.2f} close={close_thr:.2f} gain={gain_db:.1f}dB clip={int(clip_flag)}"
        )
        figure = render_keyword_wheel(result, prompt_label=prompt_label, status_override=wheel_status)
        return summary, status, format_top_confusions(result), figure, state

    def predict(self, audio: object | None):
        try:
            result = predict_web_clip(self._bundle(), audio)
        except Exception as exc:
            summary = "Label: ERROR\nConfidence: 0.000\nWake probability: 0.000\nGate: closed\nLatency: 0.00 ms"
            status = f"Model startup or inference failed: {type(exc).__name__}: {exc}"
            return summary, status, "No confusion data available because inference did not complete.", None
        figure = render_keyword_wheel(result)
        summary = (
            f"Label: {result.label}\n"
            f"Confidence: {result.confidence:.3f}\n"
            f"Wake probability: {result.wake_prob:.3f}\n"
            f"Gate: {'open' if result.gate_open else 'closed'}\n"
            f"Latency: {result.latency_ms:.2f} ms"
        )
        return summary, result.status_message, format_top_confusions(result), figure

    def _render_stream_error(
        self,
        *,
        state: PublicBrowserStreamState | None,
        stage: str,
        exc: Exception,
        bundle: LoadedWebDemo | None = None,
    ):
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        result = _blank_result(label="ERROR", status_message=f"Live stream failed: {type(exc).__name__}: {exc}")
        extra_status = f"Failure stage: {stage}\nError detail: {type(exc).__name__}: {exc}"
        if bundle is None:
            summary = (
                "Prompt: ERROR\n"
                "Label: ERROR\n"
                "Confidence: 0.000\n"
                "Wake probability: 0.000\n"
                "Gate: closed\n"
                "Latency: 0.00 ms"
            )
            return summary, f"{result.status_message}\n{extra_status}", format_top_confusions(result), None, state
        return self._render_stream_outputs(bundle, state, result, prompt_label="ERROR", extra_status=extra_status)

    def advance_stream_state(
        self,
        bundle: LoadedWebDemo,
        audio: object | None,
        state: PublicBrowserStreamState | None,
    ) -> StreamAdvanceStep:
        try:
            state, sample_rate, chunk_np = self._ensure_stream_state(bundle, audio, state)
        except Exception as exc:
            raise StreamAdvanceError("input-normalization", exc) from exc

        try:
            chunk_rms = compute_rms(chunk_np)
            chunk_peak = float(np.max(np.abs(chunk_np))) if chunk_np.size > 0 else 0.0
            speech_like_active, boosted_chunk_rms, boosted_chunk_peak = _update_speech_activity(
                state,
                chunk_rms=chunk_rms,
                chunk_peak=chunk_peak,
            )
            state.precheck_rms_hist.append(chunk_rms)
            state.precheck_peak_hist.append(chunk_peak)
            state.buffer.append(chunk_np)
        except Exception as exc:
            raise StreamAdvanceError("buffer-update", exc) from exc

        now = time.monotonic()
        try:
            if not state.precheck_passed:
                if speech_like_active and state.buffer.size >= state.min_buffer_samples:
                    state.precheck_passed = True
                    state.mic_state = MIC_RUNNING
                elif (now - state.precheck_started_at) < STREAM_PRECHECK_SECONDS:
                    state.mic_state = MIC_CHECK
                else:
                    med = (
                        float(np.median(np.asarray(state.precheck_rms_hist, dtype=np.float64)))
                        if state.precheck_rms_hist
                        else 0.0
                    )
                    peak95 = (
                        float(np.percentile(np.asarray(state.precheck_peak_hist, dtype=np.float64), 95.0))
                        if state.precheck_peak_hist
                        else 0.0
                    )
                    state.mic_state, state.precheck_threshold = classify_precheck_signal(med, peak95, STREAM_MIN_RMS)
                    state.precheck_passed = state.mic_state == MIC_RUNNING or speech_like_active
                    if state.precheck_passed:
                        state.mic_state = MIC_RUNNING
        except Exception as exc:
            raise StreamAdvanceError("precheck", exc) from exc

        extra_status = (
            f"Buffered samples: {state.buffer.size} / ready {state.min_buffer_samples} "
            f"(capacity {state.buffer.capacity})"
            f" boosted_rms={boosted_chunk_rms:.4f} boosted_peak={boosted_chunk_peak:.3f}"
        )
        if state.buffer.size < state.min_buffer_samples:
            result = _blank_result(label="MIC CHECK", status_message="Listening for a stable microphone signal.")
            if state.mic_state == MIC_NO_SIGNAL:
                result = _blank_result(label="NO SIGNAL", status_message="No strong microphone signal yet. Try speaking a little louder.")
            elif state.mic_state == MIC_PERMISSION_DENIED:
                result = _blank_result(label="MIC BLOCKED", status_message="Browser microphone permission looks blocked.")
            prompt = "MIC CHECK" if state.mic_state == MIC_CHECK else state.mic_state.replace("_", " ")
            trace_line = (
                f"prompt={prompt} mic={state.mic_state} buffered={state.buffer.size}/{state.min_buffer_samples} "
                f"preview=none gate={state.last_gate_state}"
            )
            return StreamAdvanceStep(state=state, result=result, prompt_label=prompt, extra_status=extra_status, trace_line=trace_line)

        try:
            waveform_np = state.buffer.latest().copy()
            waveform_rms = compute_rms(waveform_np)
            gain_lin, gain_db = compute_auto_gain(waveform_rms, STREAM_TARGET_RMS, STREAM_MAX_GAIN_DB, enabled=True)
            state.input_gain_db = gain_db
            waveform_np *= float(gain_lin)
            state.is_clipping = bool(np.max(np.abs(waveform_np)) >= 0.999)
            np.clip(waveform_np, -1.0, 1.0, out=waveform_np)
        except Exception as exc:
            raise StreamAdvanceError("auto-gain", exc) from exc

        if state.last_infer_at and (now - state.last_infer_at) < STREAM_HOP_SECONDS and state.last_result is not None:
            prompt_label = state.last_result.label if state.last_result.label in {"MATCH", "LISTENING", "CALIBRATING"} else "LISTENING"
            trace_line = (
                f"prompt={prompt_label} mic={state.mic_state} gate={state.last_gate_state} "
                f"preview={state.last_result.preview_label or 'none'} label={state.last_result.label}"
            )
            return StreamAdvanceStep(
                state=state,
                result=state.last_result,
                prompt_label=prompt_label,
                extra_status=extra_status,
                trace_line=trace_line,
            )
        state.last_infer_at = now

        try:
            waveform = torch.from_numpy(waveform_np)
            waveform = _resample_waveform(waveform, sample_rate, bundle.sample_rate)
            waveform = pad_or_trim(waveform, target_samples=bundle.clip_samples)
            feature = bundle.frontend(waveform)
            mean = feature.mean()
            std = feature.std().clamp(min=1e-5)
            x = ((feature - mean) / std).unsqueeze(0).to(bundle.runtime_device)
        except Exception as exc:
            raise StreamAdvanceError("feature-prep", exc) from exc

        try:
            with torch.no_grad():
                start = time.perf_counter()
                out = bundle.model(x)
                latency_ms = float((time.perf_counter() - start) * 1000.0)
                command_probs = torch.softmax(out.command_logits, dim=-1).squeeze(0).detach().cpu().numpy()
                wake_prob = float(torch.sigmoid(out.wake_logits).squeeze(0).item())
        except Exception as exc:
            raise StreamAdvanceError("model-inference", exc) from exc

        try:
            state.ema_probs = ema_update(state.ema_probs, command_probs, STREAM_EMA_ALPHA)
            state.ema_wake = ema_update_scalar(state.ema_wake, wake_prob, STREAM_EMA_ALPHA)
            smoothed_probs = state.ema_probs
            smoothed_wake = float(state.ema_wake)
            raw_command_conf = float(np.max(smoothed_probs))

            gate_open, gate_state, open_thr, close_thr = state.gate.update(
                now=now,
                wake_prob=smoothed_wake,
                command_conf=raw_command_conf,
                calibration_wake_prob=None if speech_like_active else smoothed_wake,
            )
            state.last_gate_state = gate_state
            state.last_open_thr = open_thr
            state.last_close_thr = close_thr

            decision = resolve_runtime_decision(
                now=now,
                wheel=bundle.wheel,
                command31_labels=bundle.command31_labels,
                command_probs=smoothed_probs,
                gate_open=gate_open,
                gate_state=gate_state,
                wake_prob=smoothed_wake,
                display_wake_thr=bundle.display_wake_thr,
                calibration=bundle.keyword_calibration,
                default_conf_thr=bundle.display_conf_thr,
                default_vote_window=bundle.default_vote_window,
                default_vote_min_count=bundle.default_vote_min_count,
                smoother=state.smoother,
                preview=state.preview,
                prototype_similarity=0.0,
                preview_requires_gate=False,
                preview_min_command_conf=WEB_PREVIEW_MIN_COMMAND_CONF,
                preview_min_wake_prob=0.0 if speech_like_active else WEB_PREVIEW_MIN_WAKE_PROB,
                preview_hold_seconds=WEB_PREVIEW_HOLD_SECONDS,
            )
            verifier_decision = None
            verifier_rejected = False
            kws12_probs = aggregate_command_probs_to_kws12(smoothed_probs, bundle.command31_labels)
            detector_idx = int(np.argmax(kws12_probs))
            detector_label = KWS12_LABELS[detector_idx]
            if kws12_probs.size > 1:
                top2 = np.partition(kws12_probs, -2)[-2:]
                detector_margin = float(top2[-1] - top2[-2])
            else:
                detector_margin = float(kws12_probs[detector_idx])
            verifier_candidate = decision.active_label or select_verifier_candidate(
                detector_label=detector_label,
                detector_margin=detector_margin,
                detector_probs_kws12=kws12_probs,
                decision_profile="stable",
                margin_trigger=0.20,
            )
            if verifier_candidate in KWS12_LABELS[2:]:
                verifier_decision = verify_keyword(bundle.verifier, x, candidate_label=verifier_candidate)
                verifier_rejected = bool(verifier_decision is not None and not verifier_decision.accepted)
        except Exception as exc:
            raise StreamAdvanceError("runtime-decision", exc) from exc

        if verifier_rejected:
            result = WebDemoResult(
                label="UNKNOWN",
                confidence=0.0,
                wake_prob=smoothed_wake,
                gate_open=bool(gate_open),
                status_message=f"Verifier rejected the initial candidate and preferred '{verifier_decision.top_label}'.",
                keyword_scores={label: float(kws12_probs[idx]) for idx, label in enumerate(KWS12_LABELS)},
                top_confusions=_top_confusions(decision.raw_active_label, smoothed_probs, bundle.command31_labels),
                latency_ms=latency_ms,
                wheel_active_label=None,
                preview_label=decision.preview_label,
                preview_reason=f"{decision.preview_reason}+verifier-reject",
                raw_command_confidence=float(np.max(kws12_probs)),
                raw_wake_probability=smoothed_wake,
            )
            state.last_result = result
            prompt_label = "LISTENING" if result.preview_label else "READY"
            extra_status = f"Gate={state.last_gate_state} thr={state.last_open_thr:.2f}/{state.last_close_thr:.2f}"
            trace_line = (
                f"{result.label} wake={result.wake_prob:.2f} conf={result.confidence:.2f} "
                f"preview={result.preview_label or '-'} reason={result.preview_reason}"
            )
            return StreamAdvanceStep(
                state=state,
                result=result,
                prompt_label=prompt_label,
                extra_status=extra_status,
                trace_line=trace_line,
            )

        result = WebDemoResult(
            label=decision.final_label,
            confidence=float(decision.command_confidence if decision.active_label is not None else max(kws12_probs)),
            wake_prob=smoothed_wake,
            gate_open=bool(gate_open),
            status_message=decision.status_message,
            keyword_scores={label: float(kws12_probs[idx]) for idx, label in enumerate(KWS12_LABELS)},
            top_confusions=_top_confusions(decision.raw_active_label, smoothed_probs, bundle.command31_labels),
            latency_ms=latency_ms,
            wheel_active_label=decision.highlight_label,
            preview_label=decision.preview_label,
            preview_reason=decision.preview_reason,
            raw_command_confidence=decision.raw_command_confidence,
            raw_wake_probability=decision.raw_wake_probability,
        )
        if result.preview_label and not result.gate_open and result.label == "LISTENING":
            result = WebDemoResult(
                label=result.label,
                confidence=result.confidence,
                wake_prob=result.wake_prob,
                gate_open=result.gate_open,
                status_message=f"Heard speech-like audio. Previewing '{result.preview_label}' while waiting for a stable match.",
                keyword_scores=result.keyword_scores,
                top_confusions=result.top_confusions,
                latency_ms=result.latency_ms,
                wheel_active_label=result.wheel_active_label,
                preview_label=result.preview_label,
                preview_reason=result.preview_reason,
                raw_command_confidence=result.raw_command_confidence,
                raw_wake_probability=result.raw_wake_probability,
            )

        state.last_result = result
        if (not state.precheck_passed) and not (result.preview_label or result.wheel_active_label):
            blank = _blank_result(label="MIC CHECK", status_message="Listening for a stable microphone signal.")
            if state.mic_state == MIC_NO_SIGNAL:
                blank = _blank_result(label="NO SIGNAL", status_message="No strong microphone signal yet. Try speaking a little louder.")
            elif state.mic_state == MIC_PERMISSION_DENIED:
                blank = _blank_result(label="MIC BLOCKED", status_message="Browser microphone permission looks blocked.")
            prompt = "MIC CHECK" if state.mic_state == MIC_CHECK else state.mic_state.replace("_", " ")
            trace_line = (
                f"prompt={prompt} mic={state.mic_state} gate={state.last_gate_state} "
                f"preview=none wake={result.wake_prob:.3f} conf={result.raw_command_confidence:.3f}"
            )
            return StreamAdvanceStep(state=state, result=blank, prompt_label=prompt, extra_status=extra_status, trace_line=trace_line)

        prompt_label = "LISTENING" if (result.preview_label and not result.gate_open and decision.prompt_label != "MATCH") else decision.prompt_label
        trace_line = (
            f"prompt={prompt_label} mic={state.mic_state} gate={state.last_gate_state} "
            f"preview={result.preview_label or 'none'} highlight={result.wheel_active_label or 'none'} "
            f"wake={result.raw_wake_probability:.3f} conf={result.raw_command_confidence:.3f}"
        )
        return StreamAdvanceStep(state=state, result=result, prompt_label=prompt_label, extra_status=extra_status, trace_line=trace_line)

    def start_stream(self):
        bundle = self._bundle()
        result = _blank_result(label="MIC CHECK", status_message="Browser microphone opened. Start speaking to calibrate the live gate.")
        return self._render_stream_outputs(bundle, None, result, prompt_label="MIC CHECK", extra_status="Live browser stream is waiting for audio.")

    def stop_stream(self, state: PublicBrowserStreamState | None):
        bundle = self._bundle()
        if state is None or state.last_result is None:
            result = _blank_result(label="READY", status_message="Recording stopped. Press record to listen again.")
            return self._render_stream_outputs(bundle, state, result, prompt_label="READY", extra_status="No live session is active.")
        result = _blank_result(
            label=state.last_result.label,
            status_message="Recording stopped. Press record to listen again.",
            gate_open=state.last_result.gate_open,
            confidence=state.last_result.confidence,
            wake_prob=state.last_result.wake_prob,
            latency_ms=state.last_result.latency_ms,
            wheel_active_label=state.last_result.wheel_active_label,
        )
        result.keyword_scores.update(state.last_result.keyword_scores)
        result.top_confusions.extend(state.last_result.top_confusions)
        return self._render_stream_outputs(bundle, state, result, prompt_label="READY", extra_status="Live streaming is paused.")

    def reset_stream(self):
        bundle = self._bundle()
        result = _blank_result(label="READY", status_message="Press record to start the live browser microphone demo.")
        return self._render_stream_outputs(bundle, None, result, prompt_label="READY", extra_status="Session reset.")

    def stream(self, audio: object | None, state: PublicBrowserStreamState | None):
        bundle = None
        try:
            bundle = self._bundle()
            step = self.advance_stream_state(bundle, audio, state)
            return self._render_stream_outputs(
                bundle,
                step.state,
                step.result,
                prompt_label=step.prompt_label,
                extra_status=step.extra_status,
            )
        except StreamAdvanceError as exc:
            return self._render_stream_error(state=state, stage=exc.stage, exc=exc.original, bundle=bundle)
        except Exception as exc:
            return self._render_stream_error(state=state, stage="stream-entry", exc=exc, bundle=bundle)

    def debug_stream_file(self, audio_file: object | None):
        bundle = None
        try:
            bundle = self._bundle()
            sample_rate, waveform_np = _normalize_audio_input(audio_file)
            chunk_size = max(1, int(round(sample_rate * STREAM_HOP_SECONDS)))
            state: PublicBrowserStreamState | None = None
            trace_lines: list[str] = []
            for frame_idx, start in enumerate(range(0, waveform_np.size, chunk_size), start=1):
                chunk = waveform_np[start : start + chunk_size]
                if chunk.size == 0:
                    continue
                step = self.advance_stream_state(bundle, (sample_rate, chunk), state)
                state = step.state
                if not trace_lines or trace_lines[-1] != step.trace_line:
                    trace_lines.append(f"{frame_idx:03d}: {step.trace_line}")
            if state is None:
                raise ValueError("No audio frames were available after loading the debug file.")
            if state.last_result is None:
                result = _blank_result(label="READY", status_message="Debug stream processed no frames.")
                outputs = self._render_stream_outputs(
                    bundle,
                    state,
                    result,
                    prompt_label="READY",
                    extra_status="Debug stream processed no frames.",
                )
            else:
                outputs = self._render_stream_outputs(
                    bundle,
                    state,
                    state.last_result,
                    prompt_label=(state.last_result.label if state.last_result.label in {"MATCH", "LISTENING", "CALIBRATING"} else "LISTENING"),
                    extra_status="Debug stream replay complete.",
                )
            trace_text = "\n".join(trace_lines[-24:]) or "No debug trace recorded."
            return (*outputs[:4], trace_text)
        except StreamAdvanceError as exc:
            error_outputs = self._render_stream_error(
                state=None,
                stage=exc.stage,
                exc=exc.original,
                bundle=bundle,
            )
            trace_text = f"debug-stream failed at {exc.stage}: {type(exc.original).__name__}: {exc.original}"
            return (*error_outputs[:4], trace_text)
        except Exception as exc:
            error_outputs = self._render_stream_error(state=None, stage="debug-stream", exc=exc, bundle=bundle)
            trace_text = f"debug-stream failed at debug-stream: {type(exc).__name__}: {exc}"
            return (*error_outputs[:4], trace_text)


def create_gradio_app(checkpoint: str = "auto", *, selection_profile: str = "stable", sensitivity_profile: str = "strict"):
    import gradio as gr

    demo = PublicBrowserDemo(
        checkpoint=checkpoint,
        selection_profile=selection_profile,
        sensitivity_profile=sensitivity_profile,
    )
    description = (
        "Speak directly into your browser mic to run the public keyword spotting demo in a continuous live loop. "
        "The UI mirrors the local desktop demo more closely with live listening, gate calibration, and keyword-wheel updates."
    )
    with gr.Blocks(theme=gr.themes.Soft(), title="Public KWS Browser Demo") as app:
        session = gr.State(value=None)
        gr.Markdown("# Public Keyword Spotting Demo")
        gr.Markdown(description)
        gr.Markdown("**Startup note**: the model loads lazily when you press record. The first live response may take a bit longer than later ones.")
        audio = gr.Audio(
            sources=["microphone"],
            type="numpy",
            streaming=True,
            show_download_button=False,
            label="Continuous browser microphone",
        )
        reset = gr.Button("Reset Session", variant="secondary")
        with gr.Row():
            summary = gr.Textbox(label="Prediction", lines=6, interactive=False)
            status = gr.Textbox(label="Status", lines=7, interactive=False)
        with gr.Row():
            confusions = gr.Textbox(label="Hard-word competitors", lines=5, interactive=False)
            figure = gr.Plot(label="Keyword wheel")
        debug_file = gr.File(label="Debug stream file", type="filepath", visible=False)
        debug_trace = gr.Textbox(label="Debug stream trace", visible=False)
        app.load(fn=demo.reset_stream, outputs=[summary, status, confusions, figure, session])
        audio.start_recording(fn=demo.start_stream, outputs=[summary, status, confusions, figure, session])
        audio.stream(
            fn=demo.stream,
            inputs=[audio, session],
            outputs=[summary, status, confusions, figure, session],
            stream_every=0.10,
            time_limit=None,
            concurrency_limit=1,
        )
        audio.stop_recording(fn=demo.stop_stream, inputs=[session], outputs=[summary, status, confusions, figure, session])
        reset.click(fn=demo.reset_stream, outputs=[summary, status, confusions, figure, session])
        debug_file.change(
            fn=demo.debug_stream_file,
            inputs=[debug_file],
            outputs=[summary, status, confusions, figure, debug_trace],
            api_name="debug_stream_file",
            show_progress="hidden",
        )
        gr.Markdown(
            "**Notes**\n"
            "- This hosted demo now runs as a continuous browser microphone loop instead of a one-shot clip button.\n"
            "- CPU latency in the cloud may differ from local Apple Silicon results.\n"
            "- Hard words such as left / on / down use stricter guardrails to reduce confusion."
        )
    return app.queue(default_concurrency_limit=1)


__all__ = [
    "DEFAULT_PUBLIC_CHECKPOINT",
    "DEFAULT_SPACE_CACHE_DIR",
    "PUBLIC_CHECKPOINTS",
    "PUBLIC_RELEASE_BASE",
    "LoadedWebDemo",
    "PublicBrowserDemo",
    "WebDemoResult",
    "create_gradio_app",
    "ensure_public_checkpoint",
    "format_top_confusions",
    "load_cached_web_demo",
    "load_web_demo",
    "predict_web_clip",
    "render_keyword_wheel",
]
