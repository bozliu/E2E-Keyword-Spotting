"""Realtime microphone demo for dual-task KWS (matplotlib GUI)."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio.transforms as ta_transforms

from kws.constants import CLIP_SECONDS, CLIP_SAMPLES, COMMAND31_LABELS, KWS12_LABELS, TARGET_KEYWORDS_10, command31_to_kws12
from kws.data.audio import MelFrontend, pad_or_trim
from kws.demo.rank_checkpoints import benchmark_latency_ms, select_best_checkpoint
from kws.demo.user_profile import DEFAULT_USER_PROFILE_PATH, PassiveKeywordProfile, blend_keyword_score
from kws.demo.visuals import TEXT, apply_theme, build_wheel, create_hud, resolve_active_index, update_wheel
from kws.models import create_model
from kws.train.engine import pick_device
from kws.utils.keyword_focus import DEFAULT_FOCUS_RUNTIME_OVERRIDES, DEFAULT_RUNTIME_CONFUSION_GROUPS, load_keyword_calibration

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None


MIC_NO_DEVICE = "NO_DEVICE"
MIC_PERMISSION_DENIED = "PERMISSION_DENIED"
MIC_NO_SIGNAL = "NO_SIGNAL"
MIC_READY = "READY"
MIC_RUNNING = "RUNNING"
MIC_CHECK = "MIC_CHECK"
MIC_RUNTIME_ERROR = "RUNTIME_ERROR"
DEFAULT_DEMO_LOCK_PATH = Path.home() / ".kws_demo" / "realtime.lock"


@dataclass(frozen=True)
class DemoSnapshot:
    updated_at: float
    gate_open: bool
    gate_state: str
    command_label: str
    display_label: str
    active_label: str | None
    highlight_label: str | None
    command_conf: float
    wake_prob: float
    wake_open_thr: float
    wake_close_thr: float
    latency_ms: float
    prompt_word: str
    prompt_status: str
    device: str
    selected_device: str
    queue_fill_ratio: float
    mic_state: str
    mic_rms: float
    input_gain_db: float
    is_clipping: bool
    precheck_passed: bool
    precheck_threshold: float
    stream_sample_rate: float
    input_device_name: str
    error_message: str = ""


@dataclass(frozen=True)
class MicPrecheckResult:
    state: str
    message: str
    rms: float
    peak: float
    passed: bool


@dataclass(frozen=True)
class AdaptiveGateConfig:
    calibration_seconds: float = 2.0
    open_offset: float = 0.12
    close_offset: float = 0.05
    open_floor: float = 0.25
    close_floor: float = 0.15


@dataclass(frozen=True)
class SensitivityTuning:
    cmd_conf_thr: float
    open_offset: float
    close_offset: float
    open_floor: float
    close_floor: float
    display_conf_thr: float
    display_wake_thr: float
    vote_window: int
    vote_min_count: int


@dataclass(frozen=True)
class AudioInputSpec:
    index: int
    name: str
    default_samplerate: float


class DemoInstanceLock:
    def __init__(self, path: str | Path = DEFAULT_DEMO_LOCK_PATH) -> None:
        self.path = Path(path).expanduser().resolve()
        self._held = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._reclaim_stale():
                    continue
                owner_pid = self._read_pid()
                owner = f" pid={owner_pid}" if owner_pid is not None else ""
                raise RuntimeError(f"Another demo instance is already running{owner}. Close it before restarting.")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(f"{os.getpid()}\n")
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise
            self._held = True
            return
        raise RuntimeError("Could not acquire realtime demo instance lock.")

    def release(self) -> None:
        if not self._held and not self.path.exists():
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self._held = False

    def _read_pid(self) -> int | None:
        try:
            raw = self.path.read_text(encoding="utf-8").strip()
        except Exception:
            return None
        if not raw:
            return None
        first_line = raw.splitlines()[0].strip()
        try:
            pid = int(first_line)
        except ValueError:
            return None
        return pid if pid > 0 else None

    def _reclaim_stale(self) -> bool:
        pid = self._read_pid()
        if pid is not None and _pid_is_running(pid):
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        return True


class AudioRingBuffer:
    """Fixed-size rolling waveform buffer optimized for chunk appends."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(max(1, capacity))
        self._data = np.zeros(self.capacity, dtype=np.float32)
        self._linearized = np.zeros(self.capacity, dtype=np.float32)
        self._write_idx = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._size >= self.capacity

    def append(self, chunk: np.ndarray) -> None:
        x = np.asarray(chunk, dtype=np.float32).reshape(-1)
        n = int(x.size)
        if n <= 0:
            return

        if n >= self.capacity:
            self._data[:] = x[-self.capacity :]
            self._write_idx = 0
            self._size = self.capacity
            return

        tail = self.capacity - self._write_idx
        if n <= tail:
            self._data[self._write_idx : self._write_idx + n] = x
        else:
            self._data[self._write_idx :] = x[:tail]
            self._data[: n - tail] = x[tail:]
        self._write_idx = (self._write_idx + n) % self.capacity
        self._size = min(self.capacity, self._size + n)

    def latest(self) -> np.ndarray:
        if self._size == 0:
            return self._linearized[:0]
        if self._size < self.capacity:
            return self._data[: self._size]

        if self._write_idx == 0:
            return self._data

        tail = self.capacity - self._write_idx
        self._linearized[:tail] = self._data[self._write_idx :]
        self._linearized[tail:] = self._data[: self._write_idx]
        return self._linearized


class TemporalLabelSmoother:
    """Keep a stable label only after repeated agreement across frames."""

    def __init__(self, *, window_size: int, min_count: int, hold_seconds: float, max_window_size: int | None = None) -> None:
        self.window_size = int(max(1, window_size))
        self.min_count = int(max(1, min(min_count, self.window_size)))
        self.max_window_size = int(max(self.window_size, max_window_size or self.window_size))
        self.hold_seconds = float(max(0.0, hold_seconds))
        self._history: deque[str | None] = deque(maxlen=self.max_window_size)
        self._stable_label: str | None = None
        self._stable_since = 0.0
        self._last_support_at = 0.0

    def reset(self) -> None:
        self._history.clear()
        self._stable_label = None
        self._stable_since = 0.0
        self._last_support_at = 0.0

    def update(
        self,
        *,
        now: float,
        candidate_label: str | None,
        min_count_override: int | None = None,
        window_size_override: int | None = None,
    ) -> str | None:
        self._history.append(candidate_label)
        effective_window = int(max(1, min(window_size_override or self.window_size, self.max_window_size)))
        effective_min_count = int(max(1, min(min_count_override or self.min_count, effective_window)))
        if candidate_label is not None:
            self._last_support_at = now
        elif self._stable_label is not None and (now - self._last_support_at) >= self.hold_seconds:
            self.reset()
            return None
        recent = list(self._history)[-effective_window:]
        counts = Counter(label for label in recent if label is not None)

        chosen: str | None = None
        if counts:
            chosen, chosen_count = max(
                counts.items(),
                key=lambda item: (item[1], int(item[0] == self._stable_label)),
            )
            if chosen_count < effective_min_count:
                chosen = None

        if chosen is not None:
            if chosen != self._stable_label:
                self._stable_label = chosen
                self._stable_since = now
            return self._stable_label

        if self._stable_label is not None and (now - self._stable_since) < self.hold_seconds:
            return self._stable_label

        self._stable_label = None
        return None


class HighlightPreviewState:
    """Hold a fast preview label briefly so the wheel can feel responsive."""

    def __init__(self) -> None:
        self._label: str | None = None
        self._hold_until = 0.0

    def reset(self) -> None:
        self._label = None
        self._hold_until = 0.0

    def update(self, *, now: float, candidate_label: str | None, hold_seconds: float) -> str | None:
        if candidate_label is not None:
            self._label = candidate_label
            self._hold_until = now + max(0.0, float(hold_seconds))
            return self._label
        if self._label is not None and now < self._hold_until:
            return self._label
        self._label = None
        self._hold_until = 0.0
        return None


class GateStateMachine:
    """Wake gate with fixed/adaptive thresholds and hysteresis."""

    def __init__(
        self,
        *,
        mode: str,
        open_threshold: float,
        close_threshold: float,
        cmd_conf_threshold: float,
        hold_seconds: float,
        adaptive: AdaptiveGateConfig,
    ) -> None:
        self.mode = mode
        self.open_threshold = float(max(0.0, open_threshold))
        self.close_threshold = float(max(0.0, min(close_threshold, self.open_threshold)))
        self.cmd_conf_threshold = float(max(0.0, cmd_conf_threshold))
        self.hold_seconds = float(max(0.0, hold_seconds))
        self.adaptive = adaptive

        self.state = "closed"
        self._below_close_frames = 0
        self._hold_until = 0.0
        self._calibration_start = time.monotonic()
        self._calibration_scores: List[float] = []
        self._calibrated = mode == "fixed"

    def _finish_calibration(self, now: float) -> None:
        if self.mode != "adaptive" or self._calibrated:
            return
        if (now - self._calibration_start) < self.adaptive.calibration_seconds:
            return
        if not self._calibration_scores:
            return

        p95 = float(np.percentile(np.asarray(self._calibration_scores, dtype=np.float64), 95.0))
        open_thr = max(p95 + self.adaptive.open_offset, self.adaptive.open_floor)
        close_thr = max(p95 + self.adaptive.close_offset, self.adaptive.close_floor)
        if close_thr >= open_thr:
            close_thr = max(0.05, open_thr - 0.02)

        self.open_threshold = float(open_thr)
        self.close_threshold = float(close_thr)
        self._calibrated = True
        self.state = "closed"
        self._below_close_frames = 0

    def update(self, *, now: float, wake_prob: float, command_conf: float) -> Tuple[bool, str, float, float]:
        if self.mode == "adaptive" and not self._calibrated:
            self._calibration_scores.append(float(wake_prob))
            self._finish_calibration(now)
            if not self._calibrated:
                self.state = "calibrating"
                return False, self.state, self.open_threshold, self.close_threshold

        if self.state == "closed":
            if wake_prob >= self.open_threshold and command_conf >= self.cmd_conf_threshold:
                self.state = "open"
                self._below_close_frames = 0
        elif self.state == "open":
            if wake_prob < self.close_threshold:
                self._below_close_frames += 1
            else:
                self._below_close_frames = 0

            if self._below_close_frames >= 2:
                self._below_close_frames = 0
                if self.hold_seconds > 0.0:
                    self.state = "hold"
                    self._hold_until = now + self.hold_seconds
                else:
                    self.state = "closed"
        elif self.state == "hold":
            if wake_prob >= self.open_threshold and command_conf >= self.cmd_conf_threshold:
                self.state = "open"
            elif now >= self._hold_until:
                self.state = "closed"

        gate_open = self.state in ("open", "hold")
        return gate_open, self.state, self.open_threshold, self.close_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime KWS demo (GUI)")
    parser.add_argument("--checkpoint", type=str, default="auto", help="Checkpoint path or 'auto'")
    parser.add_argument("--device", type=str, default="auto", help="auto|mps|cpu|cuda")
    parser.add_argument("--wheel", type=str, default="kws12", choices=["command31", "kws12", "target10"])

    parser.add_argument("--gate-mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    parser.add_argument("--sensitivity-profile", type=str, default="high", choices=["high", "balanced", "strict"])
    parser.add_argument("--threshold", type=float, default=None, help="Legacy alias for fixed wake thresholds")
    parser.add_argument("--wake-open-thr", type=float, default=0.6, help="Wake open threshold (fixed mode)")
    parser.add_argument("--wake-close-thr", type=float, default=0.5, help="Wake close threshold (fixed mode)")
    parser.add_argument("--calibration-seconds", type=float, default=2.0, help="Adaptive gate calibration duration")
    parser.add_argument("--cmd-conf-thr", type=float, default=None, help="Min command confidence to open wake gate")
    parser.add_argument("--display-conf-thr", type=float, default=None, help="Min command confidence to display a label")
    parser.add_argument("--display-wake-thr", type=float, default=None, help="Min wake probability to display a label")
    parser.add_argument("--vote-window", type=int, default=None, help="Temporal voting window size for stable labels")
    parser.add_argument("--vote-min-count", type=int, default=None, help="Minimal agreeing votes before updating label")

    parser.add_argument("--mic-precheck-seconds", type=float, default=1.5, help="Microphone precheck duration")
    parser.add_argument("--mic-min-rms", type=float, default=0.001, help="Minimal RMS floor for adaptive mic precheck")
    parser.add_argument("--auto-gain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-rms", type=float, default=0.05, help="Target RMS for lightweight auto gain")
    parser.add_argument("--max-gain-db", type=float, default=18.0, help="Auto gain upper bound in dB")

    parser.add_argument("--hop-seconds", type=float, default=0.10, help="Inference hop size in seconds")
    parser.add_argument("--ema-alpha", type=float, default=0.35, help="EMA smoothing factor")
    parser.add_argument("--fps", type=float, default=10.0, help="GUI refresh rate")
    parser.add_argument("--hold-ms", type=float, default=300.0, help="Gate/label hold window (ms)")

    # kept for backward compatibility; no longer used for UI prompting
    parser.add_argument("--prompt-interval", type=float, default=4.0, help="Deprecated")

    parser.add_argument("--audio-device", type=str, default="", help="Optional sounddevice device id/name")
    parser.add_argument("--stream-sample-rate", type=float, default=0.0, help="Optional input stream sample rate override")
    parser.add_argument("--list-audio-devices", action="store_true", help="Print audio devices and exit")
    parser.add_argument("--disable-passive-adaptation", action="store_true", help="Disable local passive user adaptation")
    parser.add_argument("--reset-user-profile", action="store_true", help="Reset local passive user adaptation cache before start")
    parser.add_argument("--user-profile-path", type=str, default=str(DEFAULT_USER_PROFILE_PATH), help="Local passive user profile cache path")
    parser.add_argument("--rebuild-ranking", action="store_true", help="Rebuild demo auto-ranking report")
    parser.add_argument("--no-cache-ranking", action="store_true", help="Do not reuse cached ranking report")
    parser.add_argument(
        "--ranking-iters",
        type=int,
        default=8,
        help="Latency benchmark iterations for auto-checkpoint ranking",
    )
    parser.add_argument(
        "--device-auto-bench-iters",
        type=int,
        default=6,
        help="Startup benchmark iterations for selecting fastest runtime device",
    )
    return parser.parse_args()


def get_sensitivity_tuning(profile: str) -> SensitivityTuning:
    profile = profile.lower().strip()
    if profile == "high":
        return SensitivityTuning(
            cmd_conf_thr=0.20,
            open_offset=0.06,
            close_offset=0.03,
            open_floor=0.18,
            close_floor=0.10,
            display_conf_thr=0.35,
            display_wake_thr=0.45,
            vote_window=4,
            vote_min_count=2,
        )
    if profile == "strict":
        return SensitivityTuning(
            cmd_conf_thr=0.40,
            open_offset=0.14,
            close_offset=0.08,
            open_floor=0.28,
            close_floor=0.18,
            display_conf_thr=0.55,
            display_wake_thr=0.60,
            vote_window=5,
            vote_min_count=3,
        )
    return SensitivityTuning(
        cmd_conf_thr=0.30,
        open_offset=0.10,
        close_offset=0.05,
        open_floor=0.22,
        close_floor=0.14,
        display_conf_thr=0.40,
        display_wake_thr=0.50,
        vote_window=4,
        vote_min_count=2,
    )


def map_demo_label(command_label: str, wake_prob: float, threshold: float) -> str:
    """Map model outputs to a demo label name."""
    return "unknown" if wake_prob < threshold else command_label


def ema_update(prev: Optional[np.ndarray], current: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return current
    return alpha * current + (1.0 - alpha) * prev


def ema_update_scalar(prev: Optional[float], current: float, alpha: float) -> float:
    if prev is None:
        return current
    return alpha * current + (1.0 - alpha) * prev


def _wheel_labels(wheel: str, command31_labels: Sequence[str]) -> List[str]:
    if wheel == "command31":
        return list(command31_labels)
    if wheel == "target10":
        return list(TARGET_KEYWORDS_10)
    return ["silence", "unknown", *TARGET_KEYWORDS_10]


def _active_label_for_wheel(wheel: str, command_label: str, gate_open: bool) -> Tuple[str | None, str]:
    if not gate_open:
        return None, "UNKNOWN"

    if wheel == "command31":
        return command_label, command_label

    if wheel == "target10":
        if command_label in TARGET_KEYWORDS_10:
            return command_label, command_label
        return None, "UNKNOWN"

    mapped_idx = command31_to_kws12(command_label)
    kws12_labels = ["silence", "unknown", *TARGET_KEYWORDS_10]
    mapped = kws12_labels[int(mapped_idx)]
    if mapped in ("unknown", "silence"):
        return None, mapped.upper()
    return mapped, mapped


def _accept_display_candidate(
    *,
    gate_open: bool,
    candidate_label: str | None,
    command_conf: float,
    wake_prob: float,
    min_command_conf: float,
    min_wake_prob: float,
) -> str | None:
    if not gate_open or candidate_label is None:
        return None
    if command_conf < min_command_conf:
        return None
    if wake_prob < min_wake_prob:
        return None
    return candidate_label


def _keyword_runtime_params(
    calibration: Dict[str, object] | None,
    label: str | None,
    *,
    default_conf_thr: float,
    default_vote_window: int,
    default_vote_min_count: int,
) -> Tuple[float, int, int, float, float, float]:
    payload = calibration or {}
    defaults = payload.get("defaults", {}) if isinstance(payload.get("defaults", {}), dict) else {}
    keywords = payload.get("keywords", {}) if isinstance(payload.get("keywords", {}), dict) else {}
    entry = keywords.get(str(label), {}) if label else {}
    if not isinstance(entry, dict):
        entry = {}
    weak_keywords = set(str(x) for x in payload.get("weak_keywords", [])) if isinstance(payload.get("weak_keywords", []), list) else set()
    focus_keywords = set(str(x) for x in payload.get("focus_keywords", [])) if isinstance(payload.get("focus_keywords", []), list) else set()
    groups = _confusable_groups(payload)
    is_weak = bool(label and str(label) in weak_keywords)
    has_confusions = bool(label and groups.get(str(label)))
    fallback_min_margin = 0.12 if is_weak else (0.08 if has_confusions else 0.0)
    fallback_highlight_hold_ms = 280.0 if is_weak else 220.0
    conf_thr = float(entry.get("command_conf_threshold", defaults.get("command_conf_threshold", default_conf_thr)))
    vote_window = int(entry.get("vote_window", defaults.get("vote_window", default_vote_window)))
    vote_min_count = int(entry.get("vote_min_count", defaults.get("vote_min_count", default_vote_min_count)))
    prototype_bonus = float(entry.get("prototype_bonus_max", defaults.get("prototype_bonus_max", 0.04)))
    min_margin = float(entry.get("min_margin", defaults.get("min_margin", fallback_min_margin)))
    highlight_hold_ms = float(entry.get("highlight_hold_ms", defaults.get("highlight_hold_ms", fallback_highlight_hold_ms)))
    if label and (str(label) in focus_keywords or str(label) in DEFAULT_FOCUS_RUNTIME_OVERRIDES):
        overrides = DEFAULT_FOCUS_RUNTIME_OVERRIDES.get(str(label), {})
        vote_window = max(vote_window, int(overrides.get("vote_window", vote_window)))
        vote_min_count = max(vote_min_count, int(overrides.get("vote_min_count", vote_min_count)))
        min_margin = max(min_margin, float(overrides.get("min_margin", min_margin)))
        highlight_hold_ms = max(highlight_hold_ms, float(overrides.get("highlight_hold_ms", highlight_hold_ms)))
    return (conf_thr, vote_window, vote_min_count, prototype_bonus, min_margin, highlight_hold_ms)


def _confusable_groups(calibration: Dict[str, object] | None) -> Dict[str, Tuple[str, ...]]:
    payload = calibration or {}
    groups = dict(DEFAULT_RUNTIME_CONFUSION_GROUPS)
    raw = payload.get("confusable_groups", {}) if isinstance(payload.get("confusable_groups", {}), dict) else {}
    for label, values in raw.items():
        if isinstance(values, (list, tuple)):
            groups[str(label)] = tuple(str(v) for v in values if str(v))
    return groups


def _command_prob_for_label(command_probs: np.ndarray, command31_labels: Sequence[str], label: str) -> float:
    try:
        idx = list(command31_labels).index(str(label))
    except ValueError:
        return 0.0
    return float(command_probs[idx])


def _passes_confusion_guardrail(
    *,
    candidate_label: str | None,
    command_probs: np.ndarray,
    command31_labels: Sequence[str],
    calibration: Dict[str, object] | None,
    min_margin: float,
) -> Tuple[bool, float]:
    if candidate_label is None:
        return False, 0.0
    margin_required = float(max(0.0, min_margin))
    if margin_required <= 0.0:
        return True, float("inf")

    groups = _confusable_groups(calibration)
    confusing = groups.get(str(candidate_label), ())
    if not confusing:
        return True, float("inf")

    candidate_score = _command_prob_for_label(command_probs, command31_labels, str(candidate_label))
    if candidate_score <= 0.0:
        return False, -1.0
    max_conf = max((_command_prob_for_label(command_probs, command31_labels, label) for label in confusing), default=0.0)
    margin = float(candidate_score - max_conf)
    return margin >= margin_required, margin


def aggregate_command_probs_to_kws12(command_probs: np.ndarray, command31_labels: Sequence[str]) -> np.ndarray:
    kws12_probs = np.zeros(len(KWS12_LABELS), dtype=np.float32)
    for idx, label in enumerate(command31_labels):
        kws12_probs[command31_to_kws12(label)] += float(command_probs[idx])
    return kws12_probs


def resolve_display_candidate(
    wheel: str,
    command31_labels: Sequence[str],
    command_probs: np.ndarray,
    gate_open: bool,
) -> Tuple[str | None, str, float]:
    if not gate_open:
        return None, "UNKNOWN", 0.0

    if wheel == "command31":
        idx = int(np.argmax(command_probs))
        label = str(command31_labels[idx])
        return label, label, float(command_probs[idx])

    kws12_probs = aggregate_command_probs_to_kws12(command_probs, command31_labels)
    idx = int(np.argmax(kws12_probs))
    label = KWS12_LABELS[idx]
    conf = float(kws12_probs[idx])

    if wheel == "target10":
        if label in TARGET_KEYWORDS_10:
            return label, label, conf
        return None, "UNKNOWN", conf

    if label in TARGET_KEYWORDS_10:
        return label, label, conf
    return None, label.upper(), conf


def _sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _drain_audio_queue(q: "queue.Queue[np.ndarray]") -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


def _compute_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def compute_auto_gain(rms: float, target_rms: float, max_gain_db: float, enabled: bool) -> Tuple[float, float]:
    if (not enabled) or rms <= 1e-8:
        return 1.0, 0.0

    target_rms = max(1e-5, float(target_rms))
    max_gain_lin = 10.0 ** (float(max_gain_db) / 20.0)
    desired = target_rms / float(rms)
    gain = min(max_gain_lin, max(1.0, desired))
    gain_db = float(20.0 * np.log10(max(1e-8, gain)))
    return float(gain), gain_db


def _mic_state_label(state: str) -> str:
    return state.replace("_", " ")


def compute_precheck_threshold(median_rms: float, peak95: float, min_rms: float) -> float:
    floor = max(float(min_rms), 1e-4)
    adaptive = min(0.003, max(peak95 * 0.12, median_rms * 0.9))
    return float(max(floor, adaptive))


def classify_precheck_signal(
    median_rms: float,
    peak95: float,
    min_rms: float,
    *,
    permission_peak_eps: float = 1e-5,
) -> Tuple[str, float]:
    threshold = compute_precheck_threshold(median_rms, peak95, min_rms)
    if peak95 <= permission_peak_eps:
        return MIC_PERMISSION_DENIED, threshold
    if median_rms >= threshold or peak95 >= max(0.006, threshold * 6.0):
        return MIC_RUNNING, threshold
    return MIC_NO_SIGNAL, threshold


def _classify_stream_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "permission" in msg or "not permitted" in msg or "denied" in msg:
        return MIC_PERMISSION_DENIED
    return "STREAM_ERROR"


def _open_mic_privacy_settings() -> None:
    # macOS deep-link to microphone privacy settings.
    url = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
    try:
        subprocess.run(["open", url], check=False)
    except Exception:
        pass


class InferenceWorker(threading.Thread):
    def __init__(
        self,
        *,
        q: "queue.Queue[np.ndarray]",
        stop: threading.Event,
        lock: threading.Lock,
        snapshot_ref: List[DemoSnapshot | None],
        model: torch.nn.Module,
        frontend: MelFrontend,
        device: torch.device,
        command31_labels: Sequence[str],
        wheel: str,
        gate: GateStateMachine,
        hop_seconds: float,
        ema_alpha: float,
        hold_ms: float,
        selected_device_label: str,
        input_device_name: str,
        stream_sample_rate: float,
        model_sample_rate: int,
        audio_seconds: float,
        mic_precheck_seconds: float,
        mic_min_rms: float,
        auto_gain: bool,
        target_rms: float,
        max_gain_db: float,
        display_conf_thr: float,
        display_wake_thr: float,
        vote_window: int,
        vote_min_count: int,
        reset_precheck_event: threading.Event,
        passive_profile: PassiveKeywordProfile | None,
        keyword_calibration: Dict[str, object] | None,
    ) -> None:
        super().__init__(daemon=True)
        self.q = q
        self.stop = stop
        self.lock = lock
        self.snapshot_ref = snapshot_ref
        self.model = model
        self.frontend = frontend
        self.device = device
        self.command31_labels = list(command31_labels)
        self.wheel = wheel
        self.gate = gate
        self.hop_seconds = float(hop_seconds)
        self.ema_alpha = float(ema_alpha)
        self.hold_s = max(0.0, float(hold_ms) / 1000.0)
        self.selected_device_label = selected_device_label
        self.input_device_name = str(input_device_name)
        self.stream_sample_rate = float(max(1.0, stream_sample_rate))
        self.model_sample_rate = int(max(1, model_sample_rate))
        self.audio_seconds = float(max(0.1, audio_seconds))

        self.mic_precheck_seconds = float(max(0.0, mic_precheck_seconds))
        self.mic_min_rms = float(max(1e-6, mic_min_rms))
        self.auto_gain = bool(auto_gain)
        self.target_rms = float(max(1e-6, target_rms))
        self.max_gain_db = float(max(0.0, max_gain_db))
        self.display_conf_thr = float(max(0.0, display_conf_thr))
        self.display_wake_thr = float(max(0.0, display_wake_thr))
        self.reset_precheck_event = reset_precheck_event
        self.passive_profile = passive_profile
        self.keyword_calibration = dict(keyword_calibration or {})
        self.default_vote_window = int(vote_window)
        self.default_vote_min_count = int(vote_min_count)

        self._stream_clip_samples = int(round(self.audio_seconds * self.stream_sample_rate))
        self._model_clip_samples = int(round(self.audio_seconds * float(self.model_sample_rate)))
        self._buffer = AudioRingBuffer(self._stream_clip_samples)
        self._last_infer_at = 0.0
        self._ema_probs: Optional[np.ndarray] = None
        self._ema_wake: Optional[float] = None
        self._label_smoother = TemporalLabelSmoother(
            window_size=int(vote_window),
            min_count=int(vote_min_count),
            hold_seconds=self.hold_s,
            max_window_size=max(int(vote_window), 6),
        )
        self._highlight_preview = HighlightPreviewState()
        self._resample_needed = int(round(self.stream_sample_rate)) != int(self.model_sample_rate)
        self._resampler = (
            ta_transforms.Resample(orig_freq=int(round(self.stream_sample_rate)), new_freq=self.model_sample_rate)
            if self._resample_needed
            else None
        )

        self._precheck_started_at = time.monotonic()
        self._precheck_rms_hist: deque[float] = deque(maxlen=256)
        self._precheck_peak_hist: deque[float] = deque(maxlen=256)
        self._precheck_passed = False
        self._mic_state = MIC_CHECK
        self._mic_rms = 0.0
        self._input_gain_db = 0.0
        self._is_clipping = False
        self._precheck_threshold = self.mic_min_rms
        self._last_active_label: str | None = None
        self._active_label_streak = 0
        self._last_profile_update_at: Dict[str, float] = {}

    def _reset_precheck(self, now: float) -> None:
        self._precheck_started_at = now
        self._precheck_rms_hist.clear()
        self._precheck_peak_hist.clear()
        self._precheck_passed = False
        self._mic_state = MIC_CHECK
        self._label_smoother.reset()
        self._highlight_preview.reset()
        self._precheck_threshold = self.mic_min_rms

    def _update_precheck(self, now: float, chunk_rms: float, chunk_peak: float) -> None:
        self._precheck_rms_hist.append(float(chunk_rms))
        self._precheck_peak_hist.append(float(chunk_peak))

        if self._precheck_passed:
            self._mic_state = MIC_RUNNING
            return

        elapsed = now - self._precheck_started_at
        if elapsed < self.mic_precheck_seconds:
            self._mic_state = MIC_CHECK
            return

        med = float(np.median(np.asarray(self._precheck_rms_hist, dtype=np.float64))) if self._precheck_rms_hist else 0.0
        peak95 = (
            float(np.percentile(np.asarray(self._precheck_peak_hist, dtype=np.float64), 95.0))
            if self._precheck_peak_hist
            else 0.0
        )
        state, threshold = classify_precheck_signal(med, peak95, self.mic_min_rms)
        self._precheck_threshold = float(threshold)
        if state == MIC_RUNNING:
            self._precheck_passed = True
            self._mic_state = MIC_RUNNING
        else:
            self._mic_state = state

    def _publish_snapshot(
        self,
        *,
        now_wall: float,
        gate_open: bool,
        gate_state: str,
        command_label: str,
        display_label: str,
        active_label: str | None,
        highlight_label: str | None,
        command_conf: float,
        wake_prob: float,
        wake_open_thr: float,
        wake_close_thr: float,
        latency_ms: float,
        prompt_status: str,
        error_message: str = "",
    ) -> None:
        qmax = self.q.maxsize if self.q.maxsize > 0 else 1
        queue_fill_ratio = min(1.0, float(self.q.qsize()) / float(qmax))

        snapshot = DemoSnapshot(
            updated_at=now_wall,
            gate_open=gate_open,
            gate_state=gate_state,
            command_label=command_label,
            display_label=display_label,
            active_label=active_label,
            highlight_label=highlight_label,
            command_conf=float(command_conf),
            wake_prob=float(wake_prob),
            wake_open_thr=float(wake_open_thr),
            wake_close_thr=float(wake_close_thr),
            latency_ms=float(latency_ms),
            prompt_word="",
            prompt_status=prompt_status,
            device=str(self.device),
            selected_device=self.selected_device_label,
            queue_fill_ratio=queue_fill_ratio,
            mic_state=self._mic_state,
            mic_rms=float(self._mic_rms),
            input_gain_db=float(self._input_gain_db),
            is_clipping=bool(self._is_clipping),
            precheck_passed=bool(self._precheck_passed),
            precheck_threshold=float(self._precheck_threshold),
            stream_sample_rate=float(self.stream_sample_rate),
            input_device_name=self.input_device_name,
            error_message=str(error_message),
        )
        with self.lock:
            self.snapshot_ref[0] = snapshot

    def _publish_runtime_error(self, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"
        self._mic_state = MIC_RUNTIME_ERROR
        self._publish_snapshot(
            now_wall=time.time(),
            gate_open=False,
            gate_state="error",
            command_label="silence",
            display_label="ERROR",
            active_label=None,
            highlight_label=None,
            command_conf=0.0,
            wake_prob=0.0,
            wake_open_thr=self.gate.open_threshold,
            wake_close_thr=self.gate.close_threshold,
            latency_ms=0.0,
            prompt_status=MIC_RUNTIME_ERROR,
            error_message=message,
        )

    def run(self) -> None:
        try:
            while not self.stop.is_set():
                if self.reset_precheck_event.is_set():
                    self.reset_precheck_event.clear()
                    self._reset_precheck(time.monotonic())

                try:
                    chunk = self.q.get(timeout=0.1)
                except queue.Empty:
                    continue

                chunk = np.asarray(chunk, dtype=np.float32)
                chunk_rms = _compute_rms(chunk)
                chunk_peak = float(np.max(np.abs(chunk))) if chunk.size > 0 else 0.0
                self._mic_rms = chunk_rms

                self._buffer.append(chunk)

                now = time.monotonic()
                self._update_precheck(now, chunk_rms, chunk_peak)

                if not self._precheck_passed:
                    status = self._mic_state if self._mic_state != MIC_CHECK else MIC_CHECK
                    display = "MIC CHECK" if status == MIC_CHECK else _mic_state_label(status)
                    self._publish_snapshot(
                        now_wall=time.time(),
                        gate_open=False,
                        gate_state="mic_check",
                        command_label="silence",
                        display_label=display,
                        active_label=None,
                        highlight_label=None,
                        command_conf=0.0,
                        wake_prob=0.0,
                        wake_open_thr=self.gate.open_threshold,
                        wake_close_thr=self.gate.close_threshold,
                        latency_ms=0.0,
                        prompt_status=status,
                    )
                    continue

                if not self._buffer.is_ready:
                    continue

                if (now - self._last_infer_at) < self.hop_seconds:
                    continue
                self._last_infer_at = now

                raw_waveform_np = self._buffer.latest()
                self._mic_rms = _compute_rms(raw_waveform_np)
                gain_lin, gain_db = compute_auto_gain(self._mic_rms, self.target_rms, self.max_gain_db, self.auto_gain)
                self._input_gain_db = gain_db
                waveform_np = raw_waveform_np.copy()
                waveform_np *= float(gain_lin)
                self._is_clipping = bool(np.max(np.abs(waveform_np)) >= 0.999)
                np.clip(waveform_np, -1.0, 1.0, out=waveform_np)

                waveform = torch.from_numpy(waveform_np)
                if self._resample_needed:
                    waveform = self._resampler(waveform.unsqueeze(0)).squeeze(0)
                waveform = pad_or_trim(waveform, target_samples=self._model_clip_samples)
                feature = self.frontend(waveform)
                mean = feature.mean()
                std = feature.std().clamp(min=1e-5)
                feature = (feature - mean) / std
                x = feature.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    t0 = time.perf_counter()
                    out = self.model(x)
                    _sync_device(self.device)
                    t1 = time.perf_counter()

                    probs = torch.softmax(out.command_logits, dim=-1).squeeze(0).detach().cpu().numpy()
                    wake_prob = float(torch.sigmoid(out.wake_logits).squeeze(0).item())
                    embedding = out.embedding.squeeze(0).detach().cpu()

                self._ema_probs = ema_update(self._ema_probs, probs, self.ema_alpha)
                self._ema_wake = ema_update_scalar(self._ema_wake, wake_prob, self.ema_alpha)

                command_idx = int(np.argmax(self._ema_probs))
                command_label = self.command31_labels[command_idx]
                raw_command_conf = float(np.max(self._ema_probs))
                wake_prob_s = float(self._ema_wake)

                gate_open, gate_state, open_thr, close_thr = self.gate.update(
                    now=now,
                    wake_prob=wake_prob_s,
                    command_conf=raw_command_conf,
                )

                raw_active_label, fallback_display_label, decision_conf = resolve_display_candidate(
                    self.wheel,
                    self.command31_labels,
                    self._ema_probs,
                    gate_open,
                )
                keyword_conf_thr, vote_window, vote_min_count, prototype_bonus_cap, min_margin, highlight_hold_ms = _keyword_runtime_params(
                    self.keyword_calibration,
                    raw_active_label,
                    default_conf_thr=self.display_conf_thr,
                    default_vote_window=self.default_vote_window,
                    default_vote_min_count=self.default_vote_min_count,
                )
                prototype_similarity = 0.0
                if raw_active_label in TARGET_KEYWORDS_10 and self.passive_profile is not None:
                    prototype_similarity = self.passive_profile.similarity(raw_active_label, embedding)
                decision_conf = blend_keyword_score(
                    decision_conf,
                    wake_prob_s,
                    prototype_similarity,
                    prototype_bonus_cap=prototype_bonus_cap,
                )
                passes_margin, _margin = _passes_confusion_guardrail(
                    candidate_label=raw_active_label,
                    command_probs=self._ema_probs,
                    command31_labels=self.command31_labels,
                    calibration=self.keyword_calibration,
                    min_margin=min_margin,
                )
                accepted_label = _accept_display_candidate(
                    gate_open=gate_open,
                    candidate_label=raw_active_label if passes_margin else None,
                    command_conf=decision_conf,
                    wake_prob=wake_prob_s,
                    min_command_conf=keyword_conf_thr,
                    min_wake_prob=self.display_wake_thr,
                )
                active_label = self._label_smoother.update(
                    now=now,
                    candidate_label=accepted_label,
                    min_count_override=vote_min_count,
                    window_size_override=vote_window,
                )
                display_label = active_label if active_label is not None else fallback_display_label
                preview_candidate = _accept_display_candidate(
                    gate_open=gate_open,
                    candidate_label=raw_active_label if passes_margin else None,
                    command_conf=decision_conf,
                    wake_prob=wake_prob_s,
                    min_command_conf=keyword_conf_thr,
                    min_wake_prob=self.display_wake_thr,
                )
                highlight_label = self._highlight_preview.update(
                    now=now,
                    candidate_label=active_label if active_label is not None else preview_candidate,
                    hold_seconds=float(highlight_hold_ms) / 1000.0,
                )

                if active_label is not None and active_label == self._last_active_label:
                    self._active_label_streak += 1
                elif active_label is not None:
                    self._last_active_label = active_label
                    self._active_label_streak = 1
                else:
                    self._last_active_label = None
                    self._active_label_streak = 0

                if (
                    self.passive_profile is not None
                    and active_label in TARGET_KEYWORDS_10
                    and wake_prob_s >= 0.92
                    and decision_conf >= 0.85
                    and self._active_label_streak >= 3
                ):
                    last_update = self._last_profile_update_at.get(active_label, 0.0)
                    if (now - last_update) >= 1.0:
                        self.passive_profile.update(active_label, embedding)
                        self._last_profile_update_at[active_label] = now

                prompt_status = "MATCH" if active_label is not None else "LISTENING"

                self._publish_snapshot(
                    now_wall=time.time(),
                    gate_open=gate_open,
                    gate_state=gate_state,
                    command_label=command_label,
                    display_label=display_label,
                    active_label=active_label,
                    highlight_label=highlight_label,
                    command_conf=decision_conf,
                    wake_prob=wake_prob_s,
                    wake_open_thr=float(open_thr),
                    wake_close_thr=float(close_thr),
                    latency_ms=(t1 - t0) * 1000.0,
                    prompt_status=prompt_status,
                )
        except Exception as exc:
            self.stop.set()
            self._publish_runtime_error(exc)


def _list_audio_devices() -> None:
    if sd is None:
        raise RuntimeError("sounddevice is not installed.")
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        name = dev.get("name", "")
        in_ch = dev.get("max_input_channels", 0)
        out_ch = dev.get("max_output_channels", 0)
        default_sr = dev.get("default_samplerate", "")
        print(f"[{idx:>2}] in={in_ch} out={out_ch} sr={default_sr}  {name}")


def _find_demo_checkpoint_candidates(outputs_root: str | Path = "outputs", limit: int = 8) -> List[Path]:
    root = Path(outputs_root).expanduser().resolve()
    candidates: List[Path] = []
    if not root.exists():
        return candidates
    for run_dir in sorted(root.glob("*")):
        if not run_dir.is_dir():
            continue
        best_kws = run_dir / "best_kws12.pt"
        best_wake = run_dir / "best_wake_frr.pt"
        if best_kws.exists():
            candidates.append(best_kws.resolve())
        elif best_wake.exists():
            candidates.append(best_wake.resolve())
        if len(candidates) >= limit:
            break
    return candidates


def _checkpoint_error_message(
    attempted: Path,
    *,
    used_auto: bool,
    details: str = "",
    outputs_root: str | Path = "outputs",
) -> str:
    lines = [
        f"Could not resolve/load checkpoint: {attempted}",
        f"checkpoint_mode={'auto' if used_auto else 'explicit'}",
    ]
    if details:
        lines.append(f"details={details}")
    candidates = _find_demo_checkpoint_candidates(outputs_root=outputs_root)
    if candidates:
        lines.append("candidate_checkpoints:")
        for c in candidates:
            lines.append(f"  - {c}")
    else:
        lines.append("candidate_checkpoints: none found under outputs/*")
    return "\n".join(lines)


def _resolve_gate_args(args: argparse.Namespace) -> Tuple[str, float, float]:
    mode = str(args.gate_mode).lower().strip()
    open_thr = float(args.wake_open_thr)
    close_thr = float(args.wake_close_thr)

    if args.threshold is not None:
        mode = "fixed"
        open_thr = float(args.threshold)
        close_thr = float(args.threshold)

    if close_thr > open_thr:
        close_thr = open_thr
    return mode, open_thr, close_thr


def _runtime_auto_candidates() -> List[str]:
    names = ["cpu"]
    if torch.backends.mps.is_available():
        names.append("mps")
    if torch.cuda.is_available():
        names.append("cuda")
    return names


def _benchmark_runtime_device(checkpoint: Dict[str, object], device_name: str, iters: int) -> float:
    if device_name == "mps" and not torch.backends.mps.is_available():
        return float("inf")
    if device_name == "cuda" and not torch.cuda.is_available():
        return float("inf")
    device = torch.device(device_name)
    return float(benchmark_latency_ms(checkpoint, device=device, iters=max(3, int(iters))))


def _resolve_runtime_device(
    *,
    preferred: str,
    checkpoint: Dict[str, object],
    benchmark_iters: int,
) -> Tuple[torch.device, str, Dict[str, float]]:
    preferred = preferred.lower().strip()
    if preferred != "auto":
        device = pick_device(preferred)
        return device, str(device), {}

    timings: Dict[str, float] = {}
    for name in _runtime_auto_candidates():
        try:
            timings[name] = float(_benchmark_runtime_device(checkpoint, name, benchmark_iters))
        except Exception:
            timings[name] = float("inf")

    finite = {k: v for k, v in timings.items() if np.isfinite(v)}
    if finite:
        best_name = min(finite, key=finite.get)
        return torch.device(best_name), f"auto->{best_name}", timings

    fallback = pick_device("auto")
    return fallback, f"auto->{fallback.type}", timings


def _default_input_index(devices: Sequence[dict]) -> int | None:
    default_device = getattr(sd, "default", None)
    default_pair = getattr(default_device, "device", None)
    if isinstance(default_pair, (list, tuple)) and len(default_pair) >= 1:
        idx = int(default_pair[0])
        if 0 <= idx < len(devices) and int(devices[idx].get("max_input_channels", 0)) > 0:
            return idx
    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0)) > 0:
            return idx
    return None


def _resolve_audio_input_spec(audio_device: str) -> MicPrecheckResult | AudioInputSpec:
    if sd is None:
        return MicPrecheckResult(state=MIC_NO_DEVICE, message="sounddevice not installed", rms=0.0, peak=0.0, passed=False)

    devices = sd.query_devices()
    default_idx = _default_input_index(devices)
    if default_idx is None:
        return MicPrecheckResult(state=MIC_NO_DEVICE, message="No input device detected", rms=0.0, peak=0.0, passed=False)

    if not audio_device:
        dev = devices[default_idx]
        return AudioInputSpec(
            index=int(default_idx),
            name=str(dev.get("name", f"input-{default_idx}")),
            default_samplerate=float(dev.get("default_samplerate", 16000.0)),
        )

    try:
        idx = int(audio_device)
        if idx < 0 or idx >= len(devices):
            return MicPrecheckResult(state=MIC_NO_DEVICE, message=f"Audio device id out of range: {idx}", rms=0.0, peak=0.0, passed=False)
        dev = devices[idx]
        if int(dev.get("max_input_channels", 0)) <= 0:
            return MicPrecheckResult(state=MIC_NO_DEVICE, message=f"Device {idx} has no input channels", rms=0.0, peak=0.0, passed=False)
        return AudioInputSpec(
            index=int(idx),
            name=str(dev.get("name", f"input-{idx}")),
            default_samplerate=float(dev.get("default_samplerate", 16000.0)),
        )
    except ValueError:
        name = audio_device.strip().lower()
        for idx, dev in enumerate(devices):
            if int(dev.get("max_input_channels", 0)) <= 0:
                continue
            if name in str(dev.get("name", "")).lower():
                return AudioInputSpec(
                    index=int(idx),
                    name=str(dev.get("name", f"input-{idx}")),
                    default_samplerate=float(dev.get("default_samplerate", 16000.0)),
                )
        return MicPrecheckResult(state=MIC_NO_DEVICE, message=f"No input device matches: {audio_device}", rms=0.0, peak=0.0, passed=False)


def _resolve_stream_device(audio_device: str) -> MicPrecheckResult | int:
    resolved = _resolve_audio_input_spec(audio_device)
    if isinstance(resolved, MicPrecheckResult):
        return resolved
    return int(resolved.index)


def main() -> None:
    args = parse_args()

    if args.list_audio_devices:
        _list_audio_devices()
        return

    if sd is None:
        raise RuntimeError("sounddevice is not installed. Install it to run realtime demo.")

    instance_lock = DemoInstanceLock()
    instance_lock.acquire()

    passive_profile: PassiveKeywordProfile | None = None
    worker: InferenceWorker | None = None

    try:
        ckpt_arg = args.checkpoint.strip()
        used_auto = ckpt_arg.lower() == "auto"
        ranked_runtime_device = ""
        if used_auto:
            try:
                ckpt_path, ranked_runtime_device = select_best_checkpoint(
                    outputs_root="outputs",
                    device=args.device,
                    benchmark_iters=int(args.ranking_iters),
                    use_cache=not args.no_cache_ranking,
                    rebuild=bool(args.rebuild_ranking),
                )
            except Exception as exc:
                msg = _checkpoint_error_message(
                    Path("outputs").resolve(),
                    used_auto=True,
                    details=str(exc),
                    outputs_root="outputs",
                )
                raise RuntimeError(msg) from exc
        else:
            ckpt_path = Path(ckpt_arg).expanduser().resolve()

        if not ckpt_path.exists():
            msg = _checkpoint_error_message(ckpt_path, used_auto=used_auto, outputs_root="outputs")
            raise FileNotFoundError(msg)

        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            msg = _checkpoint_error_message(
                ckpt_path,
                used_auto=used_auto,
                details=str(exc),
                outputs_root="outputs",
            )
            raise RuntimeError(msg) from exc
        cfg = checkpoint["config"]
        keyword_calibration = load_keyword_calibration(ckpt_path.parent / "keyword_calibration.json")
        if not keyword_calibration and isinstance(checkpoint.get("keyword_calibration"), dict):
            keyword_calibration = dict(checkpoint["keyword_calibration"])

        if used_auto and ranked_runtime_device:
            device, selected_device_label, _timings = _resolve_runtime_device(
                preferred=ranked_runtime_device,
                checkpoint=checkpoint,
                benchmark_iters=int(args.device_auto_bench_iters),
            )
            selected_device_label = f"ranked->{device.type}"
        else:
            device, selected_device_label, _timings = _resolve_runtime_device(
                preferred=args.device,
                checkpoint=checkpoint,
                benchmark_iters=int(args.device_auto_bench_iters),
            )

        tuning = get_sensitivity_tuning(args.sensitivity_profile)
        cmd_conf_thr = float(args.cmd_conf_thr) if args.cmd_conf_thr is not None else tuning.cmd_conf_thr
        display_conf_thr = float(args.display_conf_thr) if args.display_conf_thr is not None else tuning.display_conf_thr
        display_wake_thr = float(args.display_wake_thr) if args.display_wake_thr is not None else tuning.display_wake_thr
        vote_window = int(args.vote_window) if args.vote_window is not None else tuning.vote_window
        vote_min_count = int(args.vote_min_count) if args.vote_min_count is not None else tuning.vote_min_count

        adaptive_cfg = AdaptiveGateConfig(
            calibration_seconds=float(args.calibration_seconds),
            open_offset=tuning.open_offset,
            close_offset=tuning.close_offset,
            open_floor=tuning.open_floor,
            close_floor=tuning.close_floor,
        )

        model = create_model(
            cfg["model"],
            n_mels=int(cfg["features"]["n_mels"]),
            num_commands=len(checkpoint["label_set"]),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()

        frontend = MelFrontend(
            sample_rate=int(cfg["features"]["sample_rate"]),
            n_fft=int(cfg["features"]["n_fft"]),
            hop_length=int(cfg["features"]["hop_length"]),
            n_mels=int(cfg["features"]["n_mels"]),
            f_min=float(cfg["features"].get("f_min", 20.0)),
            f_max=float(cfg["features"].get("f_max", 7600.0)),
        )

        gate_mode, wake_open_thr, wake_close_thr = _resolve_gate_args(args)

        command31_labels = checkpoint.get("label_set", COMMAND31_LABELS)
        wheel_labels = _wheel_labels(args.wheel, command31_labels)

        q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=16)
        stop = threading.Event()
        lock = threading.Lock()
        reset_precheck = threading.Event()
        snapshot_ref: List[DemoSnapshot | None] = [None]

        gate = GateStateMachine(
            mode=gate_mode,
            open_threshold=wake_open_thr,
            close_threshold=wake_close_thr,
            cmd_conf_threshold=cmd_conf_thr,
            hold_seconds=float(args.hold_ms) / 1000.0,
            adaptive=adaptive_cfg,
        )

        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.0, 8.7), dpi=120)
        fig.subplots_adjust(top=0.88, bottom=0.10, left=0.06, right=0.94)
        ax.set_position([0.08, 0.12, 0.84, 0.70])
        apply_theme(fig, ax)

        if args.wheel == "command31":
            wheel_style = {"fontsize": 7, "labeldistance": 0.90, "ring_width": 0.34, "abbreviate": True}
        elif args.wheel == "target10":
            wheel_style = {"fontsize": 11, "labeldistance": 0.76, "ring_width": 0.42, "abbreviate": False}
        else:  # kws12
            wheel_style = {"fontsize": 10, "labeldistance": 0.80, "ring_width": 0.40, "abbreviate": False}

        artists = build_wheel(
            ax,
            wheel_labels,
            fontsize=int(wheel_style["fontsize"]),
            labeldistance=float(wheel_style["labeldistance"]),
            radius=1.0,
            ring_width=float(wheel_style["ring_width"]),
            abbreviate=bool(wheel_style["abbreviate"]),
            place_labels="sector_center",
        )
        hud = create_hud(fig, ax, title_text="Google Speech Dataset Demo")
        ani_ref: Dict[str, object | None] = {"obj": None}
        stream_ref: Dict[str, object | None] = {"obj": None}
        shutdown_state = {"done": False}

        def shutdown(*, close_figure: bool) -> None:
            if not shutdown_state["done"]:
                shutdown_state["done"] = True
                stop.set()

                ani = ani_ref.get("obj")
                event_source = getattr(ani, "event_source", None)
                if event_source is not None:
                    try:
                        event_source.stop()
                    except Exception:
                        pass

                stream = stream_ref.get("obj")
                stream_ref["obj"] = None
                if stream is not None:
                    for method_name in ("abort", "stop", "close"):
                        method = getattr(stream, method_name, None)
                        if callable(method):
                            try:
                                method()
                            except Exception:
                                pass

                _drain_audio_queue(q)

                if worker is not None and worker.is_alive():
                    worker.join(timeout=2.0)

                if passive_profile is not None:
                    passive_profile.close()

            if close_figure:
                try:
                    if plt.fignum_exists(fig.number):
                        plt.close(fig)
                except Exception:
                    pass

        def on_close(_evt) -> None:
            shutdown(close_figure=False)

        def on_key(evt) -> None:
            key = str(getattr(evt, "key", "")).lower()
            if key == "r":
                reset_precheck.set()
            elif key == "p":
                _open_mic_privacy_settings()

        fig.canvas.mpl_connect("close_event", on_close)
        fig.canvas.mpl_connect("key_press_event", on_key)

        input_spec = _resolve_audio_input_spec(args.audio_device)
        if isinstance(input_spec, MicPrecheckResult):
            hud.prompt.set_text(f"[{input_spec.state}]")
            hud.prompt.set_color("#f38ba8")
            hud.center.set_text(_mic_state_label(input_spec.state))
            hud.center.set_color("#f38ba8")
            hud.status.set_text(f"{input_spec.message}  | Press P to open Mic settings")
            try:
                plt.show()
            finally:
                shutdown(close_figure=True)
            return

        stream_sample_rate = float(args.stream_sample_rate) if float(args.stream_sample_rate) > 0.0 else float(input_spec.default_samplerate)
        model_sample_rate = int(cfg["features"]["sample_rate"])
        audio_seconds = float(cfg["features"].get("audio_seconds", CLIP_SECONDS))
        passive_profile = PassiveKeywordProfile(
            path=args.user_profile_path,
            enabled=not bool(args.disable_passive_adaptation),
            max_prototypes=5,
        )
        if args.reset_user_profile:
            passive_profile.reset()

        worker = InferenceWorker(
            q=q,
            stop=stop,
            lock=lock,
            snapshot_ref=snapshot_ref,
            model=model,
            frontend=frontend,
            device=device,
            command31_labels=command31_labels,
            wheel=args.wheel,
            gate=gate,
            hop_seconds=args.hop_seconds,
            ema_alpha=args.ema_alpha,
            hold_ms=args.hold_ms,
            selected_device_label=selected_device_label,
            input_device_name=input_spec.name,
            stream_sample_rate=stream_sample_rate,
            model_sample_rate=model_sample_rate,
            audio_seconds=audio_seconds,
            mic_precheck_seconds=float(args.mic_precheck_seconds),
            mic_min_rms=float(args.mic_min_rms),
            auto_gain=bool(args.auto_gain),
            target_rms=float(args.target_rms),
            max_gain_db=float(args.max_gain_db),
            display_conf_thr=display_conf_thr,
            display_wake_thr=display_wake_thr,
            vote_window=vote_window,
            vote_min_count=vote_min_count,
            reset_precheck_event=reset_precheck,
            passive_profile=passive_profile,
            keyword_calibration=keyword_calibration,
        )
        worker.start()
        render_state: Dict[str, int | None] = {"active_idx": None}
        audio_status_log: Dict[str, float | str] = {"updated_at": 0.0, "message": ""}

        def callback(indata, frames, time_info, status):  # type: ignore[override]
            if stop.is_set():
                return
            if status:
                now = time.monotonic()
                message = str(status)
                # Non-fatal; keep streaming. Throttle repetitive prints to avoid callback overhead.
                if message != audio_status_log["message"] or (now - float(audio_status_log["updated_at"])) >= 1.0:
                    print(f"Audio status: {status}")
                    audio_status_log["message"] = message
                    audio_status_log["updated_at"] = now
            # sounddevice buffers are only valid inside callback; copy before enqueue.
            chunk = np.asarray(indata[:, 0], dtype=np.float32).copy()
            try:
                q.put_nowait(chunk)
            except queue.Full:
                try:
                    _ = q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(chunk)
                except queue.Full:
                    pass

        def render(_frame):
            with lock:
                snap = snapshot_ref[0]

            if snap is None:
                if render_state["active_idx"] is not None:
                    update_wheel(artists, None)
                    render_state["active_idx"] = None
                hud.prompt.set_text("[MIC CHECK]")
                hud.prompt.set_color("#f9e2af")
                hud.center.set_text("MIC CHECK")
                hud.center.set_color("#f9e2af")
                hud.status.set_text(
                    f"mic={MIC_CHECK} input={input_spec.name} sr={stream_sample_rate:.0f}->{model_sample_rate} "
                    f"dev={selected_device_label} | Press R to retry, P for Mic settings"
                )
                return []

            active_idx = resolve_active_index(wheel_labels, snap.highlight_label)
            if render_state["active_idx"] != active_idx:
                update_wheel(artists, active_idx)
                render_state["active_idx"] = active_idx

            if not snap.precheck_passed:
                hud.prompt.set_text(f"[{_mic_state_label(snap.prompt_status)}]")
                hud.prompt.set_color("#f9e2af" if snap.prompt_status == MIC_CHECK else "#f38ba8")
                hud.center.set_text(_mic_state_label(snap.mic_state))
                hud.center.set_color("#f9e2af" if snap.mic_state == MIC_CHECK else "#f38ba8")
                error_suffix = f" error={snap.error_message}" if snap.error_message else ""
                hud.status.set_text(
                    f"mic={snap.mic_state} input={snap.input_device_name} sr={snap.stream_sample_rate:.0f}->{model_sample_rate} "
                    f"rms={snap.mic_rms:.4f} thr={snap.precheck_threshold:.4f} gain={snap.input_gain_db:.1f}dB clip={int(snap.is_clipping)} "
                    f"dev={snap.selected_device} q={snap.queue_fill_ratio:.2f}{error_suffix} | Press R retry, P Mic settings"
                )
                return []

            hud.prompt.set_text(f"[{snap.prompt_status}]")
            if snap.prompt_status == MIC_RUNTIME_ERROR:
                hud.prompt.set_color("#f38ba8")
                hud.center.set_text("ERROR")
                hud.center.set_color("#f38ba8")
            else:
                hud.prompt.set_color("#a6e3a1" if snap.prompt_status == "MATCH" else TEXT)
                hud.center.set_text(snap.display_label.upper() if snap.display_label else "UNKNOWN")
                hud.center.set_color("#ffffff" if snap.gate_open else "#cdd6f4")

            error_suffix = f" error={snap.error_message}" if snap.error_message else ""
            hud.status.set_text(
                f"mic={snap.mic_state} input={snap.input_device_name} sr={snap.stream_sample_rate:.0f}->{model_sample_rate} "
                f"rms={snap.mic_rms:.4f} thr={snap.precheck_threshold:.4f} gain={snap.input_gain_db:.1f}dB clip={int(snap.is_clipping)}  "
                f"wake={snap.wake_prob:.3f} open={snap.wake_open_thr:.2f} close={snap.wake_close_thr:.2f} "
                f"conf={snap.command_conf:.2f} lat={snap.latency_ms:5.1f}ms dev={snap.selected_device} q={snap.queue_fill_ratio:.2f}{error_suffix}"
            )

            if snap.prompt_status == MIC_RUNTIME_ERROR:
                shutdown(close_figure=False)

            return []

        interval_ms = max(10, int(1000.0 / max(1.0, float(args.fps))))
        _ani = animation.FuncAnimation(fig, render, interval=interval_ms, blit=False, cache_frame_data=False)
        ani_ref["obj"] = _ani

        hop_samples = max(1, int(round(float(args.hop_seconds) * stream_sample_rate)))

        stream_kwargs = {
            "samplerate": stream_sample_rate,
            "channels": 1,
            "callback": callback,
            "blocksize": hop_samples,
            "device": int(input_spec.index),
            "dtype": "float32",
        }

        try:
            stream = sd.InputStream(**stream_kwargs)
            stream_ref["obj"] = stream
            stream.start()
            plt.show()
        except Exception as exc:
            shutdown(close_figure=False)
            state = _classify_stream_error(exc)
            hud.prompt.set_text(f"[{state}]")
            hud.prompt.set_color("#f38ba8")
            hud.center.set_text(_mic_state_label(state))
            hud.center.set_color("#f38ba8")
            hud.status.set_text(f"{exc}  | Press P to open Mic settings")
            try:
                plt.show()
            finally:
                shutdown(close_figure=True)
                return
        finally:
            shutdown(close_figure=True)
    finally:
        instance_lock.release()


if __name__ == "__main__":
    main()
