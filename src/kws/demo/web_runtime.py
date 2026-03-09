"""Shared keyword-decision helpers that are safe for browser/web runtimes."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from kws.constants import KWS12_LABELS, TARGET_KEYWORDS_10, command31_to_kws12
from kws.demo.user_profile import blend_keyword_score
from kws.utils.keyword_focus import DEFAULT_FOCUS_RUNTIME_OVERRIDES, DEFAULT_RUNTIME_CONFUSION_GROUPS


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
class AdaptiveGateConfig:
    calibration_seconds: float = 2.0
    open_offset: float = 0.12
    close_offset: float = 0.05
    open_floor: float = 0.25
    close_floor: float = 0.15
    calibration_score_cap: float = 1.0
    max_open_threshold: float = 0.98


@dataclass(frozen=True)
class RuntimeDecision:
    raw_active_label: str | None
    fallback_display_label: str
    accepted_label: str | None
    active_label: str | None
    preview_label: str | None
    highlight_label: str | None
    display_label: str
    command_confidence: float
    raw_command_confidence: float
    preview_confidence: float
    raw_wake_probability: float
    preview_reason: str
    prompt_label: str
    final_label: str
    status_message: str
    passes_confusion_margin: bool
    confusion_margin: float


class AudioRingBuffer:
    """Fixed-size rolling waveform buffer optimized for browser audio chunks."""

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
    """Hold a fast preview label briefly so the wheel feels responsive."""

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
        self._calibration_start = 0.0
        self._calibration_scores: list[float] = []
        self._calibrated = mode == "fixed"

    def reset(self, *, now: float | None = None) -> None:
        self.state = "closed"
        self._below_close_frames = 0
        self._hold_until = 0.0
        self._calibration_start = float(now or 0.0)
        self._calibration_scores = []
        self._calibrated = self.mode == "fixed"

    def _finish_calibration(self, now: float) -> None:
        if self.mode != "adaptive" or self._calibrated:
            return
        if (now - self._calibration_start) < self.adaptive.calibration_seconds:
            return
        if not self._calibration_scores:
            self._calibrated = True
            self.state = "closed"
            self._below_close_frames = 0
            return

        p95 = float(np.clip(np.percentile(np.asarray(self._calibration_scores, dtype=np.float64), 95.0), 0.0, 1.0))
        max_open_thr = float(np.clip(self.adaptive.max_open_threshold, 0.05, 0.99))
        open_thr = min(max(p95 + self.adaptive.open_offset, self.adaptive.open_floor), max_open_thr)
        close_ceiling = max(0.05, open_thr - 0.02)
        close_thr = min(max(p95 + self.adaptive.close_offset, self.adaptive.close_floor), close_ceiling)
        if close_thr >= open_thr:
            close_thr = max(0.05, open_thr - 0.02)

        self.open_threshold = float(open_thr)
        self.close_threshold = float(close_thr)
        self._calibrated = True
        self.state = "closed"
        self._below_close_frames = 0

    def update(
        self,
        *,
        now: float,
        wake_prob: float,
        command_conf: float,
        calibration_wake_prob: float | None = None,
    ) -> Tuple[bool, str, float, float]:
        if self._calibration_start == 0.0:
            self._calibration_start = now

        if self.mode == "adaptive" and not self._calibrated:
            if calibration_wake_prob is not None:
                capped_wake = min(
                    max(0.0, float(calibration_wake_prob)),
                    float(max(0.0, self.adaptive.calibration_score_cap)),
                )
                self._calibration_scores.append(capped_wake)
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


def ema_update(prev: Optional[np.ndarray], current: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return current
    return alpha * current + (1.0 - alpha) * prev


def ema_update_scalar(prev: Optional[float], current: float, alpha: float) -> float:
    if prev is None:
        return current
    return alpha * current + (1.0 - alpha) * prev


def compute_rms(x: np.ndarray) -> float:
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
        return "PERMISSION_DENIED", threshold
    if median_rms >= threshold or peak95 >= max(0.006, threshold * 6.0):
        return "RUNNING", threshold
    return "NO_SIGNAL", threshold


def wheel_labels(wheel: str, command31_labels: Sequence[str]) -> list[str]:
    if wheel == "command31":
        return list(command31_labels)
    if wheel == "target10":
        return list(TARGET_KEYWORDS_10)
    return ["silence", "unknown", *TARGET_KEYWORDS_10]


def accept_display_candidate(
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


def _confusable_groups(calibration: Dict[str, object] | None) -> Dict[str, Tuple[str, ...]]:
    payload = calibration or {}
    groups = dict(DEFAULT_RUNTIME_CONFUSION_GROUPS)
    raw = payload.get("confusable_groups", {}) if isinstance(payload.get("confusable_groups", {}), dict) else {}
    for label, values in raw.items():
        if isinstance(values, (list, tuple)):
            groups[str(label)] = tuple(str(v) for v in values if str(v))
    return groups


def keyword_runtime_params(
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


def _command_prob_for_label(command_probs: np.ndarray, command31_labels: Sequence[str], label: str) -> float:
    try:
        idx = list(command31_labels).index(str(label))
    except ValueError:
        return 0.0
    return float(command_probs[idx])


def passes_confusion_guardrail(
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


def resolve_preview_candidate(
    wheel: str,
    command31_labels: Sequence[str],
    command_probs: np.ndarray,
) -> Tuple[str | None, float, str]:
    if wheel == "command31":
        idx = int(np.argmax(command_probs))
        label = str(command31_labels[idx])
        return label, float(command_probs[idx]), "top-command31"

    kws12_probs = aggregate_command_probs_to_kws12(command_probs, command31_labels)
    idx = int(np.argmax(kws12_probs))
    label = KWS12_LABELS[idx]
    conf = float(kws12_probs[idx])
    if wheel == "target10":
        if label in TARGET_KEYWORDS_10:
            return label, conf, "top-target10"
        return None, conf, f"top-{label}"

    if label in TARGET_KEYWORDS_10:
        return label, conf, "top-kws12"
    return None, conf, f"top-{label}"


def resolve_runtime_decision(
    *,
    now: float,
    wheel: str,
    command31_labels: Sequence[str],
    command_probs: np.ndarray,
    gate_open: bool,
    gate_state: str,
    wake_prob: float,
    display_wake_thr: float,
    calibration: Dict[str, object] | None,
    default_conf_thr: float,
    default_vote_window: int,
    default_vote_min_count: int,
    smoother: TemporalLabelSmoother,
    preview: HighlightPreviewState,
    prototype_similarity: float = 0.0,
    preview_requires_gate: bool = True,
    preview_min_command_conf: float | None = None,
    preview_min_wake_prob: float | None = None,
    preview_hold_seconds: float | None = None,
    listening_label: str = "LISTENING",
    listening_message: str = "Listening for a stable keyword match.",
    calibrating_label: str = "CALIBRATING",
    calibrating_message: str = "Calibrating the live gate. Keep speaking naturally.",
    matched_message_template: str = "Detected '{label}' in the live stream.",
) -> RuntimeDecision:
    raw_preview_label, preview_conf, preview_source = resolve_preview_candidate(
        wheel,
        command31_labels,
        command_probs,
    )
    raw_active_label, fallback_display_label, decision_conf = resolve_display_candidate(
        wheel,
        command31_labels,
        command_probs,
        gate_open,
    )
    keyword_conf_thr, vote_window, vote_min_count, prototype_bonus_cap, min_margin, highlight_hold_ms = keyword_runtime_params(
        calibration,
        raw_active_label,
        default_conf_thr=default_conf_thr,
        default_vote_window=default_vote_window,
        default_vote_min_count=default_vote_min_count,
    )
    decision_conf = float(
        blend_keyword_score(
            decision_conf,
            wake_prob,
            prototype_similarity,
            prototype_bonus_cap=prototype_bonus_cap,
        )
    )
    passes_margin, margin = passes_confusion_guardrail(
        candidate_label=raw_active_label,
        command_probs=command_probs,
        command31_labels=command31_labels,
        calibration=calibration,
        min_margin=min_margin,
    )
    accepted_label = accept_display_candidate(
        gate_open=gate_open,
        candidate_label=raw_active_label if passes_margin else None,
        command_conf=decision_conf,
        wake_prob=wake_prob,
        min_command_conf=keyword_conf_thr,
        min_wake_prob=display_wake_thr,
    )
    active_label = smoother.update(
        now=now,
        candidate_label=accepted_label,
        min_count_override=vote_min_count,
        window_size_override=vote_window,
    )
    effective_preview_command_conf = float(
        max(0.0, preview_min_command_conf if preview_min_command_conf is not None else min(default_conf_thr, 0.25))
    )
    effective_preview_wake = float(
        max(0.0, preview_min_wake_prob if preview_min_wake_prob is not None else min(display_wake_thr, 0.25))
    )
    preview_reason = preview_source
    preview_candidate: str | None = raw_preview_label
    if raw_preview_label is None:
        preview_candidate = None
    elif preview_requires_gate and not gate_open:
        preview_candidate = None
        preview_reason = "gate-closed"
    elif wake_prob < effective_preview_wake:
        preview_candidate = None
        preview_reason = "wake-low"
    elif preview_conf < effective_preview_command_conf:
        preview_candidate = None
        preview_reason = "conf-low"
    else:
        preview_reason = "preview-ok"
    highlight_hold_seconds = (
        max(0.0, float(preview_hold_seconds))
        if preview_hold_seconds is not None
        else float(highlight_hold_ms) / 1000.0
    )
    highlight_label = preview.update(
        now=now,
        candidate_label=active_label if active_label is not None else preview_candidate,
        hold_seconds=highlight_hold_seconds,
    )
    display_label = active_label if active_label is not None else fallback_display_label

    if active_label is not None:
        prompt_label = "MATCH"
        final_label = active_label.upper()
        status_message = matched_message_template.format(label=active_label)
    elif gate_state == "calibrating":
        prompt_label = calibrating_label
        final_label = listening_label
        status_message = calibrating_message
    elif gate_open:
        prompt_label = listening_label
        final_label = display_label.upper() if display_label and display_label != "UNKNOWN" else listening_label
        status_message = listening_message
    else:
        prompt_label = listening_label
        final_label = listening_label
        status_message = listening_message

    return RuntimeDecision(
        raw_active_label=raw_active_label,
        fallback_display_label=fallback_display_label,
        accepted_label=accepted_label,
        active_label=active_label,
        preview_label=preview_candidate,
        highlight_label=highlight_label,
        display_label=display_label,
        command_confidence=decision_conf,
        raw_command_confidence=preview_conf,
        preview_confidence=preview_conf,
        raw_wake_probability=wake_prob,
        preview_reason=preview_reason,
        prompt_label=prompt_label,
        final_label=final_label,
        status_message=status_message,
        passes_confusion_margin=passes_margin,
        confusion_margin=margin,
    )
