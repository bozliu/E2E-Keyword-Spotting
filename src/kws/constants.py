"""Project-wide constants and label maps."""

from __future__ import annotations

from typing import Dict, List

SAMPLE_RATE = 16_000
CLIP_SECONDS = 1.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

# Speech Commands V1 30-word setup.
SPEECH_COMMANDS_30: List[str] = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
]

SILENCE_LABEL = "silence"
UNKNOWN_LABEL = "unknown"

COMMAND31_LABELS: List[str] = [SILENCE_LABEL] + SPEECH_COMMANDS_30
COMMAND31_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(COMMAND31_LABELS)}
INDEX_TO_COMMAND31: Dict[int, str] = {idx: name for name, idx in COMMAND31_TO_INDEX.items()}

# Derived KWS12 labels: silence + unknown + target 10 keywords.
TARGET_KEYWORDS_10: List[str] = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]

KWS12_LABELS: List[str] = [SILENCE_LABEL, UNKNOWN_LABEL] + TARGET_KEYWORDS_10
KWS12_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(KWS12_LABELS)}
INDEX_TO_KWS12: Dict[int, str] = {idx: name for name, idx in KWS12_TO_INDEX.items()}

IGNORE_INDEX = -100


def command31_to_kws12(label_name: str) -> int:
    """Map a command31 label name to KWS12 class index."""
    if label_name == SILENCE_LABEL:
        return KWS12_TO_INDEX[SILENCE_LABEL]
    if label_name in TARGET_KEYWORDS_10:
        return KWS12_TO_INDEX[label_name]
    return KWS12_TO_INDEX[UNKNOWN_LABEL]
