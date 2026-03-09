"""Synthetic hard-negative generation for verifier training."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch

from kws.constants import COMMAND31_LABELS, COMMAND31_TO_INDEX, SAMPLE_RATE, TARGET_KEYWORDS_10
from kws.env import ensure_repo_import
from kws.data.manifest import ManifestRecord, write_manifest
from kws.utils.keyword_focus import DEFAULT_RUNTIME_CONFUSION_GROUPS


DEFAULT_SAY_VOICES: tuple[str, ...] = ("Samantha", "Alex", "Daniel", "Karen", "Tessa", "Moira")
DEFAULT_SAY_RATES: tuple[int, ...] = (140, 175, 210)
DEFAULT_SPLIT_RATIO = 0.8


def available_say_voices() -> list[str]:
    if shutil.which("say") is None:
        return []
    proc = subprocess.run(["say", "-v", "?"], check=True, capture_output=True, text=True)
    voices = []
    for line in proc.stdout.splitlines():
        voice = line.strip().split(maxsplit=1)[0].strip()
        if voice:
            voices.append(voice)
    return voices


def resolve_say_voices(preferred: Sequence[str] = DEFAULT_SAY_VOICES) -> list[str]:
    available = set(available_say_voices())
    chosen = [voice for voice in preferred if voice in available]
    if not chosen:
        raise RuntimeError(
            "No preferred macOS 'say' voices are available. "
            f"Preferred={list(preferred)} available={sorted(available)}"
        )
    if shutil.which("afconvert") is None:
        raise RuntimeError("afconvert is required to convert 'say' output into 16 kHz WAV files.")
    return chosen


def _load_keyword_focus(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_confusable_groups(
    source_checkpoint: str | Path,
    *,
    selection_report: str | Path | None = None,
) -> Dict[str, list[str]]:
    ckpt_path = Path(source_checkpoint).expanduser().resolve()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    groups: Dict[str, list[str]] = {key: list(values) for key, values in DEFAULT_RUNTIME_CONFUSION_GROUPS.items()}

    training = cfg.get("training", {}) if isinstance(cfg, Mapping) else {}
    keyword_focus = training.get("keyword_focus", {}) if isinstance(training, Mapping) else {}
    if isinstance(keyword_focus, Mapping):
        raw_groups = keyword_focus.get("confusion_groups", {})
        if isinstance(raw_groups, Mapping):
            for keyword, values in raw_groups.items():
                label = str(keyword).strip()
                if label not in TARGET_KEYWORDS_10 or not isinstance(values, Sequence):
                    continue
                groups.setdefault(label, [])
                for value in values:
                    rival = str(value).strip()
                    if rival and rival not in groups[label]:
                        groups[label].append(rival)

    focus_payload = _load_keyword_focus(ckpt_path.parent / "keyword_focus.json")
    per_keyword = focus_payload.get("per_keyword", {}) if isinstance(focus_payload, Mapping) else {}
    if isinstance(per_keyword, Mapping):
        for keyword, stats in per_keyword.items():
            label = str(keyword).strip()
            if label not in TARGET_KEYWORDS_10 or not isinstance(stats, Mapping):
                continue
            groups.setdefault(label, [])
            for item in stats.get("top_confusions", []):
                if not isinstance(item, Mapping):
                    continue
                rival = str(item.get("label", "")).strip()
                if rival and rival != label and rival not in groups[label]:
                    groups[label].append(rival)

    if selection_report is not None:
        report_path = Path(selection_report).expanduser().resolve()
        if report_path.exists():
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            chosen = payload.get("chosen", {})
            chosen_path = Path(str(chosen.get("checkpoint", ""))).expanduser()
            if chosen_path.exists() and chosen_path.resolve() == ckpt_path:
                for keyword, values in _load_keyword_focus(ckpt_path.parent / "keyword_focus.json").get("confusable_groups", {}).items():
                    label = str(keyword).strip()
                    if label not in TARGET_KEYWORDS_10 or not isinstance(values, Sequence):
                        continue
                    groups.setdefault(label, [])
                    for value in values:
                        rival = str(value).strip()
                        if rival and rival not in groups[label]:
                            groups[label].append(rival)

    normalized: Dict[str, list[str]] = {}
    for keyword in TARGET_KEYWORDS_10:
        merged: list[str] = []
        for rival in groups.get(keyword, []):
            label = str(rival).strip()
            if label and label != keyword and label in COMMAND31_LABELS and label not in merged:
                merged.append(label)
        normalized[keyword] = merged
    return normalized


def _stable_split(stem: str, *, train_ratio: float) -> str:
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
    return "train" if bucket < float(train_ratio) else "valid"


def _synthesize_with_say(output_wav: Path, *, voice: str, rate: int, text: str) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kws_say_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        tmp_aiff = tmpdir_path / "clip.aiff"
        subprocess.run(["say", "-v", voice, "-r", str(int(rate)), "-o", str(tmp_aiff), text], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", str(tmp_aiff), str(output_wav)],
            check=True,
        )


def generate_hard_negative_dataset(
    *,
    output_dir: str | Path,
    manifests_dir: str | Path,
    source_checkpoint: str | Path,
    selection_report: str | Path | None = None,
    voices: Sequence[str] = DEFAULT_SAY_VOICES,
    rates: Sequence[int] = DEFAULT_SAY_RATES,
    overwrite: bool = False,
    train_ratio: float = DEFAULT_SPLIT_RATIO,
) -> Dict[str, List[ManifestRecord]]:
    output_root = Path(output_dir).expanduser().resolve()
    manifests_root = Path(manifests_dir).expanduser().resolve()
    manifests_root.mkdir(parents=True, exist_ok=True)
    chosen_voices = resolve_say_voices(voices)
    confusable_groups = collect_confusable_groups(source_checkpoint, selection_report=selection_report)

    manifests: Dict[str, List[ManifestRecord]] = {"train": [], "valid": []}
    for keyword in TARGET_KEYWORDS_10:
        for rival in confusable_groups.get(keyword, []):
            for voice in chosen_voices:
                for rate in rates:
                    stem = f"{keyword}__reject__{rival}__{voice}__r{int(rate)}"
                    split = _stable_split(stem, train_ratio=train_ratio)
                    wav_path = output_root / split / keyword / f"{stem}.wav"
                    if overwrite or not wav_path.exists():
                        _synthesize_with_say(wav_path, voice=voice, rate=int(rate), text=rival)
                    manifests[split].append(
                        ManifestRecord(
                            path=str(wav_path),
                            source="synthetic_hard_negative",
                            split=split,
                            command_label=COMMAND31_TO_INDEX.get(rival),
                            wake_label=0,
                            sr=SAMPLE_RATE,
                            speaker_id=f"{voice}_r{int(rate)}",
                            transcript=rival,
                            is_synthetic=True,
                            difficulty_bucket="hard_negative",
                        )
                    )

    for split, records in manifests.items():
        write_manifest(records, manifests_root / f"synthetic_hard_negative_{split}.jsonl")

    stats_path = manifests_root / "synthetic_hard_negative_stats.json"
    stats_payload = {
        "source_checkpoint": str(Path(source_checkpoint).expanduser().resolve()),
        "voices": list(chosen_voices),
        "rates": [int(rate) for rate in rates],
        "counts": {split: len(records) for split, records in manifests.items()},
        "confusable_groups": confusable_groups,
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic hard negatives for verifier training")
    parser.add_argument("--source-checkpoint", type=str, required=True)
    parser.add_argument("--selection-report", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="data/synthetic/hard_negatives")
    parser.add_argument("--manifests-dir", type=str, default="data/processed/manifests")
    parser.add_argument("--voice", action="append", default=[])
    parser.add_argument("--rate", action="append", type=int, default=[])
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIO)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_path(project_root: Path, value: str, default: str) -> Path:
    path = Path(value or default).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    ensure_repo_import(repo_root)
    project_root = Path.cwd().resolve()
    source_checkpoint = _resolve_path(project_root, args.source_checkpoint, "")
    selection_report = _resolve_path(project_root, args.selection_report, "") if args.selection_report else None
    output_dir = _resolve_path(project_root, args.output_dir, "data/synthetic/hard_negatives")
    manifests_dir = _resolve_path(project_root, args.manifests_dir, "data/processed/manifests")
    voices = tuple(args.voice) if args.voice else DEFAULT_SAY_VOICES
    rates = tuple(int(rate) for rate in args.rate) if args.rate else DEFAULT_SAY_RATES

    manifests = generate_hard_negative_dataset(
        output_dir=output_dir,
        manifests_dir=manifests_dir,
        source_checkpoint=source_checkpoint,
        selection_report=selection_report,
        voices=voices,
        rates=rates,
        overwrite=args.overwrite,
        train_ratio=args.train_ratio,
    )
    stats = {split: len(records) for split, records in manifests.items()}
    print(
        json.dumps(
            {
                "source_checkpoint": str(source_checkpoint),
                "output_dir": str(output_dir),
                "manifests_dir": str(manifests_dir),
                "counts": stats,
                "voices": list(voices),
                "rates": [int(rate) for rate in rates],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
