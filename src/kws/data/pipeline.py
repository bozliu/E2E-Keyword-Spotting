"""Dataset preparation and dataloader construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from kws.constants import IGNORE_INDEX, INDEX_TO_COMMAND31, TARGET_KEYWORDS_10
from kws.data.audio import MelFrontend
from kws.data.dataset import KWSSampleDataset, build_collate_fn, load_manifests, parse_augment_config
from kws.data.hi_mia import build_himia_manifests
from kws.data.l2_arctic import build_l2_arctic_eval_manifests
from kws.data.local_speech_commands import build_local_manifests, detect_optional_test_mirror
from kws.data.manifest import ManifestRecord
from kws.data.mswc import build_mswc_manifests
from kws.utils.keyword_focus import DEFAULT_CONFUSION_GROUPS, DEFAULT_CONFUSION_OVERSAMPLE, DEFAULT_KEYWORD_OVERSAMPLE, DEFAULT_WEAK_KEYWORDS


@dataclass
class DataLoaders:
    train: DataLoader
    valid: DataLoader
    test: DataLoader
    stats: Dict[str, object]


def _source_weight(records: List[ManifestRecord]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for rec in records:
        counts[rec.source] = counts.get(rec.source, 0) + 1
    if not counts:
        return {}
    max_count = max(counts.values())
    return {source: max_count / count for source, count in counts.items()}


def _wake_weight(records: List[ManifestRecord]) -> Dict[int, float]:
    counts: Dict[int, int] = {0: 0, 1: 0}
    for rec in records:
        if rec.wake_label in (0, 1):
            counts[int(rec.wake_label)] += 1
    if min(counts.values()) == 0:
        return {0: 1.0, 1: 1.0}
    total = counts[0] + counts[1]
    return {
        0: total / (2 * counts[0]),
        1: total / (2 * counts[1]),
    }


_CONFUSABLE_NEGATIVES = {
    "two",
    "three",
    "tree",
    "wow",
    "one",
    "zero",
    "bird",
    "bed",
    "happy",
    "marvin",
    "sheila",
}


def _keyword_focus_settings(cfg: Dict[str, object]) -> Dict[str, object]:
    training = cfg.get("training", {})
    focus = training.get("keyword_focus", {}) if isinstance(training, dict) else {}
    if not isinstance(focus, dict):
        focus = {}
    return {
        "weak_keywords": [str(x) for x in focus.get("weak_keywords", list(DEFAULT_WEAK_KEYWORDS))],
        "confusion_groups": focus.get("confusion_groups", DEFAULT_CONFUSION_GROUPS),
        "oversample_factor": float(focus.get("oversample_factor", DEFAULT_KEYWORD_OVERSAMPLE)),
        "confusion_factor": float(focus.get("confusion_factor", DEFAULT_CONFUSION_OVERSAMPLE)),
    }


def _command_weight(records: List[ManifestRecord], *, keyword_focus: Dict[str, object]) -> Dict[int, float]:
    weights: Dict[int, float] = {}
    weak_keywords = set(keyword_focus.get("weak_keywords", []))
    confusions = keyword_focus.get("confusion_groups", {})
    confusable_negatives = set(_CONFUSABLE_NEGATIVES)
    if isinstance(confusions, dict):
        for values in confusions.values():
            if isinstance(values, list):
                confusable_negatives.update(str(v) for v in values)
            elif isinstance(values, tuple):
                confusable_negatives.update(str(v) for v in values)
    oversample_factor = float(keyword_focus.get("oversample_factor", DEFAULT_KEYWORD_OVERSAMPLE))
    confusion_factor = float(keyword_focus.get("confusion_factor", DEFAULT_CONFUSION_OVERSAMPLE))
    for rec in records:
        if rec.command_label is None or rec.command_label == IGNORE_INDEX:
            continue
        idx = int(rec.command_label)
        label_name = INDEX_TO_COMMAND31[idx]
        if label_name in weak_keywords:
            weights[idx] = oversample_factor
        elif label_name in TARGET_KEYWORDS_10:
            weights[idx] = 1.15
        elif label_name == "silence":
            weights[idx] = 1.0
        elif label_name in confusable_negatives:
            weights[idx] = confusion_factor
        else:
            weights[idx] = 1.45
    return weights


def _build_sampler(records: List[ManifestRecord], *, keyword_focus: Dict[str, object]) -> WeightedRandomSampler:
    source_w = _source_weight(records)
    wake_w = _wake_weight(records)
    command_w = _command_weight(records, keyword_focus=keyword_focus)
    weights = []
    for rec in records:
        label_idx = int(rec.command_label) if rec.command_label is not None else IGNORE_INDEX
        w = source_w.get(rec.source, 1.0) * wake_w.get(int(rec.wake_label or 0), 1.0) * command_w.get(label_idx, 1.0)
        weights.append(w)
    tensor = torch.tensor(weights, dtype=torch.float32)
    return WeightedRandomSampler(weights=tensor, num_samples=len(records), replacement=True)


def prepare_data(cfg: Dict[str, object], project_root: str | Path) -> Dict[str, object]:
    root = Path(project_root).resolve()
    local_split_root = (root / cfg["data"]["local"]["speech_commands_split"]).resolve()
    manifests_root = (root / cfg["data"]["manifests_dir"]).resolve()
    manifests_root.mkdir(parents=True, exist_ok=True)

    local_manifests = build_local_manifests(
        root=local_split_root,
        output_dir=manifests_root,
        silence_ratio=float(cfg["data"]["local"].get("silence_ratio", 0.1)),
        limit_per_class=cfg["data"]["local"].get("limit_per_class"),
    )

    mirror_root = root / cfg["data"]["local"].get("optional_test_mirror", "")
    mirror_stats = detect_optional_test_mirror(local_manifests["test"], mirror_root)
    external_cfg = cfg["data"].get("external", {})
    if not isinstance(external_cfg, dict):
        external_cfg = {}
    hi_cfg = external_cfg.get("hi_mia", {})
    if not isinstance(hi_cfg, dict):
        hi_cfg = {}

    external_enabled = bool(hi_cfg.get("enabled", False))
    external_stats: Dict[str, int] = {"train": 0, "valid": 0, "test": 0}
    if external_enabled:
        hi_root = (root / str(hi_cfg.get("root", ""))).resolve()
        if not hi_root.exists():
            raise FileNotFoundError(
                f"HI-MIA is enabled but not found at {hi_root}. "
                "Run: python -m kws.data.download_external --dataset hi_mia"
            )
        himia_manifests = build_himia_manifests(
            root=hi_root,
            output_dir=manifests_root,
            limit_per_split=hi_cfg.get("limit_per_split"),
        )
        external_stats = {k: len(v) for k, v in himia_manifests.items()}

    focus_cfg = _keyword_focus_settings(cfg)
    confusable_words = []
    groups = focus_cfg.get("confusion_groups", {})
    if isinstance(groups, dict):
        for values in groups.values():
            if isinstance(values, (list, tuple)):
                confusable_words.extend(str(v) for v in values)
    mswc_cfg = external_cfg.get("mswc", {})
    mswc_stats: Dict[str, int] = {"train": 0, "valid": 0, "test": 0}
    if isinstance(mswc_cfg, dict) and bool(mswc_cfg.get("enabled", False)):
        mswc_root = (root / str(mswc_cfg.get("root", ""))).resolve()
        if not mswc_root.exists():
            raise FileNotFoundError(f"MSWC is enabled but not found at {mswc_root}")
        mswc_manifests = build_mswc_manifests(
            root=mswc_root,
            output_dir=manifests_root,
            target_words=TARGET_KEYWORDS_10,
            confusable_words=confusable_words,
            metadata_file=str(mswc_cfg.get("metadata_file", "")) or None,
            audio_root=str(mswc_cfg.get("audio_root", "")) or None,
            limit_per_word=mswc_cfg.get("limit_per_word"),
        )
        mswc_stats = {split: len(mswc_manifests.get(split, [])) for split in ("train", "valid", "test")}

    l2_cfg = external_cfg.get("l2_arctic_eval", {})
    l2_stats: Dict[str, int] = {"test": 0}
    if isinstance(l2_cfg, dict) and bool(l2_cfg.get("enabled", False)):
        l2_root = (root / str(l2_cfg.get("root", ""))).resolve()
        if not l2_root.exists():
            raise FileNotFoundError(f"L2-ARCTIC eval is enabled but not found at {l2_root}")
        l2_manifests = build_l2_arctic_eval_manifests(
            root=l2_root,
            output_dir=manifests_root,
            target_words=TARGET_KEYWORDS_10,
            confusable_words=confusable_words,
            speaker_metadata_file=str(l2_cfg.get("speaker_metadata_file", "")) or None,
            limit_per_word=l2_cfg.get("limit_per_word"),
        )
        l2_stats = {"test": len(l2_manifests.get("test", []))}

    stats = {
        "local": {k: len(v) for k, v in local_manifests.items()},
        "external_hi_mia": external_stats,
        "external_mswc": mswc_stats,
        "external_l2_arctic_eval": l2_stats,
        "optional_test_mirror": mirror_stats,
    }
    with (manifests_root / "dataset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    return stats


def create_dataloaders(cfg: Dict[str, object], project_root: str | Path) -> DataLoaders:
    root = Path(project_root).resolve()
    manifests_root = (root / cfg["data"]["manifests_dir"]).resolve()

    external_cfg = cfg["data"].get("external", {})
    if not isinstance(external_cfg, dict):
        external_cfg = {}
    hi_cfg = external_cfg.get("hi_mia", {})
    if not isinstance(hi_cfg, dict):
        hi_cfg = {}
    mswc_cfg = external_cfg.get("mswc", {})
    if not isinstance(mswc_cfg, dict):
        mswc_cfg = {}
    use_hi_mia = bool(hi_cfg.get("enabled", False))
    use_mswc = bool(mswc_cfg.get("enabled", False))
    keyword_focus = _keyword_focus_settings(cfg)

    def _records(split: str) -> List[ManifestRecord]:
        paths: List[Path] = [manifests_root / f"local_{split}.jsonl"]
        if use_hi_mia:
            paths.append(manifests_root / f"hi_mia_{split}.jsonl")
        if use_mswc:
            paths.append(manifests_root / f"mswc_{split}.jsonl")
        return load_manifests(paths)

    train_records = _records("train")
    valid_records = _records("valid")
    test_records = _records("test")

    frontend = MelFrontend(
        sample_rate=int(cfg["features"]["sample_rate"]),
        n_fft=int(cfg["features"]["n_fft"]),
        hop_length=int(cfg["features"]["hop_length"]),
        n_mels=int(cfg["features"]["n_mels"]),
        f_min=float(cfg["features"].get("f_min", 20.0)),
        f_max=float(cfg["features"].get("f_max", 7600.0)),
    )

    augment_enabled, augment_cfg = parse_augment_config(cfg["training"].get("augment", True))
    train_collate_fn = build_collate_fn(frontend, specaugment_cfg=augment_cfg if augment_enabled else None)
    eval_collate_fn = build_collate_fn(frontend, specaugment_cfg=None)

    train_dataset = KWSSampleDataset(
        train_records,
        split="train",
        augment=augment_enabled,
        augment_cfg=augment_cfg,
    )
    valid_dataset = KWSSampleDataset(valid_records, split="valid", augment=False, augment_cfg=augment_cfg)
    test_dataset = KWSSampleDataset(test_records, split="test", augment=False, augment_cfg=augment_cfg)

    train_sampler = _build_sampler(train_records, keyword_focus=keyword_focus)

    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"].get("num_workers", 4))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=train_collate_fn,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=eval_collate_fn,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=eval_collate_fn,
        drop_last=False,
    )

    stats = {
        "train_samples": len(train_records),
        "valid_samples": len(valid_records),
        "test_samples": len(test_records),
        "with_hi_mia": use_hi_mia,
        "with_mswc": use_mswc,
    }

    return DataLoaders(train=train_loader, valid=valid_loader, test=test_loader, stats=stats)
