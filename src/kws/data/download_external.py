"""Download and restore datasets used by the submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kws.data.hi_mia import build_himia_manifests_with_status, download_hi_mia
from kws.data.l2_arctic import build_l2_arctic_eval_manifests
from kws.data.mswc import build_mswc_manifests
from kws.data.speech_commands import restore_speech_commands_dataset
from kws.env import ensure_repo_import
from kws.utils.keyword_focus import DEFAULT_CONFUSION_GROUPS
from kws.constants import TARGET_KEYWORDS_10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and restore datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["speech_commands", "hi_mia", "mswc", "l2_arctic_eval"])
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--manifests-dir", type=str, default="data/processed/manifests")
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--metadata-file", type=str, default="")
    parser.add_argument("--audio-root", type=str, default="")
    parser.add_argument("--speaker-metadata-file", type=str, default="")
    parser.add_argument("--download-dir", type=str, default="")
    parser.add_argument("--archive", type=str, default="")
    parser.add_argument("--speech-commands-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--silence-ratio", type=float, default=0.10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _resolve_path(project_root: Path, value: str, default: str) -> Path:
    path = Path(value or default).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    ensure_repo_import(repo_root)
    confusable_words = sorted({word for values in DEFAULT_CONFUSION_GROUPS.values() for word in values})
    manifests_dir = _resolve_path(project_root, args.manifests_dir, "data/processed/manifests")

    if args.dataset == "speech_commands":
        target = _resolve_path(project_root, args.root, "data/local/speech_commands_split")
        download_dir = _resolve_path(project_root, args.download_dir, "data/local/_downloads")
        archive_path = _resolve_path(project_root, args.archive, "") if args.archive else None
        restored = restore_speech_commands_dataset(
            split_root=target,
            manifests_dir=manifests_dir,
            version=args.speech_commands_version,
            download_dir=download_dir,
            archive_path=archive_path,
            silence_ratio=args.silence_ratio,
            limit_per_class=args.limit_per_split,
            force=args.force,
        )
        stats = restored["stats"]
        print(f"Restored Speech Commands {args.speech_commands_version} to {target}")
        print(f"Saved manifests to {manifests_dir}")
        print(f"Saved stats to {manifests_dir / 'speech_commands_stats.json'}")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    if args.dataset == "hi_mia":
        target = _resolve_path(project_root, args.root, "data/external/hi_mia")
        print(f"Downloading HI-MIA to {target} ...")
        download_hi_mia(target)

        manifests, status = build_himia_manifests_with_status(
            target,
            manifests_dir,
            limit_per_split=args.limit_per_split,
        )
        stats = {
            "manifest_counts": {split: len(records) for split, records in manifests.items()},
            "reduced_data_mode": bool(status.get("reduced_data_mode", False)),
            "present_source_splits": status.get("present_source_splits", []),
            "source_file_counts": status.get("source_file_counts", {}),
            "build_mode": status.get("build_mode", "official"),
        }
        out = manifests_dir / "hi_mia_stats.json"
        with out.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, ensure_ascii=False)
        print(f"Done. Manifest stats: {stats}")
        print(f"Saved stats to {out}")
        return

    if args.dataset == "mswc":
        target = _resolve_path(project_root, args.root, "data/external/mswc")
        manifests = build_mswc_manifests(
            target,
            manifests_dir,
            target_words=TARGET_KEYWORDS_10,
            confusable_words=confusable_words,
            metadata_file=args.metadata_file or None,
            audio_root=args.audio_root or None,
            limit_per_word=args.limit_per_split,
        )
        stats = {split: len(records) for split, records in manifests.items()}
        out = manifests_dir / "mswc_stats.json"
        with out.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, ensure_ascii=False)
        print(f"Built MSWC manifests from {target}")
        print(f"Saved stats to {out}")
        return

    if args.dataset == "l2_arctic_eval":
        target = _resolve_path(project_root, args.root, "data/external/l2_arctic_eval")
        manifests = build_l2_arctic_eval_manifests(
            target,
            manifests_dir,
            target_words=TARGET_KEYWORDS_10,
            confusable_words=confusable_words,
            speaker_metadata_file=args.speaker_metadata_file or None,
            limit_per_word=args.limit_per_split,
        )
        stats = {split: len(records) for split, records in manifests.items()}
        out = manifests_dir / "l2_arctic_eval_stats.json"
        with out.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, ensure_ascii=False)
        print(f"Built L2-ARCTIC eval manifests from {target}")
        print(f"Saved stats to {out}")


if __name__ == "__main__":
    main()
