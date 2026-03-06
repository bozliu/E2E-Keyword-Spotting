"""Download external datasets used by the public release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kws.data.hi_mia import build_himia_manifests, download_hi_mia
from kws.data.l2_arctic import build_l2_arctic_eval_manifests
from kws.data.mswc import build_mswc_manifests
from kws.utils.keyword_focus import DEFAULT_CONFUSION_GROUPS
from kws.constants import TARGET_KEYWORDS_10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download external datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["hi_mia", "mswc", "l2_arctic_eval"])
    parser.add_argument("--root", type=str, default="data/external/hi_mia")
    parser.add_argument("--manifests-dir", type=str, default="data/processed/manifests")
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--metadata-file", type=str, default="")
    parser.add_argument("--audio-root", type=str, default="")
    parser.add_argument("--speaker-metadata-file", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    confusable_words = sorted({word for values in DEFAULT_CONFUSION_GROUPS.values() for word in values})

    if args.dataset == "hi_mia":
        target = (project_root / args.root).resolve()
        print(f"Downloading HI-MIA to {target} ...")
        download_hi_mia(target)

        manifests_dir = (project_root / args.manifests_dir).resolve()
        manifests = build_himia_manifests(target, manifests_dir, limit_per_split=args.limit_per_split)
        stats = {split: len(records) for split, records in manifests.items()}
        out = manifests_dir / "hi_mia_stats.json"
        with out.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, ensure_ascii=False)
        print(f"Done. Manifest stats: {stats}")
        print(f"Saved stats to {out}")
        return

    if args.dataset == "mswc":
        target = (project_root / args.root).resolve()
        manifests_dir = (project_root / args.manifests_dir).resolve()
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
        target = (project_root / args.root).resolve()
        manifests_dir = (project_root / args.manifests_dir).resolve()
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
