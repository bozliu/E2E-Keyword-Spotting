#!/usr/bin/env python
"""Create the split directory expected by the training pipeline."""

from __future__ import annotations

import argparse

from kws.data.speech_commands_split import prepare_speech_commands_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Speech Commands train/valid/test folders from official split files.")
    parser.add_argument("--dataset-root", required=True, help="Path to the official Speech Commands dataset root.")
    parser.add_argument(
        "--output-root",
        default="data/local/speech_commands_split",
        help="Destination directory for the split tree.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Use symlinks for a lightweight public workflow or copy for portability.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output tree.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = prepare_speech_commands_split(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        mode=args.mode,
        overwrite=args.overwrite,
    )
    print("Prepared Speech Commands split:")
    for split, count in stats.items():
        print(f"  {split}: {count}")


if __name__ == "__main__":
    main()
