from __future__ import annotations

import json
from pathlib import Path

import torch

from kws.data.hard_negatives import collect_confusable_groups, generate_hard_negative_dataset
from kws.data.manifest import read_manifest


def _build_checkpoint_fixture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "outputs" / "demo_mhatt_small_focus"
    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "best_kws12.pt"
    torch.save(
        {
            "config": {
                "training": {
                    "keyword_focus": {
                        "confusion_groups": {
                            "yes": ["bird", "bed"],
                            "no": ["go"],
                        }
                    }
                }
            }
        },
        checkpoint_path,
    )
    (run_dir / "keyword_focus.json").write_text(
        json.dumps(
            {
                "per_keyword": {
                    "yes": {"top_confusions": [{"label": "cat", "count": 3}]},
                    "go": {"top_confusions": [{"label": "no", "count": 2}]},
                },
                "confusable_groups": {
                    "left": ["right"],
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return checkpoint_path


def test_collect_confusable_groups_merges_checkpoint_and_focus_report(tmp_path: Path) -> None:
    checkpoint_path = _build_checkpoint_fixture(tmp_path)

    groups = collect_confusable_groups(checkpoint_path)

    assert "bird" in groups["yes"]
    assert "bed" in groups["yes"]
    assert "cat" in groups["yes"]
    assert "go" in groups["no"]
    assert "right" in groups["left"]


def test_generate_hard_negative_dataset_writes_manifests(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = _build_checkpoint_fixture(tmp_path)
    output_dir = tmp_path / "data" / "synthetic" / "hard_negatives"
    manifests_dir = tmp_path / "data" / "processed" / "manifests"

    monkeypatch.setattr("kws.data.hard_negatives.resolve_say_voices", lambda preferred: ["Alex"])

    def _fake_synthesize(path: Path, *, voice: str, rate: int, text: str) -> None:  # noqa: ARG001
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFFsynthetic")

    monkeypatch.setattr("kws.data.hard_negatives._synthesize_with_say", _fake_synthesize)

    manifests = generate_hard_negative_dataset(
        output_dir=output_dir,
        manifests_dir=manifests_dir,
        source_checkpoint=checkpoint_path,
        voices=("Alex",),
        rates=(175,),
        overwrite=True,
    )

    total_records = sum(len(records) for records in manifests.values())
    assert total_records > 0
    assert (manifests_dir / "synthetic_hard_negative_train.jsonl").exists()
    assert (manifests_dir / "synthetic_hard_negative_valid.jsonl").exists()
    assert (manifests_dir / "synthetic_hard_negative_stats.json").exists()

    saved_records = read_manifest(manifests_dir / "synthetic_hard_negative_train.jsonl") + read_manifest(
        manifests_dir / "synthetic_hard_negative_valid.jsonl"
    )
    assert len(saved_records) == total_records
    assert all(record.source == "synthetic_hard_negative" for record in saved_records)
    assert all(record.is_synthetic for record in saved_records)
    assert all(record.difficulty_bucket == "hard_negative" for record in saved_records)
    assert all(Path(record.path).exists() for record in saved_records)
