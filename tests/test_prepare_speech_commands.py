from __future__ import annotations

from pathlib import Path

from kws.data.speech_commands_split import prepare_speech_commands_split


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF")


def test_prepare_speech_commands_split_uses_official_lists(tmp_path: Path) -> None:
    dataset_root = tmp_path / "speech_commands_v0.02"
    _touch(dataset_root / "yes" / "yes_train.wav")
    _touch(dataset_root / "yes" / "yes_valid.wav")
    _touch(dataset_root / "no" / "no_test.wav")
    _touch(dataset_root / "_background_noise_" / "noise.wav")
    (dataset_root / "validation_list.txt").write_text("yes/yes_valid.wav\n", encoding="utf-8")
    (dataset_root / "testing_list.txt").write_text("no/no_test.wav\n", encoding="utf-8")

    output_root = tmp_path / "speech_commands_split"
    stats = prepare_speech_commands_split(dataset_root, output_root, mode="copy")

    assert stats == {"train": 1, "valid": 1, "test": 1}
    assert (output_root / "train" / "yes" / "yes_train.wav").exists()
    assert (output_root / "valid" / "yes" / "yes_valid.wav").exists()
    assert (output_root / "test" / "no" / "no_test.wav").exists()
    assert (output_root / "train" / "_background_noise_" / "noise.wav").exists()
