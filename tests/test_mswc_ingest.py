from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from kws.data.mswc import build_mswc_manifests


def _write_wav(path: Path, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), (torch.randn(1, sr) * 0.01), sr)


def test_build_mswc_manifests_filters_to_target_and_confusable_words(tmp_path: Path) -> None:
    root = tmp_path / 'mswc'
    _write_wav(root / 'clips' / 'yes_a.wav')
    _write_wav(root / 'clips' / 'bed_a.wav')
    _write_wav(root / 'clips' / 'cat_a.wav')
    (root / 'metadata.csv').write_text(
        '\n'.join(
            [
                'split,word,path,speaker_id,accent_group',
                'train,yes,yes_a.wav,spk1,mandarin',
                'valid,bed,bed_a.wav,spk2,hindi',
                'test,cat,cat_a.wav,spk3,mandarin',
            ]
        ),
        encoding='utf-8',
    )

    manifests = build_mswc_manifests(
        root=root,
        output_dir=tmp_path / 'manifests',
        target_words=['yes'],
        confusable_words=['bed'],
        metadata_file=root / 'metadata.csv',
        audio_root=root / 'clips',
    )

    assert len(manifests['train']) == 1
    assert len(manifests['valid']) == 1
    assert 'test' not in manifests or len(manifests['test']) == 0
    assert manifests['train'][0].transcript == 'yes'
    assert manifests['valid'][0].difficulty_bucket == 'confusable'
    assert manifests['train'][0].accent_group == 'mandarin'
