from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from kws.data.l2_arctic import build_l2_arctic_eval_manifests


TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.0
        intervals: size = 2
        intervals [1]:
            xmin = 0.00
            xmax = 0.40
            text = "off"
        intervals [2]:
            xmin = 0.40
            xmax = 0.80
            text = "bed"
'''


def _write_wav(path: Path, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), (torch.randn(1, sr) * 0.01), sr)


def test_build_l2_arctic_eval_manifests_extracts_word_clips(tmp_path: Path) -> None:
    root = tmp_path / 'l2'
    wav_path = root / 'ABA' / 'wav' / 'sample.wav'
    _write_wav(wav_path)
    tg_path = wav_path.with_suffix('.TextGrid')
    tg_path.write_text(TEXTGRID, encoding='utf-8')
    (root / 'speakers.csv').write_text('speaker_id,accent_group,l1_group\nABA,arabic,arabic\n', encoding='utf-8')

    manifests = build_l2_arctic_eval_manifests(
        root=root,
        output_dir=tmp_path / 'manifests',
        target_words=['off'],
        confusable_words=['bed'],
        speaker_metadata_file=root / 'speakers.csv',
    )

    total = sum(len(v) for v in manifests.values())
    assert total == 2
    any_record = next(iter(next(iter(manifests.values()))))
    assert Path(any_record.path).exists()
    assert any_record.accent_group == 'arabic'
    assert any_record.difficulty_bucket in {'target', 'confusable'}
