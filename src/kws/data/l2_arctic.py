"""L2-ARCTIC eval-only word extraction for accent robustness analysis."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torchaudio

from kws.constants import CLIP_SAMPLES, COMMAND31_TO_INDEX, SAMPLE_RATE, TARGET_KEYWORDS_10
from kws.data.audio import load_audio, pad_or_trim
from kws.data.manifest import ManifestRecord, write_manifest

_INTERVAL_RE = re.compile(
    r"intervals \[[0-9]+\]:\s*xmin = ([0-9.]+)\s*xmax = ([0-9.]+)\s*text = \"([^\"]*)\"",
    flags=re.MULTILINE,
)


DEFAULT_ACCENT_BY_SPEAKER: Dict[str, str] = {
    'ABA': 'arabic',
    'BWC': 'mandarin',
    'EBVS': 'spanish',
    'ERMS': 'spanish',
    'HJK': 'korean',
    'HKK': 'mandarin',
    'HQTV': 'vietnamese',
    'LXC': 'mandarin',
    'MBMPS': 'spanish',
    'NCC': 'mandarin',
    'PNV': 'vietnamese',
    'RRBI': 'hindi',
    'SKA': 'hindi',
    'SVBI': 'hindi',
    'THV': 'vietnamese',
    'TLV': 'spanish',
    'YDCK': 'korean',
    'YKWK': 'korean',
    'ZHAA': 'arabic',
}


def _speaker_metadata(root: Path, speaker_metadata_file: str | Path | None = None) -> Dict[str, Dict[str, str]]:
    meta: Dict[str, Dict[str, str]] = {}
    if speaker_metadata_file:
        explicit = Path(speaker_metadata_file).expanduser()
        candidate_paths = [(explicit if explicit.is_absolute() else (root / explicit)).resolve()]
    else:
        candidate_paths = [*root.glob('*.csv'), *root.glob('*.tsv')]
    for path in candidate_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding='utf-8').strip().splitlines()
        if not text:
            continue
        headers = [item.strip().lower() for item in re.split(r'[\t,]', text[0])]
        if 'speaker_id' not in headers and 'speaker' not in headers:
            continue
        for line in text[1:]:
            values = [item.strip() for item in re.split(r'[\t,]', line)]
            row = dict(zip(headers, values))
            speaker = row.get('speaker_id') or row.get('speaker')
            if not speaker:
                continue
            meta[str(speaker).upper()] = {
                'accent_group': row.get('accent_group') or row.get('accent') or row.get('l1_group') or '',
                'l1_group': row.get('l1_group') or row.get('accent') or '',
            }
        break
    return meta


def _parse_textgrid(path: Path) -> List[tuple[float, float, str]]:
    text = path.read_text(encoding='utf-8', errors='ignore')
    intervals = []
    for start, end, token in _INTERVAL_RE.findall(text):
        word = token.strip().lower()
        if not word or word in {'sp', 'sil', 'silence'}:
            continue
        intervals.append((float(start), float(end), word))
    return intervals


def _iter_recordings(root: Path) -> Iterable[tuple[str, Path, Path]]:
    for wav_path in sorted(root.rglob('*.wav')):
        speaker = (wav_path.parent.parent.name if wav_path.parent.name.lower() == 'wav' else wav_path.parent.name).upper()
        textgrid = wav_path.with_suffix('.TextGrid')
        if not textgrid.exists():
            alt = wav_path.parent.parent / 'annotation' / wav_path.parent.name / f'{wav_path.stem}.TextGrid'
            textgrid = alt if alt.exists() else textgrid
        if not textgrid.exists():
            alt = wav_path.parent.parent / 'annotation' / f'{wav_path.stem}.TextGrid'
            textgrid = alt if alt.exists() else textgrid
        if not textgrid.exists():
            continue
        yield speaker, wav_path, textgrid


def build_l2_arctic_eval_manifests(
    root: str | Path,
    output_dir: str | Path,
    *,
    target_words: Sequence[str] = TARGET_KEYWORDS_10,
    confusable_words: Sequence[str] = (),
    limit_per_split: int | None = None,
    speaker_metadata_file: str | Path | None = None,
    limit_per_word: int | None = None,
) -> Dict[str, List[ManifestRecord]]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f'L2-ARCTIC root not found: {root_path}')
    output_path = Path(output_dir).expanduser().resolve()
    clips_root = output_path / 'l2_arctic_eval_clips'
    clips_root.mkdir(parents=True, exist_ok=True)

    speaker_meta = _speaker_metadata(root_path, speaker_metadata_file=speaker_metadata_file)
    allowed = set(str(word) for word in [*target_words, *confusable_words])
    manifests: Dict[str, List[ManifestRecord]] = defaultdict(list)
    per_split = defaultdict(int)
    per_word = defaultdict(int)

    speakers = sorted({speaker for speaker, _wav, _tg in _iter_recordings(root_path)})
    if len(speakers) <= 1:
        split_by_speaker = {speaker: 'test' for speaker in speakers}
    else:
        split_by_speaker = {speaker: ('valid' if idx % 2 == 0 else 'test') for idx, speaker in enumerate(speakers)}

    for speaker, wav_path, textgrid_path in _iter_recordings(root_path):
        split = split_by_speaker.get(speaker, 'test')
        waveform = load_audio(str(wav_path), sample_rate=SAMPLE_RATE)
        metadata = speaker_meta.get(speaker, {})
        accent_group = metadata.get('accent_group') or DEFAULT_ACCENT_BY_SPEAKER.get(speaker, 'l2_arctic')
        l1_group = metadata.get('l1_group') or accent_group
        for idx, (start_s, end_s, word) in enumerate(_parse_textgrid(textgrid_path)):
            if word not in allowed or word not in COMMAND31_TO_INDEX:
                continue
            if limit_per_split is not None and per_split[split] >= int(limit_per_split):
                continue
            if limit_per_word is not None and per_word[word] >= int(limit_per_word):
                continue
            start = max(0, int(round(start_s * SAMPLE_RATE)))
            end = max(start + 1, int(round(end_s * SAMPLE_RATE)))
            clip = waveform[start:end]
            clip = pad_or_trim(clip, target_samples=CLIP_SAMPLES)
            clip_dir = clips_root / split / speaker.lower() / word
            clip_dir.mkdir(parents=True, exist_ok=True)
            clip_path = clip_dir / f'{wav_path.stem}_{idx:04d}.wav'
            torchaudio.save(str(clip_path), clip.view(1, -1), SAMPLE_RATE)
            manifests[split].append(
                ManifestRecord(
                    path=str(clip_path.resolve()),
                    source='l2_arctic_eval',
                    split=split,
                    command_label=COMMAND31_TO_INDEX[word],
                    wake_label=1 if word in TARGET_KEYWORDS_10 else 0,
                    sr=SAMPLE_RATE,
                    speaker_id=speaker,
                    transcript=word,
                    accent_group=accent_group,
                    l1_group=l1_group,
                    is_synthetic=False,
                    difficulty_bucket='confusable' if word in confusable_words else 'target',
                )
            )
            per_split[split] += 1
            per_word[word] += 1

    for split in ('valid', 'test'):
        if manifests.get(split):
            write_manifest(manifests[split], output_path / f'l2_arctic_eval_{split}.jsonl')
    return manifests
