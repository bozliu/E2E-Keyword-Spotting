"""MSWC ingestion helpers for commercial-safe word-level KWS training."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from kws.constants import COMMAND31_TO_INDEX, SAMPLE_RATE, TARGET_KEYWORDS_10
from kws.data.manifest import ManifestRecord, write_manifest


def _candidate_metadata_files(root: Path) -> list[Path]:
    return sorted([*root.glob('*.csv'), *root.glob('*.tsv'), *root.glob('**/*split*.csv'), *root.glob('**/*split*.tsv')])


def _resolve_audio_path(root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    direct = (root / raw_path).resolve()
    if direct.exists():
        return direct
    for sub in ('audio', 'clips', 'wav', 'wavs'):
        nested = (root / sub / raw_path).resolve()
        if nested.exists():
            return nested
    raise FileNotFoundError(f'MSWC audio file not found: {raw_path}')


def _iter_rows_from_metadata(root: Path, metadata_file: str | Path | None = None) -> Iterable[dict[str, str]]:
    if metadata_file:
        explicit = Path(metadata_file).expanduser()
        candidate_files = [(explicit if explicit.is_absolute() else (root / explicit)).resolve()]
    else:
        candidate_files = _candidate_metadata_files(root)
    for path in candidate_files:
        delimiter = '\t' if path.suffix.lower() == '.tsv' else ','
        with path.open('r', encoding='utf-8') as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            field_names = {str(name).strip().lower() for name in (reader.fieldnames or [])}
            if not ({'path', 'audio_path', 'clip_path', 'file', 'filename'} & field_names):
                continue
            if not ({'word', 'keyword', 'label'} & field_names):
                continue
            for row in reader:
                yield {str(k).strip().lower(): str(v).strip() for k, v in row.items() if k is not None}
        return


def _scan_directory_layout(root: Path) -> Iterable[dict[str, str]]:
    for split in ('train', 'valid', 'test'):
        split_root = root / split
        if not split_root.exists():
            continue
        for word_dir in sorted(split_root.iterdir()):
            if not word_dir.is_dir():
                continue
            for wav_path in sorted(word_dir.glob('*.wav')):
                yield {
                    'split': split,
                    'word': word_dir.name,
                    'path': str(wav_path.resolve()),
                }


def build_mswc_manifests(
    root: str | Path,
    output_dir: str | Path,
    *,
    target_words: Sequence[str] = TARGET_KEYWORDS_10,
    confusable_words: Sequence[str] = (),
    metadata_file: str | Path | None = None,
    audio_root: str | Path | None = None,
    limit_per_split: int | None = None,
    limit_per_word: int | None = None,
) -> Dict[str, List[ManifestRecord]]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f'MSWC root not found: {root_path}')
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    if audio_root:
        audio_candidate = Path(audio_root).expanduser()
        audio_root_path = (audio_candidate if audio_candidate.is_absolute() else (root_path / audio_candidate)).resolve()
    else:
        audio_root_path = root_path

    allowed = set(str(word) for word in [*target_words, *confusable_words])
    manifests: Dict[str, List[ManifestRecord]] = defaultdict(list)
    per_split_counts: Dict[str, int] = defaultdict(int)
    per_word_counts: Dict[tuple[str, str], int] = defaultdict(int)

    rows = list(_iter_rows_from_metadata(root_path, metadata_file=metadata_file))
    if not rows:
        rows = list(_scan_directory_layout(root_path))

    for row in rows:
        word = str(row.get('word') or row.get('keyword') or row.get('label') or '').strip().lower()
        if not word or word not in allowed or word not in COMMAND31_TO_INDEX:
            continue
        split = str(row.get('split') or row.get('subset') or row.get('partition') or 'train').strip().lower()
        if split == 'dev':
            split = 'valid'
        if split not in ('train', 'valid', 'test'):
            continue
        if limit_per_split is not None and per_split_counts[split] >= int(limit_per_split):
            continue
        if limit_per_word is not None and per_word_counts[(split, word)] >= int(limit_per_word):
            continue
        raw_path = row.get('path') or row.get('audio_path') or row.get('clip_path') or row.get('file') or row.get('filename')
        if not raw_path:
            continue
        audio_path = _resolve_audio_path(audio_root_path, raw_path)
        manifests[split].append(
            ManifestRecord(
                path=str(audio_path),
                source='mswc',
                split=split,
                command_label=COMMAND31_TO_INDEX[word],
                wake_label=1 if word in TARGET_KEYWORDS_10 else 0,
                sr=SAMPLE_RATE,
                speaker_id=row.get('speaker_id') or row.get('speaker') or None,
                transcript=row.get('transcript') or word,
                accent_group=row.get('accent_group') or row.get('accent') or row.get('locale') or None,
                l1_group=row.get('l1_group') or row.get('native_language') or None,
                is_synthetic=False,
                difficulty_bucket='confusable' if word in confusable_words else 'target',
            )
        )
        per_split_counts[split] += 1
        per_word_counts[(split, word)] += 1

    for split in ('train', 'valid', 'test'):
        if manifests.get(split):
            write_manifest(manifests[split], output_path / f'mswc_{split}.jsonl')
    return manifests
