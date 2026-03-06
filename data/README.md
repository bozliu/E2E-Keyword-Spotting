# Data Setup

This repository does not redistribute any dataset audio.

## Required datasets

### 1. Google Speech Commands
- Official paper: https://arxiv.org/abs/1804.03209
- Official download (v0.02): https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

Prepare the split directory expected by the training pipeline:

```bash
python scripts/prepare_speech_commands.py \
  --dataset-root /path/to/speech_commands_v0.02 \
  --output-root data/local/speech_commands_split \
  --mode symlink
```

Expected layout after preparation:

```text
data/local/speech_commands_split/
  train/
  valid/
  test/
```

### 2. HI-MIA
- Official dataset page: https://www.openslr.org/85/
- Official paper: https://arxiv.org/abs/1912.01231

Download and build manifests:

```bash
bash scripts/download_hi_mia.sh
```

## Optional extension datasets

The codebase also contains optional ingestion utilities for:
- MLCommons Multilingual Spoken Words Corpus (MSWC)
- L2-ARCTIC accent evaluation slices

These are disabled by default in the public `v2` configs.

## License reminder

Always follow the original license and terms of use for every dataset you download.
