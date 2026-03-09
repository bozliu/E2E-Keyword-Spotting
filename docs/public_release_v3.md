# v3 Public Release Process

This document is the public-release companion for the repository root `README.md`.
It answers the practical questions that matter at release time: what to verify locally, what to claim publicly, what to upload to GitHub Releases and Hugging Face, and what to avoid publishing.

## Claim policy

Use the following rule before changing the README headline:

- If both `reports/realtime_accuracy_first_valid.json` and `reports/realtime_accuracy_first_test.json` exist and both pass, it is safe to state that the desktop realtime `accuracy-first` path passed the current local protocol.
- If full realtime validation is missing or fails, keep the public claim conservative:
  - offline local-protocol benchmark passed
  - desktop realtime path exists and smoke validation passed
  - full-stream realtime claim remains experimental

Do not headline reduced-data HI-MIA numbers.

## Local release gate

Run this once before opening a PR:

```bash
bash scripts/prepare_release_v3.sh
```

To include the long realtime gate:

```bash
KWS_RELEASE_RUN_LONG_VALIDATION=1 bash scripts/prepare_release_v3.sh
```

This script checks:

- `dl` exists and installs cleanly
- core entrypoints compile
- the release test subset passes
- public figures and cleaned JSON summaries are regenerated
- optionally, full realtime `valid/test` validation is rerun

## Git / PR / tag flow

Use the following flow to preserve a clean public version topology:

```bash
git fetch origin
git switch main
git pull origin main
git branch v2 v2.0.0
git push origin v2

git tag -a v2.0.0 origin/main -m "Public v2 snapshot before the v3 merge"
git push origin v2.0.0
```

Then:

1. Keep `v2` as a read-only archive branch and `v2.0.0` as the canonical v2 tag.
2. Let `core-release-checks` and `smoke-tests` pass on `main`.
3. Protect `v2` and `main`: no force push, no delete, required checks on `main`.
4. Keep `main` as the public `v3` line and `v3.0.0` as the matching release tag.
5. Remove legacy alias tags and URLs such as `v2.0-public` once all public references have moved to `v2.0.0`.
6. Create or refresh the GitHub Release from `v3.0.0`.

## GitHub Release assets

Attach only assets that are useful for public verification:

- cleaned benchmark JSON from `docs/assets/data/`
- figure PNGs from `docs/assets/`
- demo GIF or MP4 from `docs/assets/`
- our detector/verifier checkpoints if they are ours to redistribute
- calibration JSON files that belong to our released runtime path

Do not upload:

- raw datasets
- local caches
- `outputs/` training directories
- reports with absolute local filesystem paths

## Hugging Face release guidance

Use Hugging Face for our own model artifacts, not for mirroring upstream third-party weights.

Recommended split:

- GitHub: source code, README, release notes, figures, GIF, cleaned metrics
- Hugging Face model repo: our checkpoint(s), calibration, model card, usage notes
- Hugging Face Space: currently unpublished; only re-enable it if the hosted path matches the current public release story

If a hosted browser demo is re-enabled later, keep only one public Space entrypoint:

- `bozliu/e2e-keyword-spotting-demo`

Do not keep preview, smoke, or fresh-clone Spaces public.

For `ensemble/ast-superb-kws12`, publish:

- the mapping and calibration logic
- the upstream model ids
- a model card that explains it is a composed runtime, not a repackaged single upstream weight file

## Visuals policy

Prefer visuals that help a public reader understand behavior:

- system diagram
- per-class precision/recall chart
- latency vs quality chart
- a short GIF or MP4 of realtime inference

Recommended command for the GIF:

```bash
conda run -n dl python scripts/render_release_demo_gif.py \
  --label go \
  --split test \
  --checkpoint auto \
  --device mps \
  --external-kws-device mps
```

Avoid:

- raw terminal screenshots
- static UI screenshots without context
- tables without a comparability note
