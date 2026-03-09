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

Use the following flow for the public `v3` release:

```bash
git fetch origin
git switch main
git pull origin main
git tag -a v2.0.0 origin/main -m "Public v2 snapshot before the v3 merge"
git push origin v2.0.0

git switch -c v3
git add README.md docs .github scripts src tests pyproject.toml environment.yml requirements-space.txt app.py .gitignore
git commit -m "Prepare v3 public release"
git push -u origin v3
```

Then:

1. Open a PR from `v3` into `main`.
2. Let `core-release-checks` and `smoke-tests` pass.
3. Protect `main` before merge: no force push, no delete, PR-only merge, required checks enabled.
4. Merge the PR so `main` now points at the v3 content.
5. Create tag `v3.0.0` on the merged `main` head.
6. Create a GitHub Release from `v3.0.0`.

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
- Hugging Face Space: optional browser demo, if we keep the CPU public demo path alive

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
