# Public Checkpoints

Pretrained weights are distributed through GitHub Releases rather than Git history.

## Release assets expected for `v2.0-public`

- Latest public demo checkpoint  
  URL: https://github.com/bozliu/E2E-Keyword-Spotting/releases/download/v2.0-public/demo_mhatt_small_focus_lod_best_kws12.pt  
  SHA256: `2263a6bab3c0b7d6015d076c094af5b90a1efbb3c18c054e41ed4202b4c9a615`

- Distilled focus checkpoint  
  URL: https://github.com/bozliu/E2E-Keyword-Spotting/releases/download/v2.0-public/demo_mhatt_small_focus_best_kws12.pt  
  SHA256: `220c61110adf2447884852d879eb05f3fd4df1907bd1bc3485f2d468add0f4f0`

- Baseline checkpoint  
  URL: https://github.com/bozliu/E2E-Keyword-Spotting/releases/download/v2.0-public/quick_mhatt_best_kws12.pt  
  SHA256: `452b00f1733d8a33333b12a8b2fa412061c3aa6ecdd23f65fd7f4c4960f5160e`

## Placement

Place downloaded files under `checkpoints/` or point commands directly to the downloaded path.
If you want `--checkpoint auto` to work with the release assets, install the symlinks expected by the ranking flow:

```bash
bash scripts/install_public_checkpoints.sh
python scripts/select_demo_checkpoint.py
```

Examples:

```bash
python -m kws.eval --checkpoint checkpoints/demo_mhatt_small_focus_lod_best_kws12.pt --split test
python -m kws.demo.realtime --checkpoint checkpoints/demo_mhatt_small_focus_lod_best_kws12.pt --device auto --wheel kws12
```

## Auto-ranking note

`--checkpoint auto` expects ranking metadata to exist under `reports/` and the referenced checkpoint files to be available locally.
