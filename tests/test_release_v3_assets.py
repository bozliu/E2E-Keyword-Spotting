from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "generate_release_assets.py"
    spec = importlib.util.spec_from_file_location("generate_release_assets", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_release_assets_builds_expected_files(tmp_path):
    module = _load_module()
    root = Path(__file__).resolve().parents[1]
    module.REPORTS_DIR = root / "reports"
    module.ASSETS_DIR = tmp_path / "assets"
    module.ASSETS_DATA_DIR = module.ASSETS_DIR / "data"

    summary = module.build_assets()

    assert summary["headline"]["overall_passed"] is True
    assert summary["release_claim"] in {"offline_validated_realtime_smoke_only", "full_realtime_validated"}

    per_class = module.ASSETS_DIR / "per_class_valid_test.png"
    latency = module.ASSETS_DIR / "latency_vs_accuracy.png"
    summary_json = module.ASSETS_DATA_DIR / "release_summary_v3.json"
    realtime_json = module.ASSETS_DATA_DIR / "realtime_status.json"

    assert per_class.exists()
    assert latency.exists()
    assert summary_json.exists()
    assert realtime_json.exists()

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["headline"]["model"] == "ensemble/ast-superb-kws12"
    assert len(payload["version_rows"]) == 3
    assert any(row["version"] == "v3 (main / v3.0.0)" for row in payload["version_rows"])
