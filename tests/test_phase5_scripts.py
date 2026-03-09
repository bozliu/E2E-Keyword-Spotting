from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_derive_round2_detector_config_uses_fused_report(tmp_path: Path) -> None:
    script = _load_module(
        PROJECT_ROOT / "scripts" / "derive_round2_detector_config.py",
        "derive_round2_detector_config",
    )

    base_cfg = {
        "run_name": "demo_mhatt_small_focus_v3",
        "training": {
            "keyword_focus": {
                "weak_keywords": ["on", "no", "off", "down", "go"],
                "confusion_groups": {
                    "yes": ["left"],
                    "no": ["go", "down"],
                    "on": ["off"],
                },
            }
        },
    }
    fused_report = {
        "fused_per_class_kws12": {
            "yes": {"recall": 0.10, "top_confusions": [{"label": "unknown", "count": 20}, {"label": "left", "count": 5}]},
            "no": {"recall": 0.20, "top_confusions": [{"label": "go", "count": 9}, {"label": "down", "count": 7}]},
            "up": {"recall": 0.30, "top_confusions": [{"label": "silence", "count": 4}]},
            "down": {"recall": 0.40, "top_confusions": [{"label": "go", "count": 8}]},
            "left": {"recall": 0.50, "top_confusions": [{"label": "right", "count": 6}]},
            "right": {"recall": 0.60, "top_confusions": [{"label": "left", "count": 4}]},
            "on": {"recall": 0.70, "top_confusions": [{"label": "off", "count": 3}]},
            "off": {"recall": 0.80, "top_confusions": [{"label": "on", "count": 2}]},
            "stop": {"recall": 0.90, "top_confusions": [{"label": "go", "count": 1}]},
            "go": {"recall": 0.95, "top_confusions": [{"label": "no", "count": 1}]},
        }
    }

    cfg, summary = script.derive_round2_config(
        base_cfg,
        fused_report,
        weak_count=5,
        output_stem="demo_mhatt_small_focus_v3_r2",
    )

    weak_keywords = cfg["training"]["keyword_focus"]["weak_keywords"]
    assert weak_keywords == ["yes", "no", "up", "down", "left"]
    assert cfg["run_name"] == "demo_mhatt_small_focus_v3_r2"
    assert cfg["training"]["keyword_focus"]["confusion_groups"]["yes"] == ["left"]
    assert cfg["training"]["keyword_focus"]["confusion_groups"]["no"] == ["go", "down"]
    assert cfg["training"]["keyword_focus"]["confusion_groups"]["up"] == ["silence"]
    assert summary["weak_keywords"] == weak_keywords


def test_collect_phase5_round_summary_writes_failure_package(tmp_path: Path) -> None:
    script = _load_module(
        PROJECT_ROOT / "scripts" / "collect_phase5_round_summary.py",
        "collect_phase5_round_summary",
    )

    analyze = {
        "checkpoint": "outputs/demo_mhatt_small_focus_v3/best_kws12.pt",
        "metrics": {"kws12_unknown_to_target_rate": 0.03},
        "min_kws12_precision": 0.90,
        "min_kws12_recall": 0.91,
        "latency_ms": {"cpu": 12.5},
        "fused_per_class_kws12": {
            "yes": {"precision": 0.80, "recall": 0.70, "top_confusions": [{"label": "left", "count": 5}]},
            "no": {"precision": 0.92, "recall": 0.71, "top_confusions": [{"label": "go", "count": 4}]},
            "up": {"precision": 0.93, "recall": 0.72, "top_confusions": []},
            "down": {"precision": 0.94, "recall": 0.73, "top_confusions": []},
            "left": {"precision": 0.95, "recall": 0.74, "top_confusions": []},
            "right": {"precision": 0.96, "recall": 0.75, "top_confusions": []},
            "on": {"precision": 0.97, "recall": 0.76, "top_confusions": []},
            "off": {"precision": 0.98, "recall": 0.77, "top_confusions": []},
            "stop": {"precision": 0.99, "recall": 0.78, "top_confusions": []},
            "go": {"precision": 0.99, "recall": 0.79, "top_confusions": []},
        },
        "fused_min_kws12_precision": 0.80,
        "fused_min_kws12_recall": 0.70,
        "fused_unknown_to_target_rate": 0.03,
        "verify_rate": 0.30,
        "verifier_accept_rate": 0.05,
    }
    eval_report = {
        "decision_profile": "stable",
        "fused_min_kws12_precision": 0.80,
        "fused_min_kws12_recall": 0.70,
        "fused_unknown_to_target_rate": 0.03,
        "verify_rate": 0.30,
        "verifier_accept_rate": 0.05,
        "verifier_checkpoint": "outputs/demo_mhatt_small_focus_v3/best_kws12_verifier.pt",
    }
    hi_status = {
        "reduced_data_mode": False,
        "present_source_splits": ["train", "dev", "test"],
        "build_mode": "official",
    }
    verifier_metrics = {
        "verifier_macro_f1": 0.84,
        "min_verifier_precision": 0.80,
        "min_verifier_recall": 0.81,
    }

    analyze_path = tmp_path / "analyze.json"
    eval_path = tmp_path / "eval.json"
    hi_status_path = tmp_path / "hi_status.json"
    verifier_path = tmp_path / "verifier.json"
    output_path = tmp_path / "summary.json"
    failure_path = tmp_path / "failure.json"
    analyze_path.write_text(json.dumps(analyze), encoding="utf-8")
    eval_path.write_text(json.dumps(eval_report), encoding="utf-8")
    hi_status_path.write_text(json.dumps(hi_status), encoding="utf-8")
    verifier_path.write_text(json.dumps(verifier_metrics), encoding="utf-8")

    module_args = script.parse_args
    script.parse_args = lambda: type(
        "Args",
        (),
        {
            "analyze_report": str(analyze_path),
            "eval_report": str(eval_path),
            "hi_mia_status": str(hi_status_path),
            "verifier_metrics": str(verifier_path),
            "output": str(output_path),
            "failure_output": str(failure_path),
            "target_precision": 0.95,
            "target_recall": 0.95,
            "target_unknown_rate": 0.02,
            "target_latency_ms": 30.0,
        },
    )()
    try:
        script.main()
    finally:
        script.parse_args = module_args

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert summary["acceptance"]["passed"] is False
    assert "fused_min_kws12_precision" in summary["acceptance"]["failed_checks"]
    assert failure["status"] == "failed_acceptance"
    assert failure["worst5_fused_recall_labels"][0]["label"] == "yes"
