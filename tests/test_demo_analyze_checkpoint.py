from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import torch

from kws.demo import analyze_checkpoint


class _DummyAnalyzeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loaded_state = None
        self.moved_to = None

    def load_state_dict(self, state):
        self.loaded_state = state

    def to(self, device):
        self.moved_to = device
        return self


def test_analyze_checkpoint_emits_per_class_kws12_fields(tmp_path, monkeypatch) -> None:
    ckpt_dir = tmp_path / "outputs" / "demo_run"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "best_kws12.pt"
    output_path = tmp_path / "demo_analysis.json"
    torch.save(
        {
            "config": {
                "features": {
                    "n_mels": 80,
                    "sample_rate": 16000,
                    "n_fft": 1024,
                    "hop_length": 128,
                    "f_min": 20.0,
                    "f_max": 7600.0,
                    "audio_seconds": 1.0,
                },
                "data": {
                    "manifests_dir": "manifests",
                    "external": {
                        "hi_mia": {"enabled": False},
                        "l2_arctic_eval": {"enabled": False},
                    },
                },
                "training": {"loss_weights": {"kws12": 0.35}, "keyword_focus": {"top_k": 3}, "aux_margin": 0.2},
                "model": {"name": "mhatt_crnn"},
            },
            "label_set": ["silence", "yes", "no"],
            "model_state": {"dummy": 1},
        },
        ckpt_path,
    )

    dummy_model = _DummyAnalyzeModel()
    monkeypatch.setattr(
        analyze_checkpoint,
        "parse_args",
        lambda: Namespace(
            checkpoint=str(ckpt_path),
            verifier_checkpoint="off",
            verifier_backend="internal",
            external_verifier_model="",
            external_verifier_device="auto",
            decision_profile="stable",
            split="test",
            output=str(output_path),
        ),
    )
    monkeypatch.setattr(analyze_checkpoint, "prepare_data", lambda cfg, project_root: None)
    monkeypatch.setattr(analyze_checkpoint, "create_dataloaders", lambda cfg, project_root: SimpleNamespace(test="loader"))
    monkeypatch.setattr(analyze_checkpoint, "load_manifests", lambda paths: [])
    monkeypatch.setattr(analyze_checkpoint, "pick_device", lambda preferred: torch.device("cpu"))
    monkeypatch.setattr(analyze_checkpoint, "create_model", lambda cfg, n_mels, num_commands: dummy_model)
    monkeypatch.setattr(analyze_checkpoint, "run_repo_preflight", lambda *args, **kwargs: {"manifest_audit": {"is_clean": True}})
    monkeypatch.setattr(analyze_checkpoint, "load_runtime_verifier", lambda *args, **kwargs: None)
    monkeypatch.setattr(analyze_checkpoint, "MelFrontend", lambda **kwargs: object())
    monkeypatch.setattr(
        analyze_checkpoint,
        "run_epoch",
        lambda **kwargs: SimpleNamespace(
            loss=0.25,
            metrics={
                "kws12_acc": 0.97,
                "kws12_unknown_to_target_rate": 0.01,
                "per_class_kws12": {
                    "silence": {"precision": 0.99, "recall": 0.98, "f1": 0.985, "support": 20, "predicted": 20, "top_confusions": []}
                },
                "min_kws12_precision": 0.95,
                "min_kws12_recall": 0.96,
            },
        ),
    )
    monkeypatch.setattr(
        analyze_checkpoint,
        "_collect_outputs",
        lambda model, loader, device, num_commands: {
            "command_probs": np.zeros((1, 3), dtype=np.float32),
            "wake_probs": np.zeros((1,), dtype=np.float32),
            "preds": np.zeros((1,), dtype=np.int64),
            "targets": np.zeros((1,), dtype=np.int64),
        },
    )
    monkeypatch.setattr(analyze_checkpoint, "_collect_verifier_probs", lambda verifier, loader: None)
    monkeypatch.setattr(
        analyze_checkpoint,
        "compute_fused_payload",
        lambda **kwargs: {
            "fused_metrics": {
                "per_class_kws12": {"silence": {"precision": 0.98, "recall": 0.97, "f1": 0.975, "support": 20, "predicted": 20, "top_confusions": []}},
                "min_kws12_precision": 0.94,
                "min_kws12_recall": 0.95,
                "kws12_unknown_to_target_rate": 0.02,
            },
            "verify_rate": 0.15,
            "verifier_accept_rate": 0.10,
        },
    )
    monkeypatch.setattr(
        analyze_checkpoint,
        "build_keyword_focus_report",
        lambda preds, targets, top_k, focus_keywords=None, focus_pairs=None: {
            "per_keyword": {},
            "bottom3_keyword_recall": 0.9,
            "keyword_balance_gap": 0.1,
            "focus_keyword_recall_mean": 0.92,
            "focus_pair_confusions": [],
            "focus_pair_confusion_rate": 0.03,
            "weak_keywords": ["left"],
        },
    )
    monkeypatch.setattr(analyze_checkpoint, "fit_keyword_calibration", lambda command_probs, wake_probs, targets, focus: {"yes": {"threshold": 0.5}})
    monkeypatch.setattr(analyze_checkpoint, "compute_accent_slices", lambda records, preds, targets: {})
    monkeypatch.setattr(analyze_checkpoint, "benchmark_latency_ms", lambda checkpoint, device: 12.5 if device.type == "cpu" else 50.0)
    monkeypatch.setattr(analyze_checkpoint, "run_stress_eval", lambda **kwargs: {"status": "ok"})
    monkeypatch.setattr(analyze_checkpoint, "_baseline_unknown_guardrail", lambda project_root: 0.05)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    analyze_checkpoint.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["kws12_acc"] == 0.97
    assert payload["per_class_kws12"]["silence"]["precision"] == 0.99
    assert payload["min_kws12_precision"] == 0.95
    assert payload["min_kws12_recall"] == 0.96
    assert payload["verifier_backend"] == "internal"
    assert payload["fused_min_kws12_precision"] == 0.94
    assert payload["fused_unknown_to_target_rate"] == 0.02
    assert payload["selection_passed_fp_guardrail"] is True
    assert dummy_model.loaded_state == {"dummy": 1}
    assert dummy_model.moved_to == torch.device("cpu")
