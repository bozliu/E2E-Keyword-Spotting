from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import torch

from kws.eval import __main__ as eval_main


class _DummyEvalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loaded_state = None
        self.moved_to = None

    def load_state_dict(self, state):
        self.loaded_state = state

    def to(self, device):
        self.moved_to = device
        return self


def test_eval_main_emits_per_class_kws12_fields(tmp_path, monkeypatch) -> None:
    ckpt_dir = tmp_path / "outputs" / "demo_run"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "best_kws12.pt"
    output_path = tmp_path / "eval.json"
    torch.save(
            {
                "config": {
                    "model": {"name": "mhatt_crnn"},
                    "features": {"n_mels": 80, "audio_seconds": 1.0},
                    "training": {"loss_weights": {"kws12": 0.35}},
                },
            "label_set": ["silence", "yes", "no"],
            "model_state": {"dummy": 1},
        },
        ckpt_path,
    )

    dummy_model = _DummyEvalModel()
    monkeypatch.setattr(
        eval_main,
        "parse_args",
        lambda: Namespace(
            checkpoint=str(ckpt_path),
            verifier_checkpoint="off",
            verifier_backend="internal",
            external_verifier_model="",
            external_verifier_device="auto",
            decision_profile="stable",
            split="test",
            device="cpu",
            output=str(output_path),
        ),
    )
    monkeypatch.setattr(eval_main, "prepare_data", lambda cfg, project_root: None)
    monkeypatch.setattr(eval_main, "create_dataloaders", lambda cfg, project_root: SimpleNamespace(test="loader"))
    monkeypatch.setattr(eval_main, "pick_device", lambda preferred: torch.device("cpu"))
    monkeypatch.setattr(eval_main, "create_model", lambda cfg, n_mels, num_commands: dummy_model)
    monkeypatch.setattr(eval_main, "run_repo_preflight", lambda *args, **kwargs: {"manifest_audit": {"is_clean": True}})
    monkeypatch.setattr(eval_main, "load_runtime_verifier", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        eval_main,
        "_collect_outputs",
        lambda **kwargs: {
            "command_probs": torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float32).numpy(),
            "targets": torch.tensor([0], dtype=torch.long).numpy(),
            "verifier_probs": None,
        },
    )
    monkeypatch.setattr(
        eval_main,
        "compute_fused_payload",
        lambda **kwargs: {
            "fused_metrics": {
                "per_class_kws12": {"silence": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1, "predicted": 1, "top_confusions": []}},
                "min_kws12_precision": 0.97,
                "min_kws12_recall": 0.98,
                "kws12_unknown_to_target_rate": 0.01,
            },
            "verify_rate": 0.0,
            "verifier_accept_rate": 0.0,
        },
    )
    monkeypatch.setattr(
        eval_main,
        "run_epoch",
        lambda **kwargs: SimpleNamespace(
            loss=0.125,
            metrics={
                "kws12_acc": 0.98,
                "per_class_kws12": {
                    "silence": {"precision": 1.0, "recall": 0.95, "f1": 0.974, "support": 20, "predicted": 19, "top_confusions": []}
                },
                "min_kws12_precision": 0.95,
                "min_kws12_recall": 0.96,
            },
        ),
    )

    eval_main.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["kws12_acc"] == 0.98
    assert payload["per_class_kws12"]["silence"]["recall"] == 0.95
    assert payload["min_kws12_precision"] == 0.95
    assert payload["min_kws12_recall"] == 0.96
    assert payload["verifier_backend"] == "internal"
    assert payload["fused_min_kws12_precision"] == 0.97
    assert payload["fused_unknown_to_target_rate"] == 0.01
    assert dummy_model.loaded_state == {"dummy": 1}
    assert dummy_model.moved_to == torch.device("cpu")
