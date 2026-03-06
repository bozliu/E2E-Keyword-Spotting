from __future__ import annotations

import json
from pathlib import Path

import torch

from kws.constants import COMMAND31_LABELS
from kws.demo import rank_checkpoints as rc


def _write_run(
    run_dir: Path,
    *,
    run_name: str,
    acc: float,
    recall: float = 0.75,
    unknown_rate: float = 0.04,
    focus_recall: float | None = None,
    focus_rate: float | None = None,
    bottom3: float | None = None,
    balance_gap: float | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best_kws12.pt"
    ckpt = {
        "run": run_name,
        "config": {
            "model": {"name": "mhatt_crnn", "conv_channels": 8, "gru_hidden": 16, "gru_layers": 1, "num_heads": 2, "dropout": 0.0},
            "features": {"sample_rate": 16000, "audio_seconds": 1.0, "hop_length": 128, "n_mels": 80},
        },
        "label_set": COMMAND31_LABELS,
        "model_state": {},
    }
    torch.save(ckpt, ckpt_path)
    metrics = {
        "epoch": 1,
        "valid_metrics": {
            "kws12_acc": float(acc),
            "kws12_target_recall": float(recall),
            "kws12_unknown_to_target_rate": float(unknown_rate),
            "focus_keyword_recall_mean": float(focus_recall if focus_recall is not None else recall),
            "focus_pair_confusion_rate": float(focus_rate if focus_rate is not None else unknown_rate),
            "bottom3_keyword_recall": float(bottom3 if bottom3 is not None else recall),
            "keyword_balance_gap": float(balance_gap if balance_gap is not None else (1.0 - recall)),
            "wake_frr_at_1fa_per_hour": 0.95,
        },
    }
    (run_dir / "metrics_history.jsonl").write_text(json.dumps(metrics) + "\n", encoding="utf-8")
    analysis = {
        "metrics": metrics["valid_metrics"],
        "focus_keyword_recall_mean": float(focus_recall if focus_recall is not None else recall),
        "focus_pair_confusions": {},
        "focus_pair_confusion_rate": float(focus_rate if focus_rate is not None else unknown_rate),
        "bottom3_keyword_recall": float(bottom3 if bottom3 is not None else recall),
        "keyword_balance_gap": float(balance_gap if balance_gap is not None else (1.0 - recall)),
        "per_keyword": {},
    }
    (run_dir / "demo_analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    return ckpt_path


def test_rank_checkpoints_balances_accuracy_and_latency(tmp_path: Path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    _write_run(outputs / "quick_mhatt", run_name="quick_mhatt", acc=0.88, recall=0.80, unknown_rate=0.04)
    slow = _write_run(outputs / "accurate_slow", run_name="accurate_slow", acc=0.90, recall=0.78, unknown_rate=0.04)
    fast = _write_run(outputs / "fast_slightly_less", run_name="fast_slightly_less", acc=0.89, recall=0.86, unknown_rate=0.05)
    seen_iters: list[int] = []

    def fake_latency(ckpt, device, iters: int = 30) -> float:  # noqa: ARG001
        seen_iters.append(int(iters))
        return 10.0 if ckpt.get("run") == "fast_slightly_less" else 150.0

    monkeypatch.setattr(rc, "benchmark_latency_ms", fake_latency)

    ranked = rc.rank_checkpoints(outputs_root=outputs, device=torch.device("cpu"), metric_balance=(0.4, 0.6), benchmark_iters=1)
    assert ranked[0].checkpoint == fast.resolve()
    assert ranked[1].checkpoint != fast.resolve()
    assert {ranked[1].checkpoint, ranked[2].checkpoint} == {slow.resolve(), (outputs / "quick_mhatt" / "best_kws12.pt").resolve()}
    assert ranked[0].runtime_device == "cpu"
    assert seen_iters and all(iters == 1 for iters in seen_iters)


def test_select_best_checkpoint_writes_report(tmp_path: Path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    _ = _write_run(outputs / "quick_mhatt", run_name="quick_mhatt", acc=0.88, recall=0.80, unknown_rate=0.04)
    _ = _write_run(outputs / "accurate_slow", run_name="accurate_slow", acc=0.90, recall=0.78, unknown_rate=0.04)
    fast = _write_run(outputs / "fast_slightly_less", run_name="fast_slightly_less", acc=0.89, recall=0.86, unknown_rate=0.05)
    seen_iters: list[int] = []

    def fake_latency(ckpt, device, iters: int = 30) -> float:  # noqa: ARG001
        seen_iters.append(int(iters))
        return 10.0 if ckpt.get("run") == "fast_slightly_less" else 150.0

    monkeypatch.setattr(rc, "benchmark_latency_ms", fake_latency)

    report = tmp_path / "report.json"
    chosen, runtime_device = rc.select_best_checkpoint(
        outputs_root=outputs,
        device="cpu",
        metric_balance=(0.4, 0.6),
        benchmark_iters=7,
        report_path=report,
        use_cache=False,
        rebuild=True,
    )
    assert chosen == fast.resolve()
    assert runtime_device == "cpu"
    assert report.exists()
    assert seen_iters and all(iters == 7 for iters in seen_iters)


def test_select_best_checkpoint_falls_back_to_quick_mhatt_when_guardrail_fails(tmp_path: Path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    baseline = _write_run(outputs / "quick_mhatt", run_name="quick_mhatt", acc=0.87, recall=0.75, unknown_rate=0.04)
    _ = _write_run(outputs / "recall_heavy", run_name="recall_heavy", acc=0.92, recall=0.95, unknown_rate=0.10)

    def fake_latency(ckpt, device, iters: int = 30) -> float:  # noqa: ARG001
        return 8.0

    monkeypatch.setattr(rc, "benchmark_latency_ms", fake_latency)

    chosen, runtime_device = rc.select_best_checkpoint(
        outputs_root=outputs,
        device="cpu",
        benchmark_iters=1,
        report_path=tmp_path / "report.json",
        use_cache=False,
        rebuild=True,
    )

    assert chosen == baseline.resolve()
    assert runtime_device == "cpu"


def test_rank_checkpoints_uses_overall_metrics_after_focus_tie(tmp_path: Path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    _write_run(
        outputs / "quick_mhatt",
        run_name="quick_mhatt",
        acc=0.87,
        recall=0.76,
        unknown_rate=0.04,
        focus_recall=0.80,
        focus_rate=0.05,
        bottom3=0.62,
        balance_gap=0.30,
    )
    balanced = _write_run(
        outputs / "balanced",
        run_name="balanced",
        acc=0.86,
        recall=0.77,
        unknown_rate=0.05,
        focus_recall=0.80,
        focus_rate=0.05,
        bottom3=0.74,
        balance_gap=0.18,
    )
    topline = _write_run(
        outputs / "topline",
        run_name="topline",
        acc=0.89,
        recall=0.79,
        unknown_rate=0.05,
        focus_recall=0.80,
        focus_rate=0.05,
        bottom3=0.60,
        balance_gap=0.36,
    )

    def fake_latency(ckpt, device, iters: int = 30) -> float:  # noqa: ARG001
        return 9.0

    monkeypatch.setattr(rc, "benchmark_latency_ms", fake_latency)

    ranked = rc.rank_checkpoints(outputs_root=outputs, device=torch.device("cpu"), benchmark_iters=1)
    assert ranked[0].checkpoint == topline.resolve()
    assert ranked[1].checkpoint == balanced.resolve()
    assert ranked[-1].checkpoint == (outputs / "quick_mhatt" / "best_kws12.pt").resolve()


def test_rank_checkpoints_prioritizes_focus_keyword_metrics(tmp_path: Path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    _write_run(
        outputs / "quick_mhatt",
        run_name="quick_mhatt",
        acc=0.87,
        recall=0.76,
        unknown_rate=0.04,
        focus_recall=0.76,
        focus_rate=0.09,
        bottom3=0.62,
        balance_gap=0.30,
    )
    focus_winner = _write_run(
        outputs / "focus_winner",
        run_name="focus_winner",
        acc=0.86,
        recall=0.77,
        unknown_rate=0.05,
        focus_recall=0.90,
        focus_rate=0.03,
        bottom3=0.65,
        balance_gap=0.28,
    )
    topline = _write_run(
        outputs / "topline",
        run_name="topline",
        acc=0.89,
        recall=0.80,
        unknown_rate=0.05,
        focus_recall=0.79,
        focus_rate=0.10,
        bottom3=0.70,
        balance_gap=0.22,
    )

    def fake_latency(ckpt, device, iters: int = 30) -> float:  # noqa: ARG001
        return 8.0

    monkeypatch.setattr(rc, "benchmark_latency_ms", fake_latency)

    ranked = rc.rank_checkpoints(outputs_root=outputs, device=torch.device("cpu"), benchmark_iters=1)
    assert ranked[0].checkpoint == focus_winner.resolve()
    assert ranked[1].checkpoint == topline.resolve()
    assert ranked[-1].checkpoint == (outputs / "quick_mhatt" / "best_kws12.pt").resolve()
