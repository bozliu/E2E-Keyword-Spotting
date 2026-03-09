from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module(name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_check_hf_space_extracts_runtime_sha_from_list_replica() -> None:
    module = _load_script_module("check_hf_space.py")
    raw = {"replicas": [{"sha": "abc123"}]}
    assert module._extract_runtime_sha(raw) == "abc123"


def test_check_hf_space_extracts_runtime_sha_from_dict_replica() -> None:
    module = _load_script_module("check_hf_space.py")
    raw = {"replicas": {"sha": "def456"}}
    assert module._extract_runtime_sha(raw) == "def456"


def test_cutover_hf_space_extracts_runtime_sha_from_raw_fallback() -> None:
    module = _load_script_module("cutover_hf_space.py")
    raw = {"sha": "ghi789"}
    assert module._extract_runtime_sha(raw) == "ghi789"
