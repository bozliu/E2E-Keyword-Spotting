from __future__ import annotations

import os

import pytest

from kws.demo.realtime import DemoInstanceLock


def test_demo_instance_lock_reclaims_stale_file(tmp_path, monkeypatch) -> None:
    lock_path = tmp_path / "realtime.lock"
    lock_path.write_text("123\n", encoding="utf-8")
    monkeypatch.setattr("kws.demo.realtime._pid_is_running", lambda pid: False)

    lock = DemoInstanceLock(path=lock_path)
    lock.acquire()

    assert lock_path.exists()
    assert lock_path.read_text(encoding="utf-8").strip() == str(os.getpid())

    lock.release()
    assert not lock_path.exists()


def test_demo_instance_lock_rejects_live_owner(tmp_path, monkeypatch) -> None:
    lock_path = tmp_path / "realtime.lock"
    lock_path.write_text("456\n", encoding="utf-8")
    monkeypatch.setattr("kws.demo.realtime._pid_is_running", lambda pid: pid == 456)

    lock = DemoInstanceLock(path=lock_path)

    with pytest.raises(RuntimeError, match="Another demo instance is already running"):
        lock.acquire()

    assert lock_path.exists()


def test_demo_instance_lock_release_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("kws.demo.realtime._pid_is_running", lambda pid: False)
    lock = DemoInstanceLock(path=tmp_path / "realtime.lock")
    lock.acquire()

    lock.release()
    lock.release()
