"""Passive user adaptation for realtime lightweight KWS."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


DEFAULT_USER_PROFILE_PATH = Path.home() / ".kws_demo" / "user_profile.pt"


def blend_keyword_score(
    base_conf: float,
    wake_prob: float,
    prototype_similarity: float,
    *,
    prototype_bonus_cap: float = 0.04,
) -> float:
    support = max(0.0, (float(prototype_similarity) + 1.0) * 0.5)
    bonus_cap = float(min(0.12, max(0.0, prototype_bonus_cap)))
    combined = (0.88 * float(base_conf)) + (0.08 * float(wake_prob)) + (bonus_cap * support)
    upper = min(1.0, float(base_conf) + bonus_cap)
    return float(min(upper, max(float(base_conf), combined)))


class PassiveKeywordProfile:
    def __init__(
        self,
        *,
        path: str | Path = DEFAULT_USER_PROFILE_PATH,
        enabled: bool = True,
        max_prototypes: int = 5,
        save_delay_seconds: float = 0.25,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.enabled = bool(enabled)
        self.max_prototypes = int(max(1, max_prototypes))
        self.save_delay_seconds = float(max(0.0, save_delay_seconds))
        self._prototypes: Dict[str, List[torch.Tensor]] = {}
        self._dirty = False
        self._lock = threading.Lock()
        self._save_event = threading.Event()
        self._stop_event = threading.Event()
        self._writer: threading.Thread | None = None
        self._load()
        if self.enabled:
            self._writer = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer.start()

    def reset(self) -> None:
        with self._lock:
            self._prototypes.clear()
            self._dirty = False
        if self.path.exists():
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def similarity(self, label: str | None, embedding: torch.Tensor | np.ndarray | None) -> float:
        if not self.enabled or not label or embedding is None:
            return 0.0
        with self._lock:
            refs = list(self._prototypes.get(str(label), []))
        if not refs:
            return 0.0

        emb = torch.as_tensor(embedding, dtype=torch.float32).reshape(-1)
        emb = torch.nn.functional.normalize(emb, dim=0)
        sims = []
        for ref in refs:
            ref_norm = torch.nn.functional.normalize(ref.reshape(-1), dim=0)
            sims.append(float(torch.dot(emb, ref_norm).item()))
        return float(max(sims) if sims else 0.0)

    def update(self, label: str | None, embedding: torch.Tensor | np.ndarray | None) -> None:
        if not self.enabled or not label or embedding is None:
            return
        emb = torch.as_tensor(embedding, dtype=torch.float32).detach().cpu().reshape(-1)
        with self._lock:
            refs = list(self._prototypes.get(str(label), []))
            refs.append(emb)
            self._prototypes[str(label)] = refs[-self.max_prototypes :]
            self._dirty = True
        self._save_event.set()

    def flush(self) -> None:
        snapshot = self._snapshot_for_save()
        if snapshot is not None:
            self._save(snapshot)

    def close(self) -> None:
        if self._writer is not None:
            self._stop_event.set()
            self._save_event.set()
            self._writer.join(timeout=2.0)
            self._writer = None
        self.flush()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = torch.load(self.path, map_location="cpu", weights_only=False)
        except Exception:
            broken_path = self.path.with_suffix(f"{self.path.suffix}.corrupt-{int(time.time())}")
            try:
                self.path.replace(broken_path)
            except Exception:
                pass
            self._prototypes = {}
            return
        if not isinstance(payload, dict):
            return
        loaded: Dict[str, List[torch.Tensor]] = {}
        for key, value in payload.items():
            if not isinstance(key, str):
                continue
            vectors = []
            if isinstance(value, list):
                for item in value:
                    vectors.append(torch.as_tensor(item, dtype=torch.float32).reshape(-1))
            if vectors:
                loaded[key] = vectors[-self.max_prototypes :]
        self._prototypes = loaded

    def _snapshot_for_save(self) -> Dict[str, List[torch.Tensor]] | None:
        with self._lock:
            if not self._dirty:
                return None
            snapshot = {
                label: [vec.cpu().clone() for vec in refs[-self.max_prototypes :]]
                for label, refs in self._prototypes.items()
                if refs
            }
            self._dirty = False
        return snapshot

    def _writer_loop(self) -> None:
        while True:
            self._save_event.wait()
            self._save_event.clear()
            if self._stop_event.is_set():
                break

            deadline = time.monotonic() + self.save_delay_seconds
            while not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                if self._save_event.wait(timeout=remaining):
                    self._save_event.clear()
                    deadline = time.monotonic() + self.save_delay_seconds

            snapshot = self._snapshot_for_save()
            if snapshot is not None:
                self._save(snapshot)

        snapshot = self._snapshot_for_save()
        if snapshot is not None:
            self._save(snapshot)

    def _save(self, serializable: Dict[str, List[torch.Tensor]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = -1
        tmp_path = None
        try:
            fd, raw_tmp = tempfile.mkstemp(prefix=f"{self.path.stem}.", suffix=".tmp", dir=str(self.path.parent))
            tmp_path = Path(raw_tmp)
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                torch.save(serializable, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        finally:
            if fd >= 0:
                os.close(fd)
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass
