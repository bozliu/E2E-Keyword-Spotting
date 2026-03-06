from __future__ import annotations

import torch

from kws.demo.user_profile import PassiveKeywordProfile, blend_keyword_score


def test_passive_keyword_profile_caps_history(tmp_path) -> None:
    profile = PassiveKeywordProfile(path=tmp_path / "profile.pt", enabled=True, max_prototypes=2)
    profile.update("yes", torch.tensor([1.0, 0.0]))
    profile.update("yes", torch.tensor([0.9, 0.1]))
    profile.update("yes", torch.tensor([0.8, 0.2]))
    profile.close()

    reloaded = PassiveKeywordProfile(path=tmp_path / "profile.pt", enabled=True, max_prototypes=2)
    sim = reloaded.similarity("yes", torch.tensor([0.85, 0.15]))
    reloaded.close()
    assert sim > 0.95


def test_blend_keyword_score_only_small_boost() -> None:
    boosted = blend_keyword_score(0.70, 0.92, 0.98)
    assert boosted >= 0.70
    assert boosted <= 0.78


def test_blend_keyword_score_respects_bonus_cap() -> None:
    base = blend_keyword_score(0.70, 0.92, 0.98, prototype_bonus_cap=0.04)
    boosted = blend_keyword_score(0.70, 0.92, 0.98, prototype_bonus_cap=0.08)
    assert boosted >= base
    assert boosted <= 0.82


def test_passive_keyword_profile_tolerates_corrupt_cache(tmp_path) -> None:
    path = tmp_path / "profile.pt"
    path.write_bytes(b"not-a-valid-torch-file")

    profile = PassiveKeywordProfile(path=path, enabled=True, max_prototypes=2)

    assert profile.similarity("yes", torch.tensor([1.0, 0.0])) == 0.0
    profile.close()
    corrupt_backups = list(tmp_path.glob("profile.pt.corrupt-*"))
    assert corrupt_backups


def test_passive_keyword_profile_close_flushes_async_updates(tmp_path) -> None:
    path = tmp_path / "profile.pt"
    profile = PassiveKeywordProfile(path=path, enabled=True, max_prototypes=2, save_delay_seconds=10.0)
    profile.update("yes", torch.tensor([1.0, 0.0]))

    assert not path.exists()

    profile.close()

    reloaded = PassiveKeywordProfile(path=path, enabled=True, max_prototypes=2)
    sim = reloaded.similarity("yes", torch.tensor([1.0, 0.0]))
    reloaded.close()
    assert sim > 0.99


def test_passive_keyword_profile_close_is_idempotent(tmp_path) -> None:
    profile = PassiveKeywordProfile(path=tmp_path / "profile.pt", enabled=True, max_prototypes=2)
    profile.update("yes", torch.tensor([1.0, 0.0]))

    profile.close()
    profile.close()
