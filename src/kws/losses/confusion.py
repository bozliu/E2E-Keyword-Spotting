"""Confusion-aware auxiliary losses for keyword separation."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from kws.constants import COMMAND31_TO_INDEX, IGNORE_INDEX


def confusion_aware_embedding_loss(
    embedding: torch.Tensor,
    command_labels: torch.Tensor,
    *,
    confusion_groups: Mapping[str, Sequence[str]],
    margin: float = 0.20,
) -> torch.Tensor:
    valid = command_labels != IGNORE_INDEX
    if int(valid.sum()) < 2:
        return embedding.new_tensor(0.0)

    emb = F.normalize(embedding[valid], dim=-1)
    labels = command_labels[valid]
    sims = emb @ emb.t()
    n = sims.size(0)
    eye = torch.eye(n, device=sims.device, dtype=torch.bool)

    same_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & (~eye)
    if bool(same_mask.any()):
        same_loss = 1.0 - sims[same_mask].mean()
    else:
        same_loss = embedding.new_tensor(0.0)

    pair_masks = []
    for keyword, confusions in confusion_groups.items():
        kw_idx = COMMAND31_TO_INDEX.get(str(keyword))
        if kw_idx is None:
            continue
        for other in confusions:
            other_idx = COMMAND31_TO_INDEX.get(str(other))
            if other_idx is None or other_idx == kw_idx:
                continue
            pair_masks.append((labels.unsqueeze(1) == kw_idx) & (labels.unsqueeze(0) == other_idx))
            pair_masks.append((labels.unsqueeze(1) == other_idx) & (labels.unsqueeze(0) == kw_idx))

    if pair_masks:
        confusion_mask = torch.stack(pair_masks, dim=0).any(dim=0)
        confusion_mask &= ~eye
        confusion_loss = torch.relu(sims[confusion_mask] - margin).mean() if bool(confusion_mask.any()) else embedding.new_tensor(0.0)
    else:
        confusion_loss = embedding.new_tensor(0.0)

    return 0.5 * (same_loss + confusion_loss)

