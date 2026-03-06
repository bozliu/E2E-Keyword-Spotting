"""Optional prototype-style auxiliary loss inspired by triplet/cosine objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kws.constants import IGNORE_INDEX


def prototype_cosine_loss(
    embedding: torch.Tensor,
    command_labels: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Encourage same-class embeddings to be closer than different-class embeddings."""
    valid = command_labels != IGNORE_INDEX
    if valid.sum() < 4:
        return embedding.new_tensor(0.0)

    emb = F.normalize(embedding[valid], dim=-1)
    labels = command_labels[valid]

    sims = emb @ emb.t()
    n = sims.size(0)
    eye = torch.eye(n, device=sims.device, dtype=torch.bool)

    same = (labels.unsqueeze(1) == labels.unsqueeze(0)) & (~eye)
    diff = labels.unsqueeze(1) != labels.unsqueeze(0)

    if same.sum() == 0 or diff.sum() == 0:
        return embedding.new_tensor(0.0)

    pos = sims[same].mean()
    neg = sims[diff].mean()
    return torch.relu(margin - pos + neg)
