"""Pooling strategies that turn token embeddings into a single vector."""

from __future__ import annotations

import torch


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mask-aware mean over the sequence dimension.

    ``hidden`` is ``(batch, seq, dim)`` and ``mask`` is ``(batch, seq)`` with 1
    for real tokens and 0 for padding. Padded positions are excluded from the
    average; empty sequences fall back to a zero vector.
    """
    mask_f = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1.0)
    return summed / counts


def cls_pool(hidden: torch.Tensor) -> torch.Tensor:
    """Take the first token's embedding (CLS-style pooling)."""
    return hidden[:, 0]
