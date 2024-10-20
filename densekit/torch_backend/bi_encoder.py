"""A tiny, self-contained PyTorch bi-encoder.

There are no model downloads: text is tokenized by hashing whitespace tokens
into a fixed vocabulary, embedded, pooled and projected. It is deliberately
small -- enough to demonstrate contrastive training end to end and to fine-tune
on a laptop -- while sharing the offline, deterministic spirit of the NumPy core.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from torch import nn

from densekit.torch_backend.pooling import cls_pool, mean_pool

Pooling = Literal["mean", "cls"]


def _token_id(token: str, vocab_size: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    # reserve id 0 for padding
    return 1 + int.from_bytes(digest, "little") % (vocab_size - 1)


class HashingTokenizer:
    """Deterministic whitespace tokenizer that hashes tokens to ids."""

    def __init__(self, vocab_size: int = 1 << 16, max_length: int = 64) -> None:
        self.vocab_size = vocab_size
        self.max_length = max_length

    def __call__(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = [t.lower().split()[: self.max_length] for t in texts]
        width = max((len(tokens) for tokens in batch), default=1)
        ids = torch.zeros((len(batch), width), dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, tokens in enumerate(batch):
            for col, token in enumerate(tokens):
                ids[row, col] = _token_id(token, self.vocab_size)
                mask[row, col] = 1
        return ids, mask


class BiEncoder(nn.Module):
    """Embedding-bag style dual encoder with mean or CLS pooling."""

    def __init__(
        self,
        vocab_size: int = 1 << 16,
        dim: int = 128,
        *,
        pooling: Pooling = "mean",
        max_length: int = 64,
    ) -> None:
        super().__init__()
        self.tokenizer = HashingTokenizer(vocab_size, max_length)
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.projection = nn.Linear(dim, dim)
        self.pooling: Pooling = pooling

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        pooled = cls_pool(hidden) if self.pooling == "cls" else mean_pool(hidden, mask)
        projected = self.projection(pooled)
        return torch.nn.functional.normalize(projected, dim=-1)

    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        ids, mask = self.tokenizer(texts)
        device = next(self.parameters()).device
        return self(ids.to(device), mask.to(device))

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts to a detached NumPy array (inference helper)."""
        self.eval()
        return self.embed(texts).cpu().numpy()
