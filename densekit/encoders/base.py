"""Encoder interface.

An :class:`Encoder` maps a batch of texts to a dense embedding matrix. densekit
ships a couple of deterministic, offline encoders (see :mod:`densekit.encoders`)
and an optional PyTorch bi-encoder in :mod:`densekit.torch_backend`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from densekit.types import FloatArray


class Encoder(ABC):
    """Abstract text encoder producing ``(n_texts, dim)`` float embeddings."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output embedding dimension."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> FloatArray:
        """Encode a batch of texts into a 2D float array."""

    # Query and corpus sides may diverge for asymmetric models; the default
    # implementation treats them the same, which is correct for symmetric
    # encoders like the hashing encoder.
    def encode_query(self, texts: Sequence[str]) -> FloatArray:
        return self.encode(texts)

    def encode_corpus(self, texts: Sequence[str]) -> FloatArray:
        return self.encode(texts)
