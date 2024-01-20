"""Deterministic hashing encoder.

The hashing encoder turns text into a fixed-width embedding with the *hashing
trick*: every token is hashed to a bucket and a sign, and collisions are simply
summed. It needs no training data, no model download and no external services,
which makes it a convenient offline default and a stable fixture for tests.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from densekit.encoders.base import Encoder
from densekit.linalg import l2_normalize
from densekit.types import FloatArray


def _hash_token(token: str) -> int:
    """Stable 64-bit hash of a token (independent of PYTHONHASHSEED)."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


class HashingEncoder(Encoder):
    """Feature-hashing encoder over whitespace tokens.

    Parameters
    ----------
    dim:
        Width of the output embedding.
    normalize:
        L2-normalize each row, so cosine and dot product coincide.
    lowercase:
        Lowercase text before tokenizing.
    """

    def __init__(self, dim: int = 256, *, normalize: bool = True, lowercase: bool = True) -> None:
        if dim <= 0:
            msg = f"dim must be positive, got {dim}"
            raise ValueError(msg)
        self._dim = int(dim)
        self.normalize = normalize
        self.lowercase = lowercase

    @property
    def dim(self) -> int:
        return self._dim

    def _tokens(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        return text.split()

    def encode(self, texts: Sequence[str]) -> FloatArray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                h = _hash_token(token)
                bucket = (h >> 1) % self._dim
                sign = 1.0 if (h & 1) else -1.0
                out[row, bucket] += sign
        if self.normalize:
            out = l2_normalize(out)
        return out
