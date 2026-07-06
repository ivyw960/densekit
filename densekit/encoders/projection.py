"""Random-projection encoder.

Wraps another encoder and projects its output onto a lower-dimensional space
with a fixed Gaussian matrix. By the Johnson-Lindenstrauss lemma this roughly
preserves pairwise distances while shrinking the embedding, which is handy for
turning a wide hashing encoder into a compact, index-friendly vector.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from densekit.encoders.base import Encoder
from densekit.linalg import l2_normalize
from densekit.types import FloatArray
from densekit.utils import make_rng


class RandomProjectionEncoder(Encoder):
    """Project a base encoder's output to ``out_dim`` with a random matrix."""

    def __init__(
        self,
        base: Encoder,
        out_dim: int,
        *,
        seed: int | None = 0,
        normalize: bool = True,
    ) -> None:
        if out_dim <= 0:
            msg = f"out_dim must be positive, got {out_dim}"
            raise ValueError(msg)
        self.base = base
        self._dim = int(out_dim)
        self.normalize = normalize
        rng = make_rng(seed)
        scale = 1.0 / np.sqrt(out_dim)
        self._proj = (rng.standard_normal((base.dim, out_dim)) * scale).astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> FloatArray:
        projected = self.base.encode(texts) @ self._proj
        if self.normalize:
            projected = l2_normalize(projected)
        return np.asarray(projected, dtype=np.float32)
