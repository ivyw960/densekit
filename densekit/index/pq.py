"""Product quantization (PQ).

PQ splits each vector into ``n_subvectors`` contiguous slices and quantizes each
slice against its own small codebook learned with k-means. A vector is then
stored as one byte per subspace, giving large memory savings, and distances are
approximated by table lookups. This module implements the quantizer here; the
:class:`PQIndex` that searches with it is added alongside.
"""

from __future__ import annotations

import numpy as np

from densekit.cluster import kmeans
from densekit.linalg import as_float2d
from densekit.types import FloatArray


class ProductQuantizer:
    """Learn per-subspace codebooks and encode vectors to compact codes."""

    def __init__(
        self,
        dim: int,
        *,
        n_subvectors: int = 8,
        n_bits: int = 8,
        seed: int | None = 0,
    ) -> None:
        if dim % n_subvectors != 0:
            msg = f"dim {dim} must be divisible by n_subvectors {n_subvectors}"
            raise ValueError(msg)
        if not 1 <= n_bits <= 8:
            msg = f"n_bits must be in [1, 8], got {n_bits}"
            raise ValueError(msg)
        self.dim = int(dim)
        self.m = int(n_subvectors)
        self.dsub = dim // n_subvectors
        self.ksub = 1 << n_bits
        self.seed = seed
        self._codebooks: FloatArray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._codebooks is not None

    def _subspace(self, data: FloatArray, s: int) -> FloatArray:
        return data[:, s * self.dsub : (s + 1) * self.dsub]

    def fit(self, data: np.ndarray) -> ProductQuantizer:
        x = as_float2d(data)
        k = min(self.ksub, x.shape[0])
        codebooks = np.zeros((self.m, self.ksub, self.dsub), dtype=np.float32)
        for s in range(self.m):
            centroids, _ = kmeans(self._subspace(x, s), k, seed=self.seed)
            codebooks[s, :k] = centroids
        self._codebooks = codebooks
        return self

    def encode(self, data: np.ndarray) -> np.ndarray:
        if self._codebooks is None:
            msg = "call fit() before encode()"
            raise RuntimeError(msg)
        x = as_float2d(data)
        codes = np.empty((x.shape[0], self.m), dtype=np.uint8)
        for s in range(self.m):
            sub = self._subspace(x, s)
            book = self._codebooks[s]
            dist = np.sum(sub**2, axis=1, keepdims=True) - 2.0 * sub @ book.T
            codes[:, s] = np.argmin(dist, axis=1).astype(np.uint8)
        return codes

    def decode(self, codes: np.ndarray) -> FloatArray:
        if self._codebooks is None:
            msg = "call fit() before decode()"
            raise RuntimeError(msg)
        out = np.empty((codes.shape[0], self.dim), dtype=np.float32)
        for s in range(self.m):
            out[:, s * self.dsub : (s + 1) * self.dsub] = self._codebooks[s][codes[:, s]]
        return out
