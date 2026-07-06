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
from densekit.index.base import BaseIndex
from densekit.linalg import as_float2d, l2_normalize
from densekit.types import FloatArray, Metric, SearchResult


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

    def distance_table(self, query: np.ndarray) -> FloatArray:
        """Squared-distance lookup table of shape ``(m, ksub)`` for one query."""
        if self._codebooks is None:
            msg = "call fit() before distance_table()"
            raise RuntimeError(msg)
        q = as_float2d(query)[0]
        table = np.empty((self.m, self.ksub), dtype=np.float32)
        for s in range(self.m):
            sub = q[s * self.dsub : (s + 1) * self.dsub]
            table[s] = np.sum((self._codebooks[s] - sub) ** 2, axis=1)
        return table

    def asymmetric_distances(self, query: np.ndarray, codes: np.ndarray) -> FloatArray:
        """Approximate squared L2 from ``query`` to each encoded vector (ADC)."""
        table = self.distance_table(query)
        picked = table[np.arange(self.m), codes]  # (n, m)
        return np.asarray(picked.sum(axis=1), dtype=np.float32)


class PQIndex(BaseIndex):
    """Approximate index that stores PQ codes and searches by ADC.

    Only ``"l2"`` and ``"cosine"`` are supported: for unit-norm vectors ranking
    by Euclidean distance is equivalent to ranking by cosine similarity, so the
    cosine case simply normalizes inputs before quantizing.
    """

    def __init__(
        self,
        dim: int,
        *,
        n_subvectors: int = 8,
        n_bits: int = 8,
        metric: Metric = "l2",
        seed: int | None = 0,
    ) -> None:
        if metric == "dot":
            msg = "PQIndex supports 'l2' and 'cosine' only"
            raise ValueError(msg)
        super().__init__(dim, metric)
        self.pq = ProductQuantizer(dim, n_subvectors=n_subvectors, n_bits=n_bits, seed=seed)
        self._codes = np.empty((0, self.pq.m), dtype=np.uint8)

    @property
    def size(self) -> int:
        return int(self._codes.shape[0])

    def _prepare(self, vectors: np.ndarray) -> FloatArray:
        v = as_float2d(vectors)
        self._check_dim(v)
        return l2_normalize(v) if self.metric == "cosine" else v

    def train(self, vectors: np.ndarray) -> None:
        self.pq.fit(self._prepare(vectors))

    def add(self, vectors: np.ndarray) -> None:
        if not self.pq.is_fitted:
            msg = "call train() before add()"
            raise RuntimeError(msg)
        codes = self.pq.encode(self._prepare(vectors))
        self._codes = np.vstack([self._codes, codes]) if self.size else codes

    def search(self, queries: np.ndarray, k: int = 10) -> SearchResult:
        if self.size == 0:
            msg = "cannot search an empty index"
            raise RuntimeError(msg)
        q = self._prepare(queries)
        k_eff = min(k, self.size)
        all_idx = np.empty((q.shape[0], k_eff), dtype=np.int64)
        all_scores = np.empty((q.shape[0], k_eff), dtype=np.float32)
        for i in range(q.shape[0]):
            # negate distance so higher is better, matching the other indexes
            scores = -self.pq.asymmetric_distances(q[i : i + 1], self._codes)
            local = np.argpartition(-scores, k_eff - 1)[:k_eff]
            local = local[np.argsort(-scores[local])]
            all_idx[i] = local
            all_scores[i] = scores[local]
        return SearchResult(indices=all_idx, scores=all_scores)
