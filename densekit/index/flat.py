"""Exact brute-force index.

:class:`FlatIndex` scores every query against every stored vector. It is the
ground-truth baseline the approximate indexes are measured against, and it is
often fast enough on its own for corpora up to a few hundred thousand vectors.
"""

from __future__ import annotations

import numpy as np

from densekit.index.base import BaseIndex
from densekit.linalg import as_float2d, euclidean_distances, l2_normalize, top_k
from densekit.types import FloatArray, Metric, SearchResult


class FlatIndex(BaseIndex):
    """Exact nearest-neighbour search over a dense matrix."""

    def __init__(self, dim: int, metric: Metric = "cosine") -> None:
        super().__init__(dim, metric)
        self._vectors: FloatArray = np.empty((0, dim), dtype=np.float32)

    @property
    def size(self) -> int:
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        v = as_float2d(vectors)
        self._check_dim(v)
        if self.metric == "cosine":
            v = l2_normalize(v)
        self._vectors = np.vstack([self._vectors, v]) if self.size else v

    def _scores(self, queries: np.ndarray) -> FloatArray:
        q = as_float2d(queries)
        if self.metric == "cosine":
            # cosine == dot product on unit vectors, which is how we store them.
            return np.asarray(l2_normalize(q) @ self._vectors.T, dtype=np.float32)
        if self.metric == "dot":
            return np.asarray(q @ self._vectors.T, dtype=np.float32)
        # l2: rank by *negative* distance so that "higher is better" holds.
        return np.asarray(-euclidean_distances(q, self._vectors), dtype=np.float32)

    def search(self, queries: np.ndarray, k: int = 10) -> SearchResult:
        if self.size == 0:
            msg = "cannot search an empty index"
            raise RuntimeError(msg)
        q = as_float2d(queries)
        if q.shape[0] == 0:
            # an empty query batch yields an empty, well-shaped result
            return SearchResult(
                indices=np.empty((0, 0), dtype=np.int64),
                scores=np.empty((0, 0), dtype=np.float32),
            )
        scores = self._scores(q)
        indices, top_scores = top_k(scores, k)
        return SearchResult(indices=indices, scores=top_scores)
