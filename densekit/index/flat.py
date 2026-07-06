"""Exact brute-force index.

:class:`FlatIndex` scores every query against every stored vector. It is the
ground-truth baseline the approximate indexes are measured against, and it is
often fast enough on its own for corpora up to a few hundred thousand vectors.
"""

from __future__ import annotations

import numpy as np

from densekit.index.base import BaseIndex
from densekit.linalg import as_float2d, euclidean_distances, l2_normalize
from densekit.types import FloatArray, IntArray, Metric, SearchResult


def _topk(scores: FloatArray, k: int) -> tuple[IntArray, FloatArray]:
    """Return the indices and scores of the top-``k`` columns per row, desc."""
    n = scores.shape[1]
    k = min(k, n)
    rows = np.arange(scores.shape[0])[:, None]
    if k >= n:
        order = np.argsort(-scores, axis=1)
    else:
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        local_order = np.argsort(-scores[rows, part], axis=1)
        order = part[rows, local_order]
    return order.astype(np.int64), scores[rows, order].astype(np.float32)


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
        scores = self._scores(queries)
        indices, top_scores = _topk(scores, k)
        return SearchResult(indices=indices, scores=top_scores)
