"""Inverted-file (IVF) approximate index.

Vectors are partitioned into ``nlist`` Voronoi cells around k-means centroids.
At query time only the ``nprobe`` nearest cells are scanned, trading a little
recall for a large speed-up over brute force. This is the classic coarse
quantizer behind FAISS' ``IVFFlat``, implemented here in plain NumPy.
"""

from __future__ import annotations

import numpy as np

from densekit.cluster import kmeans
from densekit.index.base import BaseIndex
from densekit.linalg import as_float2d, euclidean_distances, l2_normalize
from densekit.types import FloatArray, Metric, SearchResult


class IVFIndex(BaseIndex):
    """Coarse-quantized approximate nearest-neighbour index."""

    def __init__(
        self,
        dim: int,
        *,
        nlist: int = 64,
        nprobe: int = 8,
        metric: Metric = "cosine",
        seed: int | None = 0,
    ) -> None:
        super().__init__(dim, metric)
        if nlist < 1 or nprobe < 1:
            msg = "nlist and nprobe must be >= 1"
            raise ValueError(msg)
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self.seed = seed
        self._centroids: FloatArray | None = None
        self._vectors: FloatArray = np.empty((0, dim), dtype=np.float32)
        self._cell_of: list[int] = []
        self._trained = False

    @property
    def size(self) -> int:
        return int(self._vectors.shape[0])

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _prepare(self, vectors: np.ndarray) -> FloatArray:
        v = as_float2d(vectors)
        self._check_dim(v)
        return l2_normalize(v) if self.metric == "cosine" else v

    def train(self, vectors: np.ndarray) -> None:
        """Learn the coarse centroids from a representative sample."""
        v = self._prepare(vectors)
        k = min(self.nlist, v.shape[0])
        centroids, _ = kmeans(v, k, seed=self.seed)
        self._centroids = centroids
        self.nlist = k
        self._trained = True

    def add(self, vectors: np.ndarray) -> None:
        if not self._trained or self._centroids is None:
            msg = "call train() before add()"
            raise RuntimeError(msg)
        v = self._prepare(vectors)
        cells = np.argmin(euclidean_distances(v, self._centroids), axis=1)
        self._vectors = np.vstack([self._vectors, v]) if self.size else v
        self._cell_of.extend(int(c) for c in cells)

    def search(self, queries: np.ndarray, k: int = 10) -> SearchResult:
        raise NotImplementedError  # implemented in a follow-up commit
