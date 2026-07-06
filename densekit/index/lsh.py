"""Random-hyperplane LSH index.

Each hash table projects vectors onto ``n_bits`` random hyperplanes and keeps
the sign pattern as a bucket key. Similar vectors collide with high probability
under the angular (cosine) distance, so a query only reranks the handful of
vectors sharing -- or nearly sharing -- its bucket across a few tables.
"""

from __future__ import annotations

import numpy as np

from densekit.index.base import BaseIndex
from densekit.linalg import as_float2d, euclidean_distances, l2_normalize
from densekit.types import FloatArray, Metric, SearchResult
from densekit.utils import make_rng


class LSHIndex(BaseIndex):
    """Multi-table signed-random-projection LSH with multi-probe search."""

    def __init__(
        self,
        dim: int,
        *,
        n_bits: int = 16,
        n_tables: int = 8,
        metric: Metric = "cosine",
        seed: int | None = 0,
        candidate_budget: int | None = None,
    ) -> None:
        super().__init__(dim, metric)
        if not 1 <= n_bits <= 62:
            msg = f"n_bits must be in [1, 62], got {n_bits}"
            raise ValueError(msg)
        if n_tables < 1:
            msg = "n_tables must be >= 1"
            raise ValueError(msg)
        self.n_bits = int(n_bits)
        self.n_tables = int(n_tables)
        self.candidate_budget = candidate_budget
        rng = make_rng(seed)
        self._planes = [
            rng.standard_normal((dim, n_bits)).astype(np.float32) for _ in range(n_tables)
        ]
        self._pow2 = (1 << np.arange(n_bits, dtype=np.int64)).astype(np.int64)
        self._tables: list[dict[int, list[int]]] = [{} for _ in range(n_tables)]
        self._vectors: FloatArray = np.empty((0, dim), dtype=np.float32)

    @property
    def size(self) -> int:
        return int(self._vectors.shape[0])

    def _prepare(self, vectors: np.ndarray) -> FloatArray:
        v = as_float2d(vectors)
        self._check_dim(v)
        return l2_normalize(v) if self.metric == "cosine" else v

    def _codes(self, vectors: FloatArray, table: int) -> np.ndarray:
        bits = (vectors @ self._planes[table] > 0).astype(np.int64)
        return bits @ self._pow2

    def add(self, vectors: np.ndarray) -> None:
        v = self._prepare(vectors)
        start = self.size
        self._vectors = np.vstack([self._vectors, v]) if start else v
        for t in range(self.n_tables):
            codes = self._codes(v, t)
            table = self._tables[t]
            for offset, code in enumerate(codes):
                table.setdefault(int(code), []).append(start + offset)

    def _order_buckets(self, keys: list[int], query_code: int) -> list[int]:
        """Bucket keys ordered by Hamming distance to the query code.

        The Hamming distances are computed for all buckets at once: XOR against
        the query code, then a vectorized byte-wise popcount via ``unpackbits``.
        """
        if not keys:
            return []
        arr = np.asarray(keys, dtype=np.int64)
        xor = (arr ^ np.int64(query_code)).astype(np.uint64)
        counts = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        order = np.argsort(counts, kind="stable")
        return [int(arr[i]) for i in order]

    def _score(self, query: FloatArray, candidates: FloatArray) -> FloatArray:
        if self.metric == "l2":
            return np.asarray(-euclidean_distances(query, candidates)[0], dtype=np.float32)
        return np.asarray((candidates @ query.T).ravel(), dtype=np.float32)

    def search(self, queries: np.ndarray, k: int = 10) -> SearchResult:
        if self.size == 0:
            msg = "cannot search an empty index"
            raise RuntimeError(msg)
        q = self._prepare(queries)
        budget = self.candidate_budget or max(10 * k, 100)

        all_idx = np.full((q.shape[0], k), -1, dtype=np.int64)
        all_scores = np.full((q.shape[0], k), -np.inf, dtype=np.float32)
        for i in range(q.shape[0]):
            # Rank each table's buckets by Hamming distance to the query, then
            # gather them breadth-first (nearest bucket of every table, then the
            # next-nearest, ...) so no single table dominates the candidate set.
            orders = [
                self._order_buckets(list(self._tables[t]), int(self._codes(q[i : i + 1], t)[0]))
                for t in range(self.n_tables)
            ]
            found: set[int] = set()
            depth = 0
            while len(found) < budget and any(depth < len(o) for o in orders):
                for t, order in enumerate(orders):
                    if depth < len(order):
                        found.update(self._tables[t][order[depth]])
                depth += 1
            candidates = np.fromiter(found, dtype=np.int64)
            if candidates.size == 0:
                continue
            scores = self._score(q[i : i + 1], self._vectors[candidates])
            k_eff = min(k, candidates.size)
            local = np.argpartition(-scores, k_eff - 1)[:k_eff]
            local = local[np.argsort(-scores[local])]
            all_idx[i, :k_eff] = candidates[local]
            all_scores[i, :k_eff] = scores[local]
        return SearchResult(indices=all_idx, scores=all_scores)
