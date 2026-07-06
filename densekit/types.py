"""Shared types and small data containers used across densekit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# Public type aliases. densekit stores embeddings as 2D float arrays with one
# row per item, and returns integer ids alongside float similarity scores.
FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]

# Similarity/distance modes understood by the indexes. "cosine" and "dot" are
# treated as similarities (higher is better); "l2" is a distance that is
# negated internally so that ranking stays "higher is better" everywhere.
Metric = Literal["cosine", "dot", "l2"]

VALID_METRICS: tuple[Metric, ...] = ("cosine", "dot", "l2")


@dataclass(frozen=True)
class SearchResult:
    """Result of a batched top-k search.

    Both arrays are shaped ``(n_queries, k)``. ``indices`` holds the ids of the
    retrieved items and ``scores`` the associated similarity scores, sorted so
    that column 0 is the best match for each query.
    """

    indices: IntArray
    scores: FloatArray

    def __post_init__(self) -> None:
        if self.indices.shape != self.scores.shape:
            msg = (
                "indices and scores must share a shape, got "
                f"{self.indices.shape} and {self.scores.shape}"
            )
            raise ValueError(msg)

    @property
    def n_queries(self) -> int:
        return int(self.indices.shape[0])

    @property
    def k(self) -> int:
        return int(self.indices.shape[1]) if self.indices.ndim == 2 else 0

    def for_query(self, i: int) -> list[tuple[int, float]]:
        """Return ``(id, score)`` pairs for a single query row."""
        pairs = zip(self.indices[i], self.scores[i], strict=False)
        return [(int(j), float(s)) for j, s in pairs]
