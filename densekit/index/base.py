"""Base class shared by every densekit index.

An index stores a corpus of embeddings and answers top-k nearest-neighbour
queries. Concrete subclasses implement exact search (:class:`FlatIndex`) or one
of the approximate strategies (IVF, LSH, PQ).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from densekit.types import Metric, SearchResult, VALID_METRICS


class BaseIndex(ABC):
    """Abstract nearest-neighbour index over a fixed-dimensional corpus."""

    def __init__(self, dim: int, metric: Metric = "cosine") -> None:
        if metric not in VALID_METRICS:
            msg = f"unknown metric {metric!r}, expected one of {VALID_METRICS}"
            raise ValueError(msg)
        self._dim = int(dim)
        self.metric: Metric = metric

    @property
    def dim(self) -> int:
        """Embedding dimension the index was built for."""
        return self._dim

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of vectors currently stored."""

    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """Add a batch of ``(n, dim)`` vectors to the index."""

    @abstractmethod
    def search(self, queries: np.ndarray, k: int = 10) -> SearchResult:
        """Return the top-``k`` matches for each query row."""

    def _check_dim(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            msg = f"expected vectors with {self._dim} columns, got shape {vectors.shape}"
            raise ValueError(msg)
