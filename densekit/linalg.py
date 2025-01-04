"""Small linear-algebra helpers built on NumPy.

These are the numerical primitives shared by the encoders and indexes. They are
deliberately dependency-free so the core of densekit runs with NumPy alone.
"""

from __future__ import annotations

import numpy as np

from densekit.types import FloatArray, IntArray


def as_float2d(x: np.ndarray) -> FloatArray:
    """Coerce input to a contiguous 2D float32 array.

    A 1D vector is treated as a single row. This is the canonical shape used
    for embedding matrices throughout densekit.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        msg = f"expected 1D or 2D array, got shape {arr.shape}"
        raise ValueError(msg)
    return np.ascontiguousarray(arr)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> FloatArray:
    """L2-normalize along ``axis``; zero vectors are left as zeros."""
    arr = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    return np.asarray(arr / np.maximum(norm, eps), dtype=np.float32)


def dot_scores(queries: np.ndarray, corpus: np.ndarray) -> FloatArray:
    """Inner-product similarity matrix of shape ``(n_queries, n_corpus)``."""
    q = as_float2d(queries)
    c = as_float2d(corpus)
    return np.asarray(q @ c.T, dtype=np.float32)


def cosine_scores(queries: np.ndarray, corpus: np.ndarray) -> FloatArray:
    """Cosine-similarity matrix of shape ``(n_queries, n_corpus)``."""
    q = l2_normalize(as_float2d(queries))
    c = l2_normalize(as_float2d(corpus))
    return np.asarray(q @ c.T, dtype=np.float32)


def euclidean_distances(queries: np.ndarray, corpus: np.ndarray) -> FloatArray:
    """Pairwise Euclidean distances of shape ``(n_queries, n_corpus)``.

    Uses the expanded ``||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b`` form, which is
    a single matmul plus two norm vectors. Tiny negatives from rounding are
    clipped before the square root.
    """
    q = as_float2d(queries)
    c = as_float2d(corpus)
    q_sq = np.sum(q * q, axis=1, keepdims=True)
    c_sq = np.sum(c * c, axis=1, keepdims=True).T
    sq = q_sq + c_sq - 2.0 * (q @ c.T)
    return np.asarray(np.sqrt(np.maximum(sq, 0.0)), dtype=np.float32)


def top_k(scores: FloatArray, k: int) -> tuple[IntArray, FloatArray]:
    """Indices and scores of the top-``k`` columns per row, sorted descending.

    Uses ``argpartition`` to find the top ``k`` in linear time, then sorts only
    those ``k`` entries. ``k`` is clamped to the number of columns.
    """
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
