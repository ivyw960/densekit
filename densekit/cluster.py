"""A small, dependency-free k-means used by the IVF and PQ indexes.

This is Lloyd's algorithm with k-means++ seeding. It is not meant to compete
with scikit-learn on speed; it exists so the core of densekit can quantize
vectors with NumPy alone.
"""

from __future__ import annotations

import numpy as np

from densekit.linalg import as_float2d, euclidean_distances
from densekit.types import FloatArray, IntArray
from densekit.utils import make_rng


def _kmeanspp_init(data: FloatArray, k: int, rng: np.random.Generator) -> FloatArray:
    """Pick ``k`` seeds with the k-means++ D^2 sampling scheme."""
    n = data.shape[0]
    first = int(rng.integers(n))
    centroids = [data[first]]
    closest_sq = np.sum((data - centroids[0]) ** 2, axis=1)
    for _ in range(1, k):
        total = float(closest_sq.sum())
        if total <= 0.0:
            centroids.append(data[int(rng.integers(n))])
            continue
        probs = closest_sq / total
        chosen = int(rng.choice(n, p=probs))
        centroids.append(data[chosen])
        dist_sq = np.sum((data - data[chosen]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, dist_sq)
    return np.asarray(centroids, dtype=np.float32)


def _assign(data: FloatArray, centroids: FloatArray) -> IntArray:
    return np.asarray(np.argmin(euclidean_distances(data, centroids), axis=1), dtype=np.int64)


def kmeans(
    data: np.ndarray,
    n_clusters: int,
    *,
    n_iter: int = 25,
    seed: int | None = 0,
    tol: float = 1e-4,
) -> tuple[FloatArray, IntArray]:
    """Cluster ``data`` into ``n_clusters`` groups.

    Returns the ``(n_clusters, dim)`` centroids and the per-row assignments.
    Empty clusters are re-seeded from a random point to keep exactly
    ``n_clusters`` centroids alive.
    """
    x = as_float2d(data)
    n = x.shape[0]
    if not 1 <= n_clusters <= n:
        msg = f"n_clusters must be in [1, {n}], got {n_clusters}"
        raise ValueError(msg)

    rng = make_rng(seed)
    centroids = _kmeanspp_init(x, n_clusters, rng)
    assignments = _assign(x, centroids)
    for _ in range(n_iter):
        new_centroids = np.empty_like(centroids)
        for c in range(n_clusters):
            members = x[assignments == c]
            new_centroids[c] = members.mean(axis=0) if len(members) else x[int(rng.integers(n))]
        shift = float(np.linalg.norm(new_centroids - centroids))
        centroids = new_centroids
        assignments = _assign(x, centroids)
        if shift <= tol:
            break
    return centroids, assignments
