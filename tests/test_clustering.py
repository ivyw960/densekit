import numpy as np
from densekit.clustering import kmeans


def _blobs(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)
    pts = [c + rng.standard_normal((40, 2)) * 0.3 for c in centers]
    return np.vstack(pts).astype(np.float32)


def test_kmeans_recovers_blobs() -> None:
    data = _blobs()
    centroids, assign = kmeans(data, n_clusters=3, seed=1)
    assert centroids.shape == (3, 2)
    # each of the three 40-point blobs should land in one cluster
    counts = np.bincount(assign, minlength=3)
    assert sorted(counts.tolist()) == [40, 40, 40]


def test_kmeans_is_reproducible() -> None:
    data = _blobs(2)
    a, _ = kmeans(data, n_clusters=3, seed=7)
    b, _ = kmeans(data, n_clusters=3, seed=7)
    assert np.allclose(a, b)


def test_kmeans_rejects_too_many_clusters() -> None:
    import pytest

    with pytest.raises(ValueError, match="n_clusters"):
        kmeans(np.zeros((3, 2)), n_clusters=5)
