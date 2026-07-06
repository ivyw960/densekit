import numpy as np
from densekit.linalg import (
    as_float2d,
    cosine_scores,
    dot_scores,
    euclidean_distances,
    l2_normalize,
)


def test_as_float2d_promotes_vector() -> None:
    v = np.array([1.0, 2.0, 3.0])
    m = as_float2d(v)
    assert m.shape == (1, 3)
    assert m.dtype == np.float32


def test_l2_normalize_unit_norm() -> None:
    x = np.array([[3.0, 4.0], [0.0, 2.0]])
    n = l2_normalize(x)
    norms = np.linalg.norm(n, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_l2_normalize_zero_vector_is_safe() -> None:
    x = np.zeros((1, 5))
    n = l2_normalize(x)
    assert np.all(np.isfinite(n))
    assert np.allclose(n, 0.0)


def test_cosine_matches_manual() -> None:
    a = np.array([[1.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    s = cosine_scores(a, b)
    assert np.allclose(s[0], [1.0, 0.0, 1.0 / np.sqrt(2)], atol=1e-6)


def test_dot_scores_shape() -> None:
    a = np.random.default_rng(0).standard_normal((4, 8))
    b = np.random.default_rng(1).standard_normal((10, 8))
    assert dot_scores(a, b).shape == (4, 10)


def test_euclidean_zero_on_self() -> None:
    a = np.random.default_rng(2).standard_normal((5, 6))
    d = euclidean_distances(a, a)
    assert np.allclose(np.diag(d), 0.0, atol=1e-4)
