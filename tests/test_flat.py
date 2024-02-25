import numpy as np

from densekit.index import FlatIndex


def _corpus(n: int = 50, dim: int = 16, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def test_add_and_size() -> None:
    idx = FlatIndex(dim=16)
    idx.add(_corpus())
    assert idx.size == 50


def test_self_query_returns_itself_first() -> None:
    corpus = _corpus()
    idx = FlatIndex(dim=16, metric="cosine")
    idx.add(corpus)
    res = idx.search(corpus[:5], k=1)
    assert list(res.indices[:, 0]) == [0, 1, 2, 3, 4]


def test_topk_scores_are_sorted_descending() -> None:
    idx = FlatIndex(dim=16)
    idx.add(_corpus())
    res = idx.search(_corpus(5, 16, seed=1), k=10)
    for row in res.scores:
        assert np.all(np.diff(row) <= 1e-6)


def test_k_larger_than_corpus_is_clamped() -> None:
    idx = FlatIndex(dim=8)
    idx.add(_corpus(3, 8))
    res = idx.search(_corpus(2, 8, seed=3), k=10)
    assert res.indices.shape == (2, 3)


def test_search_empty_index_raises() -> None:
    import pytest

    idx = FlatIndex(dim=8)
    with pytest.raises(RuntimeError, match="empty index"):
        idx.search(_corpus(1, 8), k=1)
