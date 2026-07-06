import numpy as np
from densekit.index import FlatIndex, LSHIndex


def _corpus(n: int = 600, dim: int = 32, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def test_lsh_finds_nearest_neighbours() -> None:
    corpus = _corpus()
    queries = _corpus(15, 32, seed=42)

    flat = FlatIndex(dim=32, metric="cosine")
    flat.add(corpus)
    gold = flat.search(queries, k=5).indices

    lsh = LSHIndex(dim=32, n_bits=12, n_tables=12, metric="cosine", seed=0)
    lsh.add(corpus)
    approx = lsh.search(queries, k=5).indices

    hits = sum(len(set(g) & set(a)) for g, a in zip(gold, approx, strict=True))
    assert hits / gold.size > 0.6


def test_lsh_rejects_bad_bits() -> None:
    import pytest

    with pytest.raises(ValueError, match="n_bits"):
        LSHIndex(dim=8, n_bits=0)


def test_lsh_search_empty_raises() -> None:
    import pytest

    lsh = LSHIndex(dim=8)
    with pytest.raises(RuntimeError, match="empty index"):
        lsh.search(_corpus(1, 8), k=1)
