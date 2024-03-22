import numpy as np
from densekit.index import FlatIndex, IVFIndex


def _corpus(n: int = 800, dim: int = 32, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def test_add_requires_training() -> None:
    import pytest

    idx = IVFIndex(dim=32)
    with pytest.raises(RuntimeError, match="train"):
        idx.add(_corpus())


def test_ivf_recall_is_high_against_flat() -> None:
    corpus = _corpus()
    queries = _corpus(20, 32, seed=99)

    flat = FlatIndex(dim=32, metric="cosine")
    flat.add(corpus)
    gold = flat.search(queries, k=10).indices

    ivf = IVFIndex(dim=32, nlist=32, nprobe=8, metric="cosine", seed=0)
    ivf.train(corpus)
    ivf.add(corpus)
    approx = ivf.search(queries, k=10).indices

    # recall@10: fraction of true neighbours recovered within the probed cells
    hits = sum(len(set(g) & set(a)) for g, a in zip(gold, approx, strict=True))
    recall = hits / gold.size
    assert recall > 0.7


def test_training_clamps_nlist_to_corpus() -> None:
    idx = IVFIndex(dim=32, nlist=10_000)
    idx.train(_corpus(50, 32))
    assert idx.nlist == 50
