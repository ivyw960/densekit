"""Cross-index sanity checks: every approximate index should mostly agree with
the exact FlatIndex baseline on the shared fixtures."""

import numpy as np
from densekit.index import FlatIndex, IVFIndex, LSHIndex, PQIndex


def _recall(gold: np.ndarray, approx: np.ndarray) -> float:
    hits = sum(len(set(g) & set(a)) for g, a in zip(gold, approx, strict=True))
    return hits / gold.size


def _gold(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    flat = FlatIndex(dim=corpus.shape[1], metric="cosine")
    flat.add(corpus)
    return flat.search(queries, k=k).indices


def test_ivf_agrees_with_flat(sample_corpus, sample_queries) -> None:
    gold = _gold(sample_corpus, sample_queries, k=10)
    ivf = IVFIndex(dim=24, nlist=16, nprobe=8, metric="cosine", seed=0)
    ivf.train(sample_corpus)
    ivf.add(sample_corpus)
    assert _recall(gold, ivf.search(sample_queries, k=10).indices) > 0.75


def test_lsh_agrees_with_flat(sample_corpus, sample_queries) -> None:
    gold = _gold(sample_corpus, sample_queries, k=10)
    lsh = LSHIndex(dim=24, n_bits=12, n_tables=16, metric="cosine", seed=0, candidate_budget=300)
    lsh.add(sample_corpus)
    assert _recall(gold, lsh.search(sample_queries, k=10).indices) > 0.7


def test_pq_agrees_with_flat_on_l2(sample_corpus, sample_queries) -> None:
    flat = FlatIndex(dim=24, metric="l2")
    flat.add(sample_corpus)
    gold = flat.search(sample_queries, k=10).indices
    pq = PQIndex(dim=24, n_subvectors=6, n_bits=8, metric="l2", seed=0)
    pq.train(sample_corpus)
    pq.add(sample_corpus)
    assert _recall(gold, pq.search(sample_queries, k=10).indices) > 0.6
