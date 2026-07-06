from pathlib import Path

import numpy as np
from densekit.index import FlatIndex, IVFIndex, LSHIndex, PQIndex
from densekit.io import load_index, save_index


def _corpus(n: int = 300, dim: int = 16, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def _assert_same_results(a, b, queries) -> None:
    ra = a.search(queries, k=5).indices
    rb = b.search(queries, k=5).indices
    assert np.array_equal(ra, rb)


def test_flat_round_trip(tmp_path: Path) -> None:
    corpus, queries = _corpus(), _corpus(5, 16, seed=1)
    idx = FlatIndex(dim=16, metric="cosine")
    idx.add(corpus)
    save_index(idx, tmp_path / "flat.npz")
    _assert_same_results(idx, load_index(tmp_path / "flat.npz"), queries)


def test_ivf_round_trip(tmp_path: Path) -> None:
    corpus, queries = _corpus(), _corpus(5, 16, seed=2)
    idx = IVFIndex(dim=16, nlist=16, nprobe=4, metric="cosine", seed=0)
    idx.train(corpus)
    idx.add(corpus)
    save_index(idx, tmp_path / "ivf.npz")
    _assert_same_results(idx, load_index(tmp_path / "ivf.npz"), queries)


def test_lsh_round_trip(tmp_path: Path) -> None:
    corpus, queries = _corpus(), _corpus(5, 16, seed=3)
    idx = LSHIndex(dim=16, n_bits=10, n_tables=8, metric="cosine", seed=0)
    idx.add(corpus)
    save_index(idx, tmp_path / "lsh.npz")
    _assert_same_results(idx, load_index(tmp_path / "lsh.npz"), queries)


def test_pq_round_trip(tmp_path: Path) -> None:
    corpus, queries = _corpus(), _corpus(5, 16, seed=4)
    idx = PQIndex(dim=16, n_subvectors=4, n_bits=6, metric="l2", seed=0)
    idx.train(corpus)
    idx.add(corpus)
    save_index(idx, tmp_path / "pq.npz")
    _assert_same_results(idx, load_index(tmp_path / "pq.npz"), queries)
