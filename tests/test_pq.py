import numpy as np
from densekit.index import ProductQuantizer


def _corpus(n: int = 512, dim: int = 16, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def test_encode_produces_one_code_per_subvector() -> None:
    pq = ProductQuantizer(dim=16, n_subvectors=4, n_bits=4, seed=0)
    data = _corpus()
    pq.fit(data)
    codes = pq.encode(data)
    assert codes.shape == (512, 4)
    assert codes.dtype == np.uint8
    assert codes.max() < 16


def test_reconstruction_error_below_identity() -> None:
    data = _corpus()
    pq = ProductQuantizer(dim=16, n_subvectors=8, n_bits=6, seed=1).fit(data)
    recon = pq.decode(pq.encode(data))
    err = np.mean(np.linalg.norm(data - recon, axis=1))
    baseline = np.mean(np.linalg.norm(data, axis=1))
    assert err < baseline


def test_rejects_indivisible_dim() -> None:
    import pytest

    with pytest.raises(ValueError, match="divisible"):
        ProductQuantizer(dim=15, n_subvectors=4)


def test_encode_before_fit_raises() -> None:
    import pytest

    pq = ProductQuantizer(dim=8, n_subvectors=2)
    with pytest.raises(RuntimeError, match="fit"):
        pq.encode(_corpus(4, 8))
