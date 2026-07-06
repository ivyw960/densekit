import numpy as np
from densekit.losses import info_nce_loss, log_softmax


def test_aligned_pairs_have_low_loss() -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal((16, 32)).astype(np.float32)
    # positives are the queries plus a tiny perturbation -> nearly identical
    aligned = base + 0.01 * rng.standard_normal((16, 32)).astype(np.float32)
    shuffled = base[rng.permutation(16)]

    good = info_nce_loss(base, aligned, temperature=0.05)
    bad = info_nce_loss(base, shuffled, temperature=0.05)
    assert good < bad


def test_loss_is_non_negative() -> None:
    rng = np.random.default_rng(1)
    q = rng.standard_normal((8, 16)).astype(np.float32)
    p = rng.standard_normal((8, 16)).astype(np.float32)
    assert info_nce_loss(q, p) >= 0.0


def test_temperature_must_be_positive() -> None:
    import pytest

    with pytest.raises(ValueError, match="temperature"):
        info_nce_loss(np.zeros((2, 4)), np.zeros((2, 4)), temperature=0.0)


def test_log_softmax_sums_to_one() -> None:
    x = np.array([[1.0, 2.0, 3.0]])
    probs = np.exp(log_softmax(x))
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)
