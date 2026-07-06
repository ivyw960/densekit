import numpy as np
from densekit.encoders import HashingEncoder


def test_encode_shape() -> None:
    enc = HashingEncoder(dim=64)
    emb = enc.encode(["hello world", "dense retrieval toolkit"])
    assert emb.shape == (2, 64)
    assert emb.dtype == np.float32


def test_encoding_is_deterministic() -> None:
    enc = HashingEncoder(dim=128)
    a = enc.encode(["the quick brown fox"])
    b = enc.encode(["the quick brown fox"])
    assert np.array_equal(a, b)


def test_normalized_rows_are_unit_norm() -> None:
    enc = HashingEncoder(dim=64, normalize=True)
    emb = enc.encode(["some non empty text here"])
    assert np.isclose(np.linalg.norm(emb[0]), 1.0, atol=1e-5)


def test_empty_text_maps_to_zero_vector() -> None:
    enc = HashingEncoder(dim=32)
    emb = enc.encode([""])
    assert np.allclose(emb, 0.0)


def test_rejects_non_positive_dim() -> None:
    import pytest

    with pytest.raises(ValueError, match="dim must be positive"):
        HashingEncoder(dim=0)
