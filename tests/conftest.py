import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def sample_corpus(rng: np.random.Generator) -> np.ndarray:
    """A small, well-separated corpus so approximate indexes recover neighbours."""
    return rng.standard_normal((400, 24)).astype(np.float32)


@pytest.fixture
def sample_queries(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((12, 24)).astype(np.float32)
