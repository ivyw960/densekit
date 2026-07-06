"""Deterministic seeding helpers."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy random generator for reproducible experiments.

    Passing ``None`` yields a fresh, non-deterministic generator; passing an
    integer makes every downstream draw reproducible.
    """
    return np.random.default_rng(seed)
