"""Contrastive losses.

The NumPy ``info_nce_loss`` here is the reference implementation used in tests
and for quick, dependency-free sanity checks. The optional PyTorch backend in
:mod:`densekit.torch_backend` mirrors it with autograd for actual training.
"""

from __future__ import annotations

import numpy as np

from densekit.linalg import as_float2d, l2_normalize
from densekit.types import FloatArray


def info_nce_loss(
    queries: np.ndarray,
    positives: np.ndarray,
    *,
    temperature: float = 0.05,
    extra_negatives: np.ndarray | None = None,
    normalize: bool = True,
) -> float:
    """In-batch InfoNCE loss for aligned ``(query, positive)`` rows.

    Row ``i`` of ``queries`` is paired with row ``i`` of ``positives``; every
    other positive in the batch (plus any ``extra_negatives``) acts as a
    negative. Returns the mean cross-entropy over the batch.
    """
    if temperature <= 0:
        msg = "temperature must be positive"
        raise ValueError(msg)
    q = as_float2d(queries)
    p = as_float2d(positives)
    if q.shape != p.shape:
        msg = f"queries and positives must match, got {q.shape} and {p.shape}"
        raise ValueError(msg)
    if normalize:
        q = l2_normalize(q)
        p = l2_normalize(p)

    logits = q @ p.T / temperature
    if extra_negatives is not None:
        neg = (
            l2_normalize(as_float2d(extra_negatives)) if normalize else as_float2d(extra_negatives)
        )
        logits = np.concatenate([logits, q @ neg.T / temperature], axis=1)

    n = q.shape[0]
    labels = np.arange(n)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1))
    log_prob = shifted[labels, labels] - log_sum_exp
    return float(-np.mean(log_prob))


def log_softmax(scores: np.ndarray, axis: int = -1) -> FloatArray:
    """Numerically stable log-softmax along ``axis``."""
    x = np.asarray(scores, dtype=np.float32)
    shifted = x - x.max(axis=axis, keepdims=True)
    return np.asarray(
        shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True)), dtype=np.float32
    )
