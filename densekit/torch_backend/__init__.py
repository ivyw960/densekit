"""Optional PyTorch backend for training bi-encoders.

Importing this package requires PyTorch (``pip install densekit[torch]``); the
NumPy core of densekit never depends on it.
"""

from __future__ import annotations

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without torch
    msg = "densekit.torch_backend requires PyTorch. Install with 'pip install densekit[torch]'."
    raise ImportError(msg) from exc

from densekit.torch_backend.bi_encoder import BiEncoder, HashingTokenizer
from densekit.torch_backend.pooling import cls_pool, mean_pool

__all__ = ["BiEncoder", "HashingTokenizer", "cls_pool", "mean_pool"]
