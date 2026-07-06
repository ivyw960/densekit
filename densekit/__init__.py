"""densekit: an offline-first dense retrieval and embedding toolkit.

The most-used building blocks are re-exported here for convenience::

    import densekit

    encoder = densekit.HashingEncoder(dim=256)
    index = densekit.FlatIndex(dim=256)
    index.add(encoder.encode(corpus))
"""

from densekit.__about__ import __version__
from densekit.encoders import Encoder, HashingEncoder, RandomProjectionEncoder
from densekit.eval import Qrels, RetrievalEvaluator
from densekit.index import FlatIndex, IVFIndex, LSHIndex, PQIndex, ProductQuantizer
from densekit.io import load_index, save_index
from densekit.losses import info_nce_loss

__all__ = [
    "Encoder",
    "FlatIndex",
    "HashingEncoder",
    "IVFIndex",
    "LSHIndex",
    "PQIndex",
    "ProductQuantizer",
    "Qrels",
    "RandomProjectionEncoder",
    "RetrievalEvaluator",
    "__version__",
    "info_nce_loss",
    "load_index",
    "save_index",
]
