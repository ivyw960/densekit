"""Save and load indexes as portable ``.npz`` archives.

Everything is stored with NumPy's native ``savez`` plus a small JSON metadata
blob, so an index round-trips without pickling and stays inspectable and
offline. Supported index types: Flat, IVF, LSH and PQ.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from densekit.index import BaseIndex, FlatIndex, IVFIndex, LSHIndex, PQIndex


def save_index(index: BaseIndex, path: str | Path) -> None:
    """Serialize ``index`` to ``path`` (a ``.npz`` archive)."""
    path = Path(path)
    kind = type(index).__name__
    meta: dict[str, object] = {"kind": kind, "dim": index.dim, "metric": index.metric}

    if isinstance(index, FlatIndex):
        np.savez(path, meta=json.dumps(meta), vectors=index._vectors)
    elif isinstance(index, IVFIndex):
        if index._centroids is None:
            msg = "cannot save an untrained IVF index"
            raise RuntimeError(msg)
        meta.update(nlist=index.nlist, nprobe=index.nprobe, seed=index.seed)
        np.savez(
            path,
            meta=json.dumps(meta),
            vectors=index._vectors,
            centroids=index._centroids,
            cell_of=np.asarray(index._cell_of, dtype=np.int64),
        )
    elif isinstance(index, LSHIndex):
        meta.update(n_bits=index.n_bits, n_tables=index.n_tables, budget=index.candidate_budget)
        np.savez(
            path, meta=json.dumps(meta), vectors=index._vectors, planes=np.stack(index._planes)
        )
    elif isinstance(index, PQIndex):
        if index.pq._codebooks is None:
            msg = "cannot save an unfitted PQ index"
            raise RuntimeError(msg)
        meta.update(n_subvectors=index.pq.m, n_bits=int(index.pq.ksub).bit_length() - 1)
        np.savez(path, meta=json.dumps(meta), codebooks=index.pq._codebooks, codes=index._codes)
    else:  # pragma: no cover - defensive
        msg = f"cannot serialize index of type {kind}"
        raise TypeError(msg)


def load_index(path: str | Path) -> BaseIndex:
    """Reconstruct an index previously written by :func:`save_index`."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        kind = meta["kind"]
        if kind == "FlatIndex":
            flat = FlatIndex(meta["dim"], meta["metric"])
            flat._vectors = data["vectors"].astype(np.float32)
            return flat
        if kind == "IVFIndex":
            ivf = IVFIndex(
                meta["dim"], nlist=meta["nlist"], nprobe=meta["nprobe"], metric=meta["metric"]
            )
            centroids = data["centroids"].astype(np.float32)
            ivf._centroids = centroids
            ivf.nlist = centroids.shape[0]
            ivf._vectors = data["vectors"].astype(np.float32)
            ivf._cell_of = [int(c) for c in data["cell_of"]]
            ivf._trained = True
            return ivf
        if kind == "LSHIndex":
            lsh = LSHIndex(
                meta["dim"],
                n_bits=meta["n_bits"],
                n_tables=meta["n_tables"],
                metric=meta["metric"],
                candidate_budget=meta["budget"],
            )
            lsh._planes = [p.astype(np.float32) for p in data["planes"]]
            vectors = data["vectors"].astype(np.float32)
            lsh._vectors = vectors
            lsh._tables = [{} for _ in range(lsh.n_tables)]
            for t in range(lsh.n_tables):
                for offset, code in enumerate(lsh._codes(vectors, t)):
                    lsh._tables[t].setdefault(int(code), []).append(offset)
            return lsh
        if kind == "PQIndex":
            pq = PQIndex(
                meta["dim"],
                n_subvectors=meta["n_subvectors"],
                n_bits=meta["n_bits"],
                metric=meta["metric"],
            )
            pq.pq._codebooks = data["codebooks"].astype(np.float32)
            pq._codes = data["codes"].astype(np.uint8)
            return pq
    msg = f"unknown index kind: {kind}"
    raise ValueError(msg)
