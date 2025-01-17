# densekit

[![CI](https://github.com/ivyw960/densekit/actions/workflows/ci.yml/badge.svg)](https://github.com/ivyw960/densekit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An offline-first dense retrieval and embedding toolkit for RAG.

densekit gives you the pieces you need to build and evaluate a dense retriever
without pulling in a heavy stack: a pure-NumPy core for encoding, indexing and
evaluation, plus an *optional* PyTorch backend for training bi-encoders. There
are no model downloads and no network calls in the core, so results are
reproducible on a laptop or in CI.

## Features

- **Encoders** — deterministic, offline `HashingEncoder` (word/char n-grams) and
  a `RandomProjectionEncoder` for dimensionality reduction.
- **Indexes** — exact `FlatIndex` plus approximate `IVFIndex`, `LSHIndex` and
  `PQIndex`, all in NumPy, all sharing one `search` API.
- **Evaluation** — recall@k, precision@k, nDCG@k, MRR and MAP with a
  `RetrievalEvaluator` that mirrors `trec_eval` conventions.
- **Training** *(optional)* — a small PyTorch bi-encoder trained with in-batch
  InfoNCE via `densekit[torch]`.
- **Portable I/O** — save and load any index as a plain `.npz` archive.
- **CLI** — `densekit encode | build | search | evaluate` for file-based
  pipelines.

## Install

```bash
pip install densekit            # NumPy core only
pip install "densekit[torch]"   # add the PyTorch bi-encoder trainer
```

Requires Python 3.10+.

## Quickstart

```python
import densekit

corpus = [
    "dense retrieval encodes text into vectors",
    "approximate nearest neighbour search is fast",
    "product quantization compresses embeddings",
]
queries = ["how does ANN search work?"]

encoder = densekit.HashingEncoder(dim=256, analyzer="char", ngram_range=(3, 5))
index = densekit.FlatIndex(dim=256, metric="cosine")
index.add(encoder.encode(corpus))

hits = index.search(encoder.encode(queries), k=2)
print(hits.for_query(0))   # [(doc_id, score), ...]
```

## Choosing an index

| Index | Kind | Good when | Trade-off |
| --- | --- | --- | --- |
| `FlatIndex` | exact | up to ~1e5 vectors | linear scan |
| `IVFIndex` | coarse quantizer | large corpora | recall vs `nprobe` |
| `LSHIndex` | angular hashing | high-dim, cosine | recall vs tables/bits |
| `PQIndex` | product quantization | memory-constrained | approximation error |

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — how the pieces fit together
- [`docs/usage.md`](docs/usage.md) — task-oriented guide
- [`docs/api-reference.md`](docs/api-reference.md) — public API
- [`docs/design-notes.md`](docs/design-notes.md) — why it works the way it does
- [`examples/`](examples/) — runnable scripts

## License

MIT — see [LICENSE](LICENSE).
