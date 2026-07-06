# densekit

An offline-first dense retrieval and embedding toolkit for RAG.

densekit gives you the pieces you need to build and evaluate a dense retriever
without pulling in a heavy stack: a pure-NumPy core for encoding, indexing and
evaluation, plus an *optional* PyTorch backend for training bi-encoders. There
are no model downloads and no network calls in the core, so results are
reproducible on a laptop or in CI.

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

Swap `FlatIndex` for `IVFIndex`, `LSHIndex` or `PQIndex` to trade a little recall
for speed and memory.

## Status

Early but usable. The NumPy core is covered by tests; the PyTorch backend is
optional and exercised separately.
