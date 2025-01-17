# Architecture

densekit is organized as a small set of independent layers. Each layer depends
only on the ones beneath it, and the whole core rests on NumPy alone.

```
                +--------------------+
                |        cli         |  encode / build / search / evaluate
                +--------------------+
                          |
   +-----------+   +-------------+   +------------+   +-----------+
   | encoders  |   |   index     |   |    eval    |   |   data    |
   | hashing   |   | flat / ivf  |   | metrics /  |   | corpus /  |
   | projection|   | lsh / pq    |   | evaluator  |   | pairs     |
   +-----------+   +-------------+   +------------+   +-----------+
          \             |                 |               /
           +------------+-----------------+--------------+
                          |
                +--------------------+
                | linalg | clustering|  NumPy primitives
                +--------------------+

     (optional)  torch_backend: bi-encoder + trainer  ->  PyTorch
```

## Layers

- **linalg / clustering** — normalization, similarity/distance matrices, a
  shared `top_k`, and a dependency-free k-means. Everything numeric bottoms out
  here.
- **encoders** — turn text into vectors. All bundled encoders are deterministic
  and offline so tests and demos are reproducible.
- **index** — store vectors and answer top-k queries. A single
  `BaseIndex.search` contract is implemented by an exact backend (`FlatIndex`)
  and three approximate ones (`IVFIndex`, `LSHIndex`, `PQIndex`).
- **eval** — ranking metrics plus a `RetrievalEvaluator` that averages them over
  a run and a set of qrels.
- **data** — `Corpus`/`Document` and `PairDataset`/`TrainingPair`, both backed by
  a line-oriented JSONL format.
- **io** — serialize any index to a portable `.npz` archive.
- **cli** — a thin argparse front-end that wires the layers into file-based
  pipelines.
- **torch_backend** *(optional)* — a self-contained PyTorch bi-encoder and an
  in-batch InfoNCE trainer. The core never imports it.

## Data flow

A typical retrieval run: text → `Encoder.encode` → embedding matrix →
`Index.add` / `Index.search` → `SearchResult` (ids + scores) → mapped back to
document ids → `RetrievalEvaluator` against qrels.

## Design invariants

- Embeddings are 2D `float32` arrays, one row per item.
- Search always ranks "higher is better"; L2 distance is negated internally.
- Randomness flows through an explicit seed, never global state.
