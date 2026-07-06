# API reference

The stable, public surface of densekit. Import the common types straight from
the package root (`import densekit`).

## Encoders

### `HashingEncoder(dim=256, *, analyzer="word", ngram_range=(1, 1), normalize=True, lowercase=True)`
Deterministic feature-hashing encoder. `analyzer` is `"word"` or `"char"`.
- `encode(texts) -> np.ndarray` of shape `(len(texts), dim)`.

### `RandomProjectionEncoder(base, out_dim, *, seed=0, normalize=True)`
Projects another encoder's output to `out_dim` with a fixed Gaussian matrix.

## Indexes

All indexes share:
- `add(vectors)` — add `(n, dim)` rows.
- `search(queries, k=10) -> SearchResult`
- `size`, `dim`, `metric` properties.

### `FlatIndex(dim, metric="cosine")`
Exact brute-force search.

### `IVFIndex(dim, *, nlist=64, nprobe=8, metric="cosine", seed=0)`
Coarse-quantized approximate search. Call `train(vectors)` before `add`.

### `LSHIndex(dim, *, n_bits=16, n_tables=8, metric="cosine", seed=0, candidate_budget=None)`
Signed-random-projection LSH with multi-probe.

### `PQIndex(dim, *, n_subvectors=8, n_bits=8, metric="l2", seed=0)`
Product-quantized search (ADC). Call `train(vectors)` before `add`.

### `ProductQuantizer(dim, *, n_subvectors=8, n_bits=8, seed=0)`
The quantizer itself: `fit`, `encode`, `decode`, `asymmetric_distances`.

## `SearchResult`
Frozen dataclass returned by every `search`.
- `indices`, `scores` — `(n_queries, k)` arrays.
- `n_queries`, `k`
- `for_query(i) -> list[tuple[int, float]]`

## Evaluation

### `Qrels(judgements=None)`
Relevance judgements. `add(query_id, doc_id, grade=1.0)`, `relevant(query_id)`,
`graded(query_id)`, `query_ids`.

### `RetrievalEvaluator(qrels, *, k_values=(1, 5, 10))`
- `evaluate(run) -> dict[str, float]` where `run` maps query id to ranked doc ids.
  Returns `recall@k`, `precision@k`, `ndcg@k` for each `k`, plus `mrr` and `map`.

### Metric functions (`densekit.metrics`)
`recall_at_k`, `precision_at_k`, `ndcg_at_k`, `reciprocal_rank`,
`average_precision`, `hit_rate_at_k`, `mrr`, `mean_average_precision`.

## I/O

- `save_index(index, path)` — write a `.npz` archive.
- `load_index(path) -> BaseIndex` — reconstruct it.

## Losses

- `info_nce_loss(queries, positives, *, temperature=0.05, extra_negatives=None, normalize=True) -> float`

## Optional PyTorch backend (`densekit.torch_backend`)

- `BiEncoder(vocab_size=2**16, dim=128, *, pooling="mean", max_length=64)`
- `BiEncoderTrainer(model, config=None)` with `TrainConfig(epochs, lr, batch_size, temperature, seed)`
- `HashingTokenizer`, `mean_pool`, `cls_pool`
