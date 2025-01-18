# Usage

A task-oriented tour of densekit. All snippets use the NumPy core unless noted.

## Encoding text

```python
from densekit import HashingEncoder, RandomProjectionEncoder

# word-level hashing
enc = HashingEncoder(dim=512)

# character n-grams are robust to typos and work across languages
char_enc = HashingEncoder(dim=512, analyzer="char", ngram_range=(3, 5))

# shrink a wide encoder for a more index-friendly vector
compact = RandomProjectionEncoder(base=char_enc, out_dim=128, seed=0)

embeddings = compact.encode(["some documents", "go here"])
```

## Building and searching an index

```python
from densekit import FlatIndex, IVFIndex

index = IVFIndex(dim=128, nlist=256, nprobe=16, metric="cosine")
index.train(embeddings)   # IVF and PQ need a training pass
index.add(embeddings)

result = index.search(query_embeddings, k=10)
for qid in range(result.n_queries):
    print(result.for_query(qid))
```

`FlatIndex` and `LSHIndex` skip `train()`. Metrics are `"cosine"`, `"dot"` or
`"l2"`.

## Saving and loading

```python
from densekit import save_index, load_index

save_index(index, "corpus.npz")
index = load_index("corpus.npz")
```

## Evaluating retrieval quality

```python
from densekit import Qrels, RetrievalEvaluator

qrels = Qrels({"q1": {"d3": 1.0, "d7": 1.0}})
run = {"q1": ["d7", "d1", "d3"]}   # ranked doc ids per query

scores = RetrievalEvaluator(qrels, k_values=[1, 5, 10]).evaluate(run)
print(scores["ndcg@10"], scores["mrr"], scores["map"])
```

## Command line

```bash
densekit encode   --corpus corpus.jsonl --out emb.npy --ids ids.json --dim 256
densekit build    --embeddings emb.npy  --out index.npz --index ivf --nlist 256
densekit search   --index index.npz     --queries q.npy --k 10
densekit evaluate --run run.jsonl        --qrels qrels.jsonl --k 1 5 10
```

## Training a bi-encoder (optional)

```python
from densekit.data import PairDataset, TrainingPair
from densekit.torch_backend import BiEncoder, BiEncoderTrainer, TrainConfig

dataset = PairDataset([TrainingPair("a query", "a relevant passage")])
model = BiEncoder(dim=128)
history = BiEncoderTrainer(model, TrainConfig(epochs=3)).fit(dataset)
```
