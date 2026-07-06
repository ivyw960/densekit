# Design notes

Some of the less obvious decisions behind densekit, and the reasoning for them.

## Why a NumPy core?

The dense-retrieval ecosystem is powerful but heavy: FAISS, PyTorch,
sentence-transformers and a model download before you can rank a single query.
That is a poor fit for teaching, for CI, and for reproducing a paper's numbers
offline. densekit keeps the whole retrieval-and-evaluation loop in NumPy so it
installs in seconds and runs deterministically. PyTorch is strictly opt-in.

## Deterministic, offline encoders

The `HashingEncoder` uses the hashing trick with BLAKE2 so token → bucket is
stable across processes and Python versions (unlike the built-in `hash`). This
makes it a reliable test fixture and a sensible default when you have no trained
model. It is lexical, not semantic — the PyTorch bi-encoder is there when you
need learned representations.

## One search contract, four indexes

Every index implements `add` and `search` and returns the same `SearchResult`.
Ranking is always "higher is better", which means L2 distance is stored as its
negation internally. This keeps evaluation code index-agnostic and lets the CLI
swap backends by name.

- **IVF** partitions the space with k-means and scans only the `nprobe` nearest
  cells. Recall is tuned by `nprobe`.
- **LSH** uses signed random projections; multi-probe visits buckets in order of
  Hamming distance, gathered breadth-first across tables so no single table
  dominates the candidate set.
- **PQ** splits vectors into subspaces and quantizes each against its own
  codebook. Distances use the asymmetric (ADC) table lookup. The nearest-centroid
  assignment keeps the `‖centroid‖²` term — dropping it silently ranks by dot
  product instead of distance (an early bug worth calling out).

## Metrics are id-agnostic

Ranking metrics are generic over the id type, so the same functions work with
string corpus ids and with the integer row indices an index returns. They rely
only on hashing and equality.

## Portable persistence

Indexes serialize to `.npz` with a small JSON metadata blob and no pickling.
Archives are inspectable, diffable in spirit, and safe to load with
`allow_pickle=False`.

## Known limitations

- The hashing encoder is lexical; expect modest recall on paraphrase-heavy data.
- Approximate indexes are pure NumPy and single-threaded — fine for research and
  mid-sized corpora, not a production ANN service.
- PQ currently supports `l2` and `cosine` (via normalization), not raw `dot`.
