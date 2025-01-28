# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-07

First public release.

### Added

- Offline encoders: `HashingEncoder` (word/char n-grams) and
  `RandomProjectionEncoder`.
- Indexes over a shared `search` API: exact `FlatIndex` and approximate
  `IVFIndex`, `LSHIndex`, `PQIndex` (plus the standalone `ProductQuantizer`).
- NumPy k-means and linear-algebra primitives (`cosine`, `dot`, `l2`, `top_k`).
- Ranking metrics: recall@k, precision@k, nDCG@k, MRR, MAP, hit-rate, and a
  `RetrievalEvaluator` over runs and qrels.
- `Corpus`/`Document` and `PairDataset`/`TrainingPair` JSONL loaders.
- Portable `.npz` index serialization via `save_index` / `load_index`.
- `densekit` CLI: `encode`, `build`, `search`, `evaluate`.
- Optional PyTorch backend: `BiEncoder` and an in-batch InfoNCE
  `BiEncoderTrainer`.
- NumPy reference `info_nce_loss`.

[Unreleased]: https://github.com/ivyw960/densekit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ivyw960/densekit/releases/tag/v0.1.0
