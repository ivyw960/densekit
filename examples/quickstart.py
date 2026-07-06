"""End-to-end quickstart: encode a tiny corpus, index it, and search.

Run with:  python examples/quickstart.py
Uses only the NumPy core -- no PyTorch, no downloads.
"""

from __future__ import annotations

import densekit

CORPUS = [
    "dense retrieval maps queries and documents into a shared vector space",
    "an inverted file index probes only the nearest clusters",
    "locality sensitive hashing buckets similar vectors together",
    "product quantization stores compact codes instead of full vectors",
    "normalized discounted cumulative gain rewards highly ranked hits",
]

QUERIES = [
    "how do we compress embeddings to save memory?",
    "what metric accounts for the rank of relevant results?",
]


def main() -> None:
    encoder = densekit.HashingEncoder(dim=512, analyzer="char", ngram_range=(3, 5))
    index = densekit.FlatIndex(dim=512, metric="cosine")
    index.add(encoder.encode(CORPUS))

    results = index.search(encoder.encode(QUERIES), k=2)
    for q, query in enumerate(QUERIES):
        print(f"\nQ: {query}")
        for doc_id, score in results.for_query(q):
            print(f"  {score:+.3f}  {CORPUS[doc_id]}")


if __name__ == "__main__":
    main()
