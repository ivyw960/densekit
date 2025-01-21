"""Evaluate a tiny BEIR-style benchmark end to end with the NumPy core.

We define a small corpus, a few queries, and relevance judgements (qrels), then
encode, index, search, and score the run with the retrieval metrics.

Run with:  python examples/evaluate_beir_like.py
"""

from __future__ import annotations

from densekit import FlatIndex, HashingEncoder, Qrels, RetrievalEvaluator

CORPUS = {
    "d1": "product quantization compresses embeddings into compact codes",
    "d2": "inverted file indexes probe only the nearest centroid clusters",
    "d3": "locality sensitive hashing groups similar vectors into buckets",
    "d4": "bi-encoders map queries and documents into one vector space",
    "d5": "ndcg and mean average precision measure ranking quality",
    "d6": "cosine similarity compares the angle between two vectors",
}

QUERIES = {
    "q1": "how to compress embeddings into compact codes",
    "q2": "which metrics measure ranking quality like ndcg",
    "q3": "hashing similar vectors into buckets",
}

QRELS = Qrels(
    {
        "q1": {"d1": 1.0},
        "q2": {"d5": 1.0},
        "q3": {"d3": 1.0},
    }
)


def main() -> None:
    encoder = HashingEncoder(dim=1024, analyzer="char", ngram_range=(3, 5))

    doc_ids = list(CORPUS)
    index = FlatIndex(dim=1024, metric="cosine")
    index.add(encoder.encode([CORPUS[d] for d in doc_ids]))

    query_ids = list(QUERIES)
    result = index.search(encoder.encode([QUERIES[q] for q in query_ids]), k=len(doc_ids))
    run = {
        qid: [doc_ids[row] for row, _ in result.for_query(i)]
        for i, qid in enumerate(query_ids)
    }

    scores = RetrievalEvaluator(QRELS, k_values=[1, 3, 5]).evaluate(run)
    for name, value in scores.items():
        print(f"{name:>12}: {value:.3f}")


if __name__ == "__main__":
    main()
