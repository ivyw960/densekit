"""Ranking-quality metrics for retrieval evaluation.

Each function scores a single ranked list of retrieved document ids against a
set (or graded mapping) of relevance judgements. Aggregation helpers average a
metric over many queries. Ids may be Python ints or NumPy integers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def _as_set(relevant: Iterable[int]) -> set[int]:
    return {int(r) for r in relevant}


def recall_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of the relevant documents that appear in the top ``k``."""
    rel = _as_set(relevant)
    if not rel:
        return 0.0
    hits = sum(1 for d in ranked[:k] if int(d) in rel)
    return hits / len(rel)


def precision_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant."""
    if k <= 0:
        msg = "k must be positive"
        raise ValueError(msg)
    rel = _as_set(relevant)
    hits = sum(1 for d in ranked[:k] if int(d) in rel)
    return hits / k
