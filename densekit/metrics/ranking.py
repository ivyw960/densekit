"""Ranking-quality metrics for retrieval evaluation.

Each function scores a single ranked list of retrieved document ids against a
set (or graded mapping) of relevance judgements. Aggregation helpers average a
metric over many queries. Ids may be Python ints or NumPy integers.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


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


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[int], relevance: Mapping[int, float], k: int) -> float:
    """Normalized discounted cumulative gain at ``k`` with graded relevance.

    ``relevance`` maps document id to a non-negative gain; documents missing
    from the mapping contribute zero. The score is the achieved DCG divided by
    the DCG of the ideal ranking, so it lies in ``[0, 1]``.
    """
    graded = {int(d): float(g) for d, g in relevance.items()}
    gains = [graded.get(int(d), 0.0) for d in ranked[:k]]
    ideal = sorted(graded.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0
