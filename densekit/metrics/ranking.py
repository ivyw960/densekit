"""Ranking-quality metrics for retrieval evaluation.

Each function scores a single ranked list of retrieved document ids against a
set (or graded mapping) of relevance judgements. Aggregation helpers average a
metric over many queries. Document ids may be strings or (NumPy) integers -- the
metrics only rely on equality and hashing, never on the id being numeric, so the
id type is a single generic ``Id`` shared by the ranking and the judgements.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TypeVar

# A document id is anything hashable; in practice a corpus id string or an
# integer row index returned by an index.
Id = TypeVar("Id", bound=Hashable)


def _as_set(relevant: Iterable[Id]) -> set[Id]:
    return set(relevant)


def recall_at_k(ranked: Sequence[Id], relevant: Iterable[Id], k: int) -> float:
    """Fraction of the relevant documents that appear in the top ``k``."""
    rel = _as_set(relevant)
    if not rel:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in rel)
    return hits / len(rel)


def precision_at_k(ranked: Sequence[Id], relevant: Iterable[Id], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant."""
    if k <= 0:
        msg = "k must be positive"
        raise ValueError(msg)
    rel = _as_set(relevant)
    hits = sum(1 for d in ranked[:k] if d in rel)
    return hits / k


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[Id], relevance: Mapping[Id, float], k: int) -> float:
    """Normalized discounted cumulative gain at ``k`` with graded relevance.

    ``relevance`` maps document id to a non-negative gain; documents missing
    from the mapping contribute zero. The score is the achieved DCG divided by
    the DCG of the ideal ranking, so it lies in ``[0, 1]``.
    """
    gains = [float(relevance.get(d, 0.0)) for d in ranked[:k]]
    ideal = sorted((float(g) for g in relevance.values()), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def reciprocal_rank(ranked: Sequence[Id], relevant: Iterable[Id]) -> float:
    """Reciprocal of the rank of the first relevant document (0 if none)."""
    rel = _as_set(relevant)
    for i, d in enumerate(ranked):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(ranked: Sequence[Id], relevant: Iterable[Id], k: int | None = None) -> float:
    """Average precision over the (optionally truncated) ranking."""
    rel = _as_set(relevant)
    if not rel:
        return 0.0
    ordered = ranked[:k] if k is not None else ranked
    hits = 0
    score = 0.0
    for i, d in enumerate(ordered):
        if d in rel:
            hits += 1
            score += hits / (i + 1)
    denom = min(len(rel), k) if k is not None else len(rel)
    return score / denom


def hit_rate_at_k(ranked: Sequence[Id], relevant: Iterable[Id], k: int) -> float:
    """1.0 if any relevant document is in the top ``k``, else 0.0."""
    rel = _as_set(relevant)
    return 1.0 if any(d in rel for d in ranked[:k]) else 0.0


def mrr(rankings: Sequence[Sequence[Id]], relevants: Sequence[Iterable[Id]]) -> float:
    """Mean reciprocal rank across a batch of queries."""
    if not rankings:
        return 0.0
    scores = [reciprocal_rank(r, rel) for r, rel in zip(rankings, relevants, strict=True)]
    return sum(scores) / len(scores)


def mean_average_precision(
    rankings: Sequence[Sequence[Id]],
    relevants: Sequence[Iterable[Id]],
    k: int | None = None,
) -> float:
    """Mean average precision (MAP) across a batch of queries."""
    if not rankings:
        return 0.0
    scores = [average_precision(r, rel, k) for r, rel in zip(rankings, relevants, strict=True)]
    return sum(scores) / len(scores)
