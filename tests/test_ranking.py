import math

from densekit.metrics import (
    average_precision,
    hit_rate_at_k,
    mean_average_precision,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_and_precision() -> None:
    ranked = [1, 2, 3, 4, 5]
    relevant = {2, 4, 9}
    assert recall_at_k(ranked, relevant, 5) == 2 / 3
    assert precision_at_k(ranked, relevant, 5) == 2 / 5


def test_recall_empty_relevant_is_zero() -> None:
    assert recall_at_k([1, 2, 3], [], 3) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank([9, 8, 2, 1], {2}) == 1 / 3
    assert reciprocal_rank([9, 8, 7], {2}) == 0.0


def test_ndcg_perfect_ranking_is_one() -> None:
    relevance = {1: 3.0, 2: 2.0, 3: 1.0}
    assert math.isclose(ndcg_at_k([1, 2, 3], relevance, 3), 1.0)


def test_ndcg_matches_manual_value() -> None:
    # gains [3, 2] at ranks 1,2 -> dcg = 3/log2(2) + 2/log2(3)
    relevance = {10: 3.0, 20: 2.0}
    dcg = 3 / math.log2(2) + 2 / math.log2(3)
    idcg = 3 / math.log2(2) + 2 / math.log2(3)
    assert math.isclose(ndcg_at_k([10, 20], relevance, 2), dcg / idcg)


def test_average_precision_known() -> None:
    # relevant at ranks 1 and 3 -> (1/1 + 2/3) / 2
    ap = average_precision([1, 5, 2, 6], {1, 2}, k=4)
    assert math.isclose(ap, (1.0 + 2 / 3) / 2)


def test_hit_rate() -> None:
    assert hit_rate_at_k([5, 6, 2], {2}, 3) == 1.0
    assert hit_rate_at_k([5, 6, 7], {2}, 3) == 0.0


def test_batch_aggregations() -> None:
    rankings = [[2, 1], [3, 4]]
    relevants = [{2}, {4}]
    assert math.isclose(mrr(rankings, relevants), (1.0 + 0.5) / 2)
    assert math.isclose(mean_average_precision(rankings, relevants), (1.0 + 0.5) / 2)
