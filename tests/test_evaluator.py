import math

from densekit.eval import Qrels, RetrievalEvaluator


def test_perfect_run_scores_one() -> None:
    qrels = Qrels({"q1": {"d1": 1.0}, "q2": {"d2": 1.0}})
    run = {"q1": ["d1", "d9"], "q2": ["d2", "d8"]}
    scores = RetrievalEvaluator(qrels, k_values=[1]).evaluate(run)
    assert math.isclose(scores["recall@1"], 1.0)
    assert math.isclose(scores["mrr"], 1.0)
    assert math.isclose(scores["map"], 1.0)


def test_queries_without_judgements_are_skipped() -> None:
    qrels = Qrels({"q1": {"d1": 1.0}})
    run = {"q1": ["d5", "d1"], "q2": ["d3"]}
    scores = RetrievalEvaluator(qrels, k_values=[2]).evaluate(run)
    # only q1 counts; d1 is at rank 2 -> reciprocal rank 0.5
    assert math.isclose(scores["mrr"], 0.5)


def test_metric_names_cover_all_k() -> None:
    qrels = Qrels({"q1": {"d1": 1.0}})
    scores = RetrievalEvaluator(qrels, k_values=[1, 5]).evaluate({"q1": ["d1"]})
    for name in ("recall@1", "precision@5", "ndcg@1", "ndcg@5", "mrr", "map"):
        assert name in scores


def test_empty_kvalues_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="k_values"):
        RetrievalEvaluator(Qrels(), k_values=[])
