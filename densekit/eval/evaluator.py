"""Aggregate retrieval evaluation over a run and a set of qrels.

A *run* is a mapping from query id to a ranked list of document ids (best
first). The evaluator computes each requested metric per query and averages
across queries, mirroring the ``trec_eval`` convention where queries with no
judgements are skipped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from densekit.eval.qrels import Qrels
from densekit.metrics import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

Run: TypeAlias = Mapping[str, Sequence[str]]

_DEFAULT_KS = (1, 5, 10)


class RetrievalEvaluator:
    """Compute averaged ranking metrics for a run against qrels."""

    def __init__(self, qrels: Qrels, *, k_values: Sequence[int] = _DEFAULT_KS) -> None:
        if not k_values:
            msg = "k_values must not be empty"
            raise ValueError(msg)
        self.qrels = qrels
        self.k_values = tuple(sorted(int(k) for k in k_values))

    def evaluate(self, run: Run) -> dict[str, float]:
        """Return a flat ``{metric_name: mean_score}`` dictionary."""
        totals: dict[str, float] = {}
        n = 0
        for qid in self.qrels.query_ids:
            ranked = [str(d) for d in run.get(qid, [])]
            relevant = self.qrels.relevant(qid)
            if not relevant:
                continue
            n += 1
            graded = self.qrels.graded(qid)
            for k in self.k_values:
                totals[f"recall@{k}"] = totals.get(f"recall@{k}", 0.0) + recall_at_k(
                    ranked, relevant, k
                )
                totals[f"precision@{k}"] = totals.get(f"precision@{k}", 0.0) + precision_at_k(
                    ranked, relevant, k
                )
                totals[f"ndcg@{k}"] = totals.get(f"ndcg@{k}", 0.0) + ndcg_at_k(ranked, graded, k)
            totals["mrr"] = totals.get("mrr", 0.0) + reciprocal_rank(ranked, relevant)
            totals["map"] = totals.get("map", 0.0) + average_precision(ranked, relevant)

        if n == 0:
            return dict.fromkeys(self._metric_names(), 0.0)
        return {name: totals.get(name, 0.0) / n for name in self._metric_names()}

    def _metric_names(self) -> list[str]:
        names: list[str] = []
        for k in self.k_values:
            names.extend([f"recall@{k}", f"precision@{k}", f"ndcg@{k}"])
        names.extend(["mrr", "map"])
        return names
