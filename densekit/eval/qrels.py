"""Relevance judgements ("qrels").

A :class:`Qrels` maps each query id to a set of relevant document ids and,
optionally, graded relevance levels. It is the ground truth an
:class:`~densekit.eval.evaluator.RetrievalEvaluator` scores runs against.
"""

from __future__ import annotations

from collections.abc import Mapping


class Qrels:
    """Container for per-query relevance judgements."""

    def __init__(self, judgements: Mapping[str, Mapping[str, float]] | None = None) -> None:
        self._data: dict[str, dict[str, float]] = {}
        if judgements:
            for qid, docs in judgements.items():
                for did, grade in docs.items():
                    self.add(qid, did, grade)

    def add(self, query_id: str, doc_id: str, grade: float = 1.0) -> None:
        """Record that ``doc_id`` is relevant to ``query_id`` with ``grade``."""
        if grade < 0:
            msg = "relevance grade must be non-negative"
            raise ValueError(msg)
        self._data.setdefault(str(query_id), {})[str(doc_id)] = float(grade)

    def relevant(self, query_id: str) -> set[str]:
        """Ids of documents with a positive grade for ``query_id``."""
        return {d for d, g in self._data.get(str(query_id), {}).items() if g > 0}

    def graded(self, query_id: str) -> dict[str, float]:
        """Full ``{doc_id: grade}`` mapping for ``query_id``."""
        return dict(self._data.get(str(query_id), {}))

    @property
    def query_ids(self) -> list[str]:
        return list(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, query_id: str) -> bool:
        return str(query_id) in self._data
