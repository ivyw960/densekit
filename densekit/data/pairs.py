"""Training pairs for contrastive bi-encoder training.

A :class:`TrainingPair` bundles a query with a positive passage and any number
of explicit hard negatives; in-batch negatives are added by the trainer. The
:class:`PairDataset` iterates mini-batches and loads the same JSONL-per-line
format as the corpus.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from densekit.utils import make_rng


@dataclass(frozen=True)
class TrainingPair:
    """A ``(query, positive)`` pair with optional hard negatives."""

    query: str
    positive: str
    negatives: tuple[str, ...] = field(default_factory=tuple)


class PairDataset:
    """A shuffleable collection of :class:`TrainingPair`."""

    def __init__(self, pairs: Sequence[TrainingPair]) -> None:
        self._pairs = list(pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, i: int) -> TrainingPair:
        return self._pairs[i]

    def __iter__(self) -> Iterator[TrainingPair]:
        return iter(self._pairs)

    def batches(
        self, batch_size: int, *, shuffle: bool = True, seed: int | None = 0
    ) -> Iterator[list[TrainingPair]]:
        """Yield mini-batches, optionally shuffled deterministically."""
        if batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        order = list(range(len(self._pairs)))
        if shuffle:
            make_rng(seed).shuffle(order)
        for start in range(0, len(order), batch_size):
            yield [self._pairs[i] for i in order[start : start + batch_size]]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> PairDataset:
        pairs: list[TrainingPair] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pairs.append(
                    TrainingPair(
                        query=obj["query"],
                        positive=obj["positive"],
                        negatives=tuple(obj.get("negatives", ())),
                    )
                )
        return cls(pairs)
