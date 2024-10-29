"""Contrastive training loop for the PyTorch bi-encoder.

Uses in-batch InfoNCE: within a mini-batch, each query's paired positive is the
target and every other positive is a negative. This mirrors the NumPy
``info_nce_loss`` reference but runs under autograd so the encoder actually
learns.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from densekit.data.pairs import PairDataset, TrainingPair
from densekit.torch_backend.bi_encoder import BiEncoder


@dataclass
class TrainConfig:
    """Hyper-parameters for :class:`BiEncoderTrainer`."""

    epochs: int = 3
    lr: float = 1e-3
    batch_size: int = 16
    temperature: float = 0.05
    seed: int = 0


class BiEncoderTrainer:
    """Fine-tune a :class:`BiEncoder` with in-batch InfoNCE."""

    def __init__(self, model: BiEncoder, config: TrainConfig | None = None) -> None:
        self.model = model
        self.config = config or TrainConfig()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)

    def _loss(self, queries: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        logits = queries @ positives.t() / self.config.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return torch.nn.functional.cross_entropy(logits, labels)

    def train_step(self, batch: list[TrainingPair]) -> float:
        self.model.train()
        q = self.model.embed([pair.query for pair in batch])
        p = self.model.embed([pair.positive for pair in batch])
        loss = self._loss(q, p)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def fit(self, dataset: PairDataset) -> list[float]:
        """Train for ``config.epochs`` and return the mean loss per epoch."""
        torch.manual_seed(self.config.seed)
        history: list[float] = []
        for epoch in range(self.config.epochs):
            losses = [
                self.train_step(batch)
                for batch in dataset.batches(
                    self.config.batch_size, shuffle=True, seed=self.config.seed + epoch
                )
                if len(batch) > 1  # in-batch negatives need at least two rows
            ]
            history.append(sum(losses) / len(losses) if losses else 0.0)
        return history
