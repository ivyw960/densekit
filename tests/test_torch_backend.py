import numpy as np
import pytest

pytest.importorskip("torch")


def test_encode_shape_and_norm() -> None:
    from densekit.torch_backend import BiEncoder

    model = BiEncoder(vocab_size=4096, dim=32)
    emb = model.encode(["dense retrieval", "vector search"])
    assert emb.shape == (2, 32)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_cls_and_mean_pooling_run() -> None:
    from densekit.torch_backend import BiEncoder

    for pooling in ("mean", "cls"):
        model = BiEncoder(vocab_size=1024, dim=16, pooling=pooling)
        emb = model.encode(["a short query", "another passage here"])
        assert emb.shape == (2, 16)


def test_training_reduces_loss() -> None:
    from densekit.data import PairDataset, TrainingPair
    from densekit.torch_backend import BiEncoder, BiEncoderTrainer, TrainConfig

    pairs = [
        TrainingPair(f"query about subject {i}", f"a passage that discusses subject {i}")
        for i in range(32)
    ]
    dataset = PairDataset(pairs)
    model = BiEncoder(vocab_size=4096, dim=32)
    trainer = BiEncoderTrainer(model, TrainConfig(epochs=6, lr=1e-2, batch_size=8, seed=0))
    history = trainer.fit(dataset)
    assert history[-1] < history[0]
