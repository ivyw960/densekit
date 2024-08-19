from pathlib import Path

from densekit.data import Corpus, Document, PairDataset, TrainingPair


def test_corpus_jsonl_round_trip(tmp_path: Path) -> None:
    corpus = Corpus(
        [
            Document("a", "first document", {"lang": "en"}),
            Document("b", "second document"),
        ]
    )
    path = tmp_path / "corpus.jsonl"
    corpus.to_jsonl(path)

    loaded = Corpus.from_jsonl(path)
    assert loaded.ids() == ["a", "b"]
    assert loaded.texts() == ["first document", "second document"]
    assert loaded[0].metadata == {"lang": "en"}
    assert loaded.position("b") == 1


def test_corpus_rejects_duplicate_ids() -> None:
    import pytest

    with pytest.raises(ValueError, match="duplicate"):
        Corpus([Document("x", "one"), Document("x", "two")])


def test_pair_dataset_batches_cover_all(tmp_path: Path) -> None:
    pairs = [TrainingPair(f"q{i}", f"p{i}") for i in range(10)]
    ds = PairDataset(pairs)
    seen = [p.query for batch in ds.batches(batch_size=3, shuffle=True, seed=1) for p in batch]
    assert sorted(seen) == sorted(p.query for p in pairs)


def test_pair_dataset_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"query": "q1", "positive": "p1", "negatives": ["n1"]}\n'
        '{"query": "q2", "positive": "p2"}\n',
        encoding="utf-8",
    )
    ds = PairDataset.from_jsonl(path)
    assert len(ds) == 2
    assert ds[0].negatives == ("n1",)
    assert ds[1].negatives == ()
