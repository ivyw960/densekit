import json
from pathlib import Path

from densekit.cli import main


def _write_corpus(path: Path) -> None:
    rows = [
        {"id": "d1", "text": "dense retrieval with bi encoders"},
        {"id": "d2", "text": "approximate nearest neighbour search"},
        {"id": "d3", "text": "product quantization for compression"},
        {"id": "d4", "text": "evaluation metrics like ndcg and recall"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_encode_build_search_pipeline(tmp_path: Path, capsys) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    emb = tmp_path / "emb.npy"
    ids = tmp_path / "ids.json"
    index = tmp_path / "index.npz"

    assert (
        main(
            ["encode", "--corpus", str(corpus), "--out", str(emb), "--ids", str(ids), "--dim", "64"]
        )
        == 0
    )
    assert main(["build", "--embeddings", str(emb), "--out", str(index), "--index", "flat"]) == 0
    capsys.readouterr()  # drop encode/build chatter so only search output remains
    assert main(["search", "--index", str(index), "--queries", str(emb), "--k", "2"]) == 0

    out = capsys.readouterr().out.strip().splitlines()
    first = json.loads(out[0])
    # the nearest neighbour of a document is itself
    assert first["hits"][0]["id"] == 0
    assert json.loads(ids.read_text()) == ["d1", "d2", "d3", "d4"]


def test_evaluate_command(tmp_path: Path, capsys) -> None:
    run = tmp_path / "run.jsonl"
    qrels = tmp_path / "qrels.jsonl"
    run.write_text(json.dumps({"query_id": "q1", "doc_ids": ["d1", "d2"]}) + "\n", encoding="utf-8")
    qrels.write_text(
        json.dumps({"query_id": "q1", "doc_id": "d1", "relevance": 1}) + "\n", encoding="utf-8"
    )
    assert main(["evaluate", "--run", str(run), "--qrels", str(qrels), "--k", "1"]) == 0
    scores = json.loads(capsys.readouterr().out)
    assert scores["recall@1"] == 1.0
