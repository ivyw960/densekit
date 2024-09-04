"""Command-line interface for densekit.

Subcommands cover the offline workflow end to end: ``encode`` a JSONL corpus
into embeddings, ``build`` an index from those embeddings, ``search`` the index
with query embeddings, and ``evaluate`` a run against qrels. Everything reads and
writes plain files so pipelines stay reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from densekit.data import Corpus
from densekit.encoders import HashingEncoder
from densekit.eval import Qrels, RetrievalEvaluator
from densekit.index import BaseIndex, FlatIndex, IVFIndex, LSHIndex, PQIndex
from densekit.io import load_index, save_index


def _make_index(kind: str, dim: int, args: argparse.Namespace) -> BaseIndex:
    if kind == "flat":
        return FlatIndex(dim, args.metric)
    if kind == "ivf":
        return IVFIndex(
            dim, nlist=args.nlist, nprobe=args.nprobe, metric=args.metric, seed=args.seed
        )
    if kind == "lsh":
        return LSHIndex(
            dim, n_bits=args.n_bits, n_tables=args.n_tables, metric=args.metric, seed=args.seed
        )
    return PQIndex(
        dim, n_subvectors=args.n_subvectors, n_bits=args.pq_bits, metric=args.metric, seed=args.seed
    )


def _cmd_encode(args: argparse.Namespace) -> int:
    corpus = Corpus.from_jsonl(args.corpus)
    encoder = HashingEncoder(
        dim=args.dim,
        analyzer=args.analyzer,
        ngram_range=(args.ngram[0], args.ngram[1]),
    )
    embeddings = encoder.encode(corpus.texts())
    np.save(args.out, embeddings)
    Path(args.ids).write_text(json.dumps(corpus.ids()), encoding="utf-8")
    print(f"encoded {len(corpus)} docs -> {args.out} ({embeddings.shape[1]}d)")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    embeddings = np.load(args.embeddings)
    index = _make_index(args.index, embeddings.shape[1], args)
    if hasattr(index, "train"):
        index.train(embeddings)
    index.add(embeddings)
    save_index(index, args.out)
    print(f"built {args.index} index over {index.size} vectors -> {args.out}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    index = load_index(args.index)
    queries = np.load(args.queries)
    result = index.search(queries, k=args.k)
    for i in range(result.n_queries):
        hits = [{"id": idx, "score": score} for idx, score in result.for_query(i)]
        print(json.dumps({"query": i, "hits": hits}))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    qrels = Qrels()
    with Path(args.qrels).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                qrels.add(row["query_id"], row["doc_id"], float(row.get("relevance", 1.0)))

    run: dict[str, list[str]] = {}
    with Path(args.run).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                run[str(row["query_id"])] = [str(d) for d in row["doc_ids"]]

    evaluator = RetrievalEvaluator(qrels, k_values=args.k)
    scores = evaluator.evaluate(run)
    print(json.dumps(scores, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="densekit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    encode = sub.add_parser("encode", help="encode a JSONL corpus into embeddings")
    encode.add_argument("--corpus", required=True, help="input corpus JSONL")
    encode.add_argument("--out", required=True, help="output embeddings .npy")
    encode.add_argument("--ids", required=True, help="output document-ids .json")
    encode.add_argument("--dim", type=int, default=256)
    encode.add_argument("--analyzer", choices=["word", "char"], default="word")
    encode.add_argument("--ngram", type=int, nargs=2, default=(1, 1), metavar=("MIN", "MAX"))
    encode.set_defaults(func=_cmd_encode)

    build = sub.add_parser("build", help="build an index from embeddings")
    build.add_argument("--embeddings", required=True, help="input embeddings .npy")
    build.add_argument("--out", required=True, help="output index .npz")
    build.add_argument("--index", choices=["flat", "ivf", "lsh", "pq"], default="flat")
    build.add_argument("--metric", choices=["cosine", "dot", "l2"], default="cosine")
    build.add_argument("--nlist", type=int, default=64)
    build.add_argument("--nprobe", type=int, default=8)
    build.add_argument("--n-bits", dest="n_bits", type=int, default=16)
    build.add_argument("--n-tables", dest="n_tables", type=int, default=8)
    build.add_argument("--n-subvectors", dest="n_subvectors", type=int, default=8)
    build.add_argument("--pq-bits", dest="pq_bits", type=int, default=8)
    build.add_argument("--seed", type=int, default=0)
    build.set_defaults(func=_cmd_build)

    search = sub.add_parser("search", help="search an index with query embeddings")
    search.add_argument("--index", required=True, help="input index .npz")
    search.add_argument("--queries", required=True, help="query embeddings .npy")
    search.add_argument("--k", type=int, default=10)
    search.set_defaults(func=_cmd_search)

    evaluate = sub.add_parser("evaluate", help="score a run against qrels")
    evaluate.add_argument("--run", required=True, help="run JSONL (query_id, doc_ids)")
    evaluate.add_argument(
        "--qrels", required=True, help="qrels JSONL (query_id, doc_id, relevance)"
    )
    evaluate.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    evaluate.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
