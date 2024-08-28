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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
