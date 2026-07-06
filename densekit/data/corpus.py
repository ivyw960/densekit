"""Corpus and document containers with a simple JSONL on-disk format.

densekit is offline-first, so the canonical corpus format is a plain JSONL file
with one ``{"id": ..., "text": ...}`` object per line -- no network, no binary
blobs, easy to diff and version.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Document:
    """A single corpus item."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Corpus:
    """An ordered collection of :class:`Document` with id lookup."""

    def __init__(self, documents: Iterable[Document] | None = None) -> None:
        self._docs: list[Document] = []
        self._index: dict[str, int] = {}
        for doc in documents or []:
            self.add(doc)

    def add(self, document: Document) -> None:
        if document.doc_id in self._index:
            msg = f"duplicate doc_id: {document.doc_id!r}"
            raise ValueError(msg)
        self._index[document.doc_id] = len(self._docs)
        self._docs.append(document)

    def texts(self) -> list[str]:
        return [doc.text for doc in self._docs]

    def ids(self) -> list[str]:
        return [doc.doc_id for doc in self._docs]

    def position(self, doc_id: str) -> int:
        """Row index of ``doc_id`` (as returned by an index search)."""
        return self._index[doc_id]

    def __len__(self) -> int:
        return len(self._docs)

    def __iter__(self) -> Iterator[Document]:
        return iter(self._docs)

    def __getitem__(self, i: int) -> Document:
        return self._docs[i]

    @classmethod
    def from_jsonl(cls, path: str | Path, *, text_key: str = "text", id_key: str = "id") -> Corpus:
        corpus = cls()
        with Path(path).open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                doc_id = str(obj.get(id_key, lineno))
                metadata = {k: v for k, v in obj.items() if k not in {id_key, text_key}}
                corpus.add(Document(doc_id=doc_id, text=obj[text_key], metadata=metadata))
        return corpus

    def to_jsonl(self, path: str | Path, *, text_key: str = "text", id_key: str = "id") -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            for doc in self._docs:
                row = {id_key: doc.doc_id, text_key: doc.text, **doc.metadata}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
