from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from ingestion.loader import Document


class BM25Retriever:

    def __init__(self) -> None:
        self._index: BM25Okapi | None = None
        self._documents: list[Document] = []

    def index(self, documents: list[Document]) -> None:
        self._documents = list(documents)
        tokenized_corpus = [
            doc.page_content.lower().split() for doc in self._documents
        ]
        self._index = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 20) -> list[Document]:
        if self._index is None:
            raise RuntimeError("Call index() before search()")

        tokenized_query = query.lower().split()
        scores = self._index.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[Document] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            doc = self._documents[idx]
            meta = dict(doc.metadata)
            meta["bm25_score"] = float(scores[idx])
            results.append(Document(page_content=doc.page_content, metadata=meta))

        return results
