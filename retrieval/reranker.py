from __future__ import annotations

from ingestion.loader import Document
from config import RERANKER_MODEL


class CrossEncoderReranker:

    def __init__(self) -> None:
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(RERANKER_MODEL)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        if not documents:
            return []

        model = self._load_model()

        pairs = [[query, doc.page_content] for doc in documents]
        scores = model.predict(pairs)

        scored = sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
        )

        results: list[Document] = []
        for doc, score in scored[:top_k]:
            meta = dict(doc.metadata)
            meta["rerank_score"] = float(score)
            results.append(Document(page_content=doc.page_content, metadata=meta))

        return results
