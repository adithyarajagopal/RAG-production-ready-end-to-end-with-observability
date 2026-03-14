"""
BGE-M3 embedding module for the ingestion pipeline.

Encodes document chunks into dense vectors using BAAI/bge-m3,
a multilingual model that handles Hindi, Tamil, and English —
critical for Indian regulatory filings (DRHPs).

Usage:
    from ingestion.embedder import BGEEmbedder
    embedder = BGEEmbedder()
    docs = embedder.embed(chunked_docs)       # batch embed
    qvec = embedder.embed_query("my query")   # single query
"""

from __future__ import annotations

from ingestion.loader import Document
from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE


class BGEEmbedder:
    """Embed documents and queries using sentence-transformers.

    Model is lazy-loaded on first call to avoid startup cost when
    the class is imported but not immediately used (~2-3s saved).

    Args:
        model_name:  HuggingFace model ID. Default: BAAI/bge-m3.
        batch_size:  Batch size for encoding. Default: 32.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        """Lazy-load the SentenceTransformer model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, documents: list[Document]) -> list[Document]:
        """Add embeddings to each Document's metadata.

        Encodes page_content in batches with normalization enabled.
        Writes metadata["embedding"] as list[float] on each Document.
        Returns the same list, mutated in-place.
        """
        if not documents:
            return documents

        model = self._load_model()
        texts = [doc.page_content for doc in documents]

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        for doc, vec in zip(documents, embeddings):
            doc.metadata["embedding"] = vec.tolist()

        return documents

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Uses the same model and normalization as document embedding
        so that query and document vectors are directly comparable.
        """
        if not text.strip():
            return []

        model = self._load_model()

        vec = model.encode(
            text,
            normalize_embeddings=True,
        )
        return vec.tolist()
