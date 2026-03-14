from __future__ import annotations

import json
import uuid

import chromadb

from ingestion.loader import Document
from config import VECTOR_COLLECTION


class VectorStore:

    def __init__(self, collection_name: str = VECTOR_COLLECTION):
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, documents: list[Document]) -> None:
        if not documents:
            return

        ids = []
        embeddings = []
        texts = []
        metadatas = []

        for doc in documents:
            if "embedding" not in doc.metadata:
                raise ValueError(
                    f"Document missing 'embedding' in metadata: "
                    f"{doc.page_content[:80]!r}"
                )

            ids.append(uuid.uuid4().hex)
            embeddings.append(doc.metadata["embedding"])
            texts.append(doc.page_content)
            metadatas.append(self._sanitize_metadata(doc.metadata))

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[Document]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            meta["score"] = distance
            documents.append(Document(page_content=text, metadata=meta))

        return documents

    @staticmethod
    def _sanitize_metadata(meta: dict) -> dict:
        clean = {}
        for k, v in meta.items():
            if k == "embedding":
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif v is None:
                continue
            else:
                clean[k] = json.dumps(v)
        return clean
