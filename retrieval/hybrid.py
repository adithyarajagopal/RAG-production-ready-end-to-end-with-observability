from __future__ import annotations

from ingestion.loader import Document


def _doc_key(doc: Document) -> str:
    m = doc.metadata
    return f"{m.get('source', '')}_{m.get('page', '')}_{m.get('chunk_index', '')}"


def rrf_fuse(
    vector_results: list[Document],
    bm25_results: list[Document],
    k: int = 60,
    top_n: int = 20,
) -> list[Document]:
    scored: dict[str, dict] = {}

    for rank, doc in enumerate(vector_results):
        key = _doc_key(doc)
        rrf_score = 1.0 / (k + rank + 1)
        if key in scored:
            scored[key]["rrf_score"] += rrf_score
        else:
            scored[key] = {"doc": doc, "rrf_score": rrf_score}

    for rank, doc in enumerate(bm25_results):
        key = _doc_key(doc)
        rrf_score = 1.0 / (k + rank + 1)
        if key in scored:
            scored[key]["rrf_score"] += rrf_score
        else:
            scored[key] = {"doc": doc, "rrf_score": rrf_score}

    ranked = sorted(scored.values(), key=lambda x: x["rrf_score"], reverse=True)

    results: list[Document] = []
    for entry in ranked[:top_n]:
        doc = entry["doc"]
        meta = dict(doc.metadata)
        meta["rrf_score"] = entry["rrf_score"]
        results.append(Document(page_content=doc.page_content, metadata=meta))

    return results
