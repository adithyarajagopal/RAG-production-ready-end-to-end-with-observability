from __future__ import annotations

import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from ingestion.loader import Document, PDFLoader
from ingestion.chunker import RecursiveChunker
from ingestion.embedder import BGEEmbedder
from retrieval.vector_store import VectorStore
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import rrf_fuse
from retrieval.reranker import CrossEncoderReranker
from generation.chain import RAGChain
from generation.citation_guard import CitationGuard
from config import VECTOR_TOP_K, BM25_TOP_K, RRF_K, RRF_TOP_N, RERANK_TOP_K

load_dotenv()

app = FastAPI(title="DRHP RAG API")

# ── Shared singletons ──────────────────────────────────────────────
embedder = BGEEmbedder()
vector_store = VectorStore()
bm25 = BM25Retriever()
reranker = CrossEncoderReranker()
rag_chain = RAGChain()
citation_guard = CitationGuard()

_all_documents: list[Document] = []


# ── Request / response models ──────────────────────────────────────
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citation_check: dict


class IngestResponse(BaseModel):
    filename: str
    chunks_stored: int


# ── POST /ingest ───────────────────────────────────────────────────
@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save upload to temp file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Loader → Chunker → Embedder → Store
        pages = PDFLoader(tmp_path).load()
        chunks = RecursiveChunker().chunk(pages)
        embedder.embed(chunks)
        vector_store.add(chunks)

        # Rebuild BM25 index with all documents
        _all_documents.extend(chunks)
        bm25.index(_all_documents)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(filename=file.filename, chunks_stored=len(chunks))


# ── POST /query ────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    if not _all_documents:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. POST a PDF to /ingest first.",
        )

    question = body.question

    # Embed query
    query_vector = embedder.embed_query(question)

    # Retrieve: vector + BM25 → RRF → rerank
    vector_results = vector_store.search(query_vector, top_k=VECTOR_TOP_K)
    bm25_results = bm25.search(question, top_k=BM25_TOP_K)
    fused = rrf_fuse(vector_results, bm25_results, k=RRF_K, top_n=RRF_TOP_N)
    top5 = reranker.rerank(question, fused, top_k=RERANK_TOP_K)

    # Generate + guard
    raw_answer = rag_chain.generate(question, top5)
    guarded = citation_guard.check(raw_answer, top5)

    return QueryResponse(
        question=question,
        answer=guarded["answer"],
        citation_check={
            "coverage": guarded["coverage"],
            "passed": guarded["passed"],
        },
    )
