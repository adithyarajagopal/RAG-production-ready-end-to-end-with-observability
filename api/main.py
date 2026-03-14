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
from observability.metrics import (
    create_trace,
    create_span,
    end_span,
    score_trace,
)

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

    trace = create_trace(
        name="ingest",
        metadata={"filename": file.filename},
        input={"filename": file.filename},
    )

    # Save upload to temp file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # ── PDF Load ──────────────────────────────────────────
        span_load = create_span(trace, "pdf_load", input={"path": tmp_path})
        pages = PDFLoader(tmp_path).load()
        end_span(span_load, output={"pages": len(pages)})

        # ── Chunk ─────────────────────────────────────────────
        span_chunk = create_span(trace, "chunk", input={"pages": len(pages)})
        chunks = RecursiveChunker().chunk(pages)
        end_span(span_chunk, output={"chunks": len(chunks)})

        # ── Embed ─────────────────────────────────────────────
        span_embed = create_span(trace, "embed", input={"chunks": len(chunks)})
        embedder.embed(chunks)
        end_span(span_embed, output={"chunks_embedded": len(chunks)})

        # ── Vector Store Add ──────────────────────────────────
        span_vadd = create_span(trace, "vector_add", input={"chunks": len(chunks)})
        vector_store.add(chunks)
        end_span(span_vadd, output={"stored": len(chunks)})

        # ── BM25 Index ────────────────────────────────────────
        _all_documents.extend(chunks)
        span_bm25 = create_span(trace, "bm25_index", input={"total_docs": len(_all_documents)})
        bm25.index(_all_documents)
        end_span(span_bm25, output={"indexed": len(_all_documents)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if trace:
        trace.update(output={"chunks_stored": len(chunks)})

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

    trace = create_trace(
        name="query",
        input={"question": question},
        metadata={"question": question},
    )

    # ── Embed Query ───────────────────────────────────────────
    span_eq = create_span(trace, "embed_query", input={"question": question})
    query_vector = embedder.embed_query(question)
    end_span(span_eq, output={"vector_dim": len(query_vector)})

    # ── Vector Search ─────────────────────────────────────────
    span_vs = create_span(trace, "vector_search", input={"top_k": VECTOR_TOP_K})
    vector_results = vector_store.search(query_vector, top_k=VECTOR_TOP_K)
    end_span(span_vs, output={"results": len(vector_results)})

    # ── BM25 Search ───────────────────────────────────────────
    span_bm = create_span(trace, "bm25_search", input={"query": question, "top_k": BM25_TOP_K})
    bm25_results = bm25.search(question, top_k=BM25_TOP_K)
    end_span(span_bm, output={"results": len(bm25_results)})

    # ── RRF Fusion ────────────────────────────────────────────
    span_rrf = create_span(trace, "rrf_fuse", input={"vector": len(vector_results), "bm25": len(bm25_results)})
    fused = rrf_fuse(vector_results, bm25_results, k=RRF_K, top_n=RRF_TOP_N)
    end_span(span_rrf, output={"fused": len(fused)})

    # ── Rerank ────────────────────────────────────────────────
    span_rr = create_span(trace, "rerank", input={"candidates": len(fused), "top_k": RERANK_TOP_K})
    top5 = reranker.rerank(question, fused, top_k=RERANK_TOP_K)
    end_span(span_rr, output={"reranked": len(top5)})

    # ── Generate (trace passed to chain for LLM generation tracking)
    span_gen = create_span(trace, "generate", input={"question": question, "context_chunks": len(top5)})
    raw_answer = rag_chain.generate(question, top5, trace=trace)
    end_span(span_gen, output={"answer_length": len(raw_answer)})

    # ── Citation Guard ────────────────────────────────────────
    span_cg = create_span(trace, "citation_guard", input={"answer_length": len(raw_answer)})
    guarded = citation_guard.check(raw_answer, top5)
    end_span(span_cg, output={
        "coverage": guarded["coverage"],
        "passed": guarded["passed"],
    })

    # ── Score the trace with citation coverage ────────────────
    score_trace(
        trace,
        name="citation_coverage",
        value=guarded["coverage"],
        comment=f"{'PASSED' if guarded['passed'] else 'FAILED'} — "
                f"{guarded['grounded_sentences']}/{guarded['total_sentences']} grounded",
    )

    if trace:
        trace.update(output={
            "answer": guarded["answer"][:200],
            "coverage": guarded["coverage"],
            "passed": guarded["passed"],
        })

    return QueryResponse(
        question=question,
        answer=guarded["answer"],
        citation_check={
            "coverage": guarded["coverage"],
            "passed": guarded["passed"],
        },
    )
