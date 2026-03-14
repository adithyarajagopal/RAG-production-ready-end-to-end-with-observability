# Production RAG System

## What This Is
Production-ready Retrieval-Augmented Generation system for document Q&A.
Built to handle Indian regulatory filings (DRHPs) including Hindi/Tamil content.

## Single Command
```
python -m uvicorn api.main:app --port 8000
```

## Project Structure
```
production-rag/
├── ingestion/
│   ├── loader.py      — PDFLoader, MarkdownLoader, TextLoader (.load() interface)
│   ├── chunker.py     — 600-token chunks, 100-token overlap
│   └── embedder.py    — BGE-M3 embeddings via sentence-transformers
├── retrieval/
│   ├── vector_store.py — ChromaDB add/search (cosine)
│   ├── bm25.py         — BM25 keyword search (rank_bm25)
│   ├── hybrid.py       — RRF fusion (k=60, top 20 from each)
│   └── reranker.py     — Cross-encoder reranking → top 5
├── generation/
│   ├── chain.py        — LLM answer generation (Claude Sonnet 4, temp=0.0)
│   ├── citation_guard.py — Refuse if citation coverage < 0.80
│   └── prompts/
│       └── qa_prompt.yaml — Versioned prompt template
├── evaluation/
│   ├── evaluator.py    — RAGAS metrics runner (faithfulness, answer relevancy)
│   └── golden_dataset.json — 5 QA pairs for benchmarking
├── observability/
│   ├── langfuse_client.py — Langfuse client init (reads keys from env)
│   └── metrics.py         — Helper functions: trace, span, score wrappers
├── api/
│   └── main.py         — FastAPI (POST /ingest, POST /query) + Langfuse traces/spans
├── .github/workflows/
│   └── eval.yml        — CI gate: RAGAS thresholds on every PR to main
├── config.yaml         — All tunable parameters
├── config.py           — Config loader (reads config.yaml at import)
└── requirements.txt
```

## Stack Decisions
- **Vector DB**: ChromaDB — runs locally, no server, good for dev
- **Embeddings**: BGE-M3 (BAAI/bge-m3) via sentence-transformers — multilingual, handles Hindi/Tamil
- **PDF parsing**: pdfplumber — text + table extraction
- **Keyword search**: rank_bm25
- **Reranking**: cross-encoder/ms-marco-MiniLM-L-12-v2 (sentence-transformers)
- **LLM**: Claude Sonnet 4 via OpenRouter (temperature 0.0)
- **API**: FastAPI + uvicorn
- **Observability**: Langfuse (cloud) — traces, spans, LLM token/cost tracking
- **Python**: 3.13

## Configuration
All parameters in `config.yaml` — zero hardcoding in Python files:
- `config.py` reads `config.yaml` once at import and exposes constants
- Every module imports from `config` instead of hardcoding values
- Change `config.yaml` → restart server → new values take effect

## Full System Architecture

### Indexing Flow (POST /ingest)
```
PDF upload → PDFLoader → RecursiveChunker (600 tokens) → BGEEmbedder (bge-m3)
  → ChromaDB add + BM25 index rebuild → {"filename", "chunks_stored"}
```

### Query Flow (POST /query)
```
Question → embed with BGE-M3 → vector search (top 20) + BM25 (top 20)
  → RRF fusion (k=60, top 20) → cross-encoder reranking (top 5)
  → Claude Sonnet 4 generation → citation guard (≥80%) → answer
```

## Ingestion Contract

### loader.py ✅
- Classes: `PDFLoader`, `MarkdownLoader`, `TextLoader` — all with `.load()` method
- `Document` dataclass: `page_content` (str) + `metadata` (dict)
- Input: file path passed to `__init__`
- Output: `list[Document]`, one per page (PDF) or one per file (Markdown/Text)
- Metadata fields: `source` (full absolute path), `file_name`, `page`, `total_pages`, `loaded_at` (UTC ISO)
- Skip pages under 50 characters (configurable via `min_chars`)

### chunker.py ✅
- Class: `RecursiveChunker(chunk_tokens=600, overlap_tokens=100)`
- Method: `.chunk(documents) -> list[Document]`
- Recursive character splitting — tries separators in order: `\n\n` → `\n` → `. ` → `, ` → ` ` → `""`
- Token approximation: `len(text) // 4` (CHARS_PER_TOKEN = 4)
- Merge pass re-joins small pieces up to chunk_size, then builds overlap from trailing pieces
- Metadata adds: `chunk_index`, `chunk_total`, `char_start`, `char_end`
- All parent metadata (source, page, etc.) propagated to every chunk

### embedder.py ✅
- Class: `BGEEmbedder(model_name="BAAI/bge-m3", batch_size=32)`
- Lazy model loading: `SentenceTransformer` instantiated on first `.embed()` / `.embed_query()` call
- `.embed(documents) -> list[Document]` — adds `metadata["embedding"]` as `list[float]`, mutates in-place
- `.embed_query(text) -> list[float]` — encodes a single query string
- `normalize_embeddings=True` — vectors are unit-length, cosine-ready
- Empty text guard: `embed_query("")` returns `[]`

## Retrieval Contract

### vector_store.py ✅
- Class: `VectorStore(collection_name="drhp_documents")`
- `.add(documents)` — requires `metadata["embedding"]` on each Document
- `.search(query_embedding, top_k=5)` — returns Documents with `metadata["score"]`
- ChromaDB in-memory client, cosine distance
- Metadata sanitized: strips embedding, JSON-encodes complex types

### bm25.py ✅
- Class: `BM25Retriever()`
- `.index(documents)` — tokenizes corpus (lowercased, whitespace split)
- `.search(query, top_k=20)` — returns Documents with `metadata["bm25_score"]`
- Filters zero-score results

### hybrid.py ✅
- Function: `rrf_fuse(vector_results, bm25_results, k=60, top_n=20)`
- RRF score: `1 / (k + rank + 1)`, summed for docs in both lists
- Dedup key: `source_page_chunk_index` (not raw text)
- Returns Documents with `metadata["rrf_score"]`

### reranker.py ✅
- Class: `CrossEncoderReranker()`
- `.rerank(query, documents, top_k=5)` — returns Documents with `metadata["rerank_score"]`
- Lazy model loading: `CrossEncoder` instantiated on first use

## Generation Contract

### chain.py ✅
- Class: `RAGChain(prompt_path=qa_prompt.yaml)`
- `.generate(query, documents) -> str`
- Context format: `[Source: {file_name}, Page {page}]\n{content}` separated by `---`
- OpenRouter → Claude Sonnet 4, temperature 0.0

### citation_guard.py ✅
- Class: `CitationGuard(threshold=0.80)`
- `.check(answer, documents) -> dict` with keys: answer, coverage, passed, total_sentences, grounded_sentences
- Grounding check: 5-gram overlap with source corpus
- If coverage < threshold: returns refusal message instead of answer

## Evaluation Contract

### evaluator.py ✅
- Class: `Evaluator(faithfulness_min=0.85, relevancy_min=0.80)`
- `.run(rag_chain) -> dict` — runs RAGAS on golden dataset, returns per-question + aggregate metrics
- `.assert_thresholds(results)` — raises AssertionError if metrics fail
- CI gate via `.github/workflows/eval.yml` on every PR to main

## API Contract

### main.py ✅
- `POST /ingest` — upload PDF → loader → chunker → embedder → vector store + BM25 → `{filename, chunks_stored}`
- `POST /query` — question → embed → vector(20) + BM25(20) → RRF(20) → rerank(5) → generate → guard → `{question, answer, citation_check}`
- Returns 400 if no documents ingested and /query is called
- All pipeline components are singletons (loaded once)

## Observability Contract

### langfuse_client.py ✅
- Singleton `Langfuse` client initialized from env vars (`LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_BASE_URL`)
- `get_client() -> Langfuse` — returns the shared client instance
- Client created once at import, reused across the app
- If env vars missing, client is `None` — observability degrades gracefully (no crashes)

### metrics.py ✅
- Helper functions wrapping Langfuse trace/span/score APIs
- `create_trace(name, user_id, metadata) -> trace` — starts a new trace for a request
- `create_span(trace, name, input, metadata) -> span` — creates a span within a trace
- `end_span(span, output, metadata)` — ends a span with output data
- `score_trace(trace, name, value, comment)` — attaches a numeric score to a trace
- `create_generation(trace, name, model, input, output, usage)` — logs an LLM generation with token counts and cost
- All functions are no-ops if Langfuse client is `None` (graceful degradation)

### Instrumentation in api/main.py
- `POST /ingest`: one trace per request, spans for: `pdf_load`, `chunk`, `embed`, `vector_add`, `bm25_index`
- `POST /query`: one trace per request, spans for: `embed_query`, `vector_search`, `bm25_search`, `rrf_fuse`, `rerank`, `generate`, `citation_guard`
- Citation coverage score attached to query traces
- Trace metadata includes: filename (ingest), question (query), chunks_stored, answer length

### Instrumentation in generation/chain.py
- LLM call wrapped with `create_generation()` to track:
  - Model name (`anthropic/claude-sonnet-4`)
  - Input (system + user messages)
  - Output (generated answer)
  - Token usage: `prompt_tokens`, `completion_tokens`, `total_tokens`
  - Cost (from OpenRouter response headers if available)

## Conventions
- One Document = one page for PDFs, one Document = one file for text/markdown
- All metadata tracked as dicts for flexibility
- Page numbers are 1-indexed
- `min_chars` threshold filters noise pages (default: 50)
- All loaders follow the same interface: `__init__(file_path)` + `.load() -> list[Document]`
- Timestamps always UTC ISO format
- All tunable values in config.yaml, never hardcoded

## Non-Negotiable Rules
1. Every loader must return `list[Document]` — no exceptions
2. Metadata must always include `source`, `file_name`, `loaded_at`
3. PDF pages under 50 chars are skipped (blank/decorative)
4. Embeddings must be normalized (cosine similarity ready)
5. Citation coverage gate at 0.80 — no hallucinated answers
6. All parameters in config.yaml — zero magic numbers in Python
