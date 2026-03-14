# Production RAG — DRHP Document Q&A

Production-ready Retrieval-Augmented Generation system for Indian regulatory filings (DRHPs).
Upload PDFs, ask questions, get cited answers.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        POST /ingest                             │
│  PDF ──→ PDFLoader ──→ RecursiveChunker ──→ BGEEmbedder         │
│              │              │                    │               │
│          320 pages      694 chunks          1024-dim vectors     │
│                                                  │               │
│                              ┌────────────────────┤               │
│                              ▼                    ▼               │
│                        ChromaDB              BM25 Index           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        POST /query                              │
│  Question ──→ BGE-M3 embed                                      │
│                   │                                             │
│        ┌──────────┴──────────┐                                  │
│        ▼                     ▼                                  │
│   ChromaDB search       BM25 search                             │
│    (top 20)              (top 20)                               │
│        │                     │                                  │
│        └──────────┬──────────┘                                  │
│                   ▼                                             │
│           RRF Fusion (k=60, top 20)                             │
│                   ▼                                             │
│      Cross-Encoder Reranking (top 5)                            │
│                   ▼                                             │
│       Claude Sonnet 4 (temp=0.0)                                │
│                   ▼                                             │
│    Citation Guard (≥80% grounded)                               │
│                   ▼                                             │
│          Answer with citations                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd production-rag
pip install -r requirements.txt

# 2. Set API key
echo "OPENROUTER_API_KEY=your-key-here" > .env

# 3. Start server
python -m uvicorn api.main:app --port 8000

# 4. Upload a DRHP
curl -X POST http://localhost:8000/ingest -F "file=@documents/DRHP.pdf"

# 5. Ask questions
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the authorized share capital?"}'
```

## API Endpoints

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/ingest` | POST | PDF file (multipart) | `{filename, chunks_stored}` |
| `/query` | POST | `{question: str}` | `{question, answer, citation_check}` |

## Project Structure

```
production-rag/
├── api/
│   └── main.py              ← FastAPI endpoints (/ingest, /query)
├── ingestion/
│   ├── loader.py             ← PDFLoader, Document dataclass
│   ├── chunker.py            ← 600-token chunks, 100-token overlap
│   └── embedder.py           ← BGE-M3 embeddings (multilingual)
├── retrieval/
│   ├── vector_store.py       ← ChromaDB cosine search
│   ├── bm25.py               ← BM25Okapi keyword search
│   ├── hybrid.py             ← Reciprocal Rank Fusion
│   └── reranker.py           ← Cross-encoder reranking
├── generation/
│   ├── chain.py              ← Claude Sonnet 4 via OpenRouter
│   ├── citation_guard.py     ← 5-gram grounding check
│   └── prompts/
│       └── qa_prompt.yaml    ← System + user prompt templates
├── evaluation/
│   ├── evaluator.py          ← RAGAS metrics (faithfulness, relevancy)
│   └── golden_dataset.json   ← 5 QA pairs for benchmarking
├── .github/workflows/
│   └── eval.yml              ← CI gate on PR to main
├── config.yaml               ← All tunable parameters (zero hardcoding)
├── config.py                 ← Config loader (reads config.yaml once)
├── requirements.txt
└── CLAUDE.md                 ← Dev documentation
```

## Configuration

All parameters live in `config.yaml` — nothing is hardcoded in Python:

| Parameter | Default | Where Used |
|-----------|---------|------------|
| `chunk_tokens` | 600 | chunker.py |
| `overlap_tokens` | 100 | chunker.py |
| `embedding_model` | BAAI/bge-m3 | embedder.py |
| `vector_top_k` | 20 | api/main.py |
| `bm25_top_k` | 20 | api/main.py |
| `rrf_k` | 60 | api/main.py |
| `rerank_top_k` | 5 | api/main.py |
| `reranker_model` | ms-marco-MiniLM-L-12-v2 | reranker.py |
| `llm_model` | claude-sonnet-4 | chain.py |
| `temperature` | 0.0 | chain.py |
| `citation_threshold` | 0.80 | citation_guard.py |
| `faithfulness_min` | 0.85 | evaluator.py |
| `relevancy_min` | 0.80 | evaluator.py |

## Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Embeddings | BGE-M3 | Multilingual (Hindi/Tamil/English), 1024-dim |
| Vector DB | ChromaDB | Local, no server, cosine distance |
| Keyword Search | BM25Okapi | Catches exact terms semantic search misses |
| Fusion | RRF (k=60) | Rank-based, no score normalization needed |
| Reranker | MS-MARCO MiniLM | Fast cross-encoder, strong on passage ranking |
| LLM | Claude Sonnet 4 | Via OpenRouter, temp=0.0 for deterministic output |
| PDF Parsing | pdfplumber | Text + table extraction |
| API | FastAPI | Async, auto-docs at /docs |

## Evaluation

```bash
# Run RAGAS evaluation (needs OPENROUTER_API_KEY)
python -m evaluation.evaluator
```

CI gates every PR to `main` on:
- **Faithfulness ≥ 0.85** — answers must be grounded in retrieved context
- **Answer Relevancy ≥ 0.80** — answers must address the question asked

## Requirements

- Python 3.13
- `OPENROUTER_API_KEY` in `.env`
- ~2GB disk for BGE-M3 model (downloaded on first run)
- No GPU required (CPU inference)
