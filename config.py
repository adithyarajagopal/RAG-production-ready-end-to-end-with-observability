"""Central configuration loader — reads config.yaml once at import."""

from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

with open(_CONFIG_PATH) as _f:
    _raw = yaml.safe_load(_f)

# ── Ingestion ──────────────────────────────────────────────────
CHUNK_TOKENS: int = _raw["ingestion"]["chunk_tokens"]
OVERLAP_TOKENS: int = _raw["ingestion"]["overlap_tokens"]
MIN_PAGE_CHARS: int = _raw["ingestion"]["min_page_chars"]
EMBEDDING_MODEL: str = _raw["ingestion"]["embedding_model"]
EMBEDDING_BATCH_SIZE: int = _raw["ingestion"]["embedding_batch_size"]

# ── Retrieval ──────────────────────────────────────────────────
VECTOR_TOP_K: int = _raw["retrieval"]["vector_top_k"]
BM25_TOP_K: int = _raw["retrieval"]["bm25_top_k"]
RRF_K: int = _raw["retrieval"]["rrf_k"]
RRF_TOP_N: int = _raw["retrieval"]["rrf_top_n"]
RERANK_TOP_K: int = _raw["retrieval"]["rerank_top_k"]
RERANKER_MODEL: str = _raw["retrieval"]["reranker_model"]
VECTOR_COLLECTION: str = _raw["retrieval"]["vector_collection"]

# ── Generation ─────────────────────────────────────────────────
LLM_MODEL: str = _raw["generation"]["llm_model"]
LLM_TEMPERATURE: float = _raw["generation"]["temperature"]
CITATION_THRESHOLD: float = _raw["generation"]["citation_threshold"]
CITATION_MIN_NGRAM: int = _raw["generation"]["citation_min_ngram"]

# ── Evaluation ─────────────────────────────────────────────────
FAITHFULNESS_MIN: float = _raw["evaluation"]["faithfulness_min"]
RELEVANCY_MIN: float = _raw["evaluation"]["relevancy_min"]
