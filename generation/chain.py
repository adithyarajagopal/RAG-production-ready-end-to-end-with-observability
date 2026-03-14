from __future__ import annotations

import os
from pathlib import Path

import yaml
from openai import OpenAI

from ingestion.loader import Document
from config import LLM_MODEL, LLM_TEMPERATURE

_DEFAULT_PROMPT = Path(__file__).parent / "prompts" / "qa_prompt.yaml"


class RAGChain:

    def __init__(self, prompt_path: str | Path = _DEFAULT_PROMPT) -> None:
        with open(prompt_path) as f:
            template = yaml.safe_load(f)
        self._system_prompt: str = template["system"]
        self._user_template: str = template["user"]
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )

    @staticmethod
    def _build_context(documents: list[Document]) -> str:
        chunks: list[str] = []
        for doc in documents:
            source = doc.metadata.get("file_name", "unknown")
            page = doc.metadata.get("page", "?")
            chunks.append(
                f"[Source: {source}, Page {page}]\n{doc.page_content}"
            )
        return "\n---\n".join(chunks)

    def generate(self, query: str, documents: list[Document]) -> str:
        context = self._build_context(documents)
        user_msg = self._user_template.format(context=context, question=query)

        response = self._client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=LLM_TEMPERATURE,
        )
        return response.choices[0].message.content
