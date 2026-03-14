from __future__ import annotations

from ingestion.loader import Document
from config import CITATION_THRESHOLD, CITATION_MIN_NGRAM

_REFUSAL = (
    "I cannot provide a reliable answer. The generated response could not be "
    "sufficiently grounded in the provided source documents."
)


class CitationGuard:

    def __init__(self, threshold: float = CITATION_THRESHOLD) -> None:
        self._threshold = threshold

    def check(self, answer: str, documents: list[Document]) -> dict:
        sentences = self._split_sentences(answer)
        if not sentences:
            return {
                "answer": _REFUSAL,
                "coverage": 0.0,
                "passed": False,
                "total_sentences": 0,
                "grounded_sentences": 0,
            }

        corpus = " ".join(doc.page_content.lower() for doc in documents)

        grounded = 0
        for sentence in sentences:
            if self._is_grounded(sentence, corpus):
                grounded += 1

        total = len(sentences)
        coverage = grounded / total

        passed = coverage >= self._threshold
        return {
            "answer": answer if passed else _REFUSAL,
            "coverage": coverage,
            "passed": passed,
            "total_sentences": total,
            "grounded_sentences": grounded,
        }

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        raw = text.replace("\n", " ").split(". ")
        sentences = [s.strip().rstrip(".") for s in raw if s.strip()]
        return [s for s in sentences if len(s.split()) >= 3]

    @staticmethod
    def _is_grounded(sentence: str, corpus: str) -> bool:
        words = sentence.lower().split()
        if len(words) < CITATION_MIN_NGRAM:
            return words and all(w in corpus for w in words)
        for i in range(len(words) - CITATION_MIN_NGRAM + 1):
            ngram = " ".join(words[i : i + CITATION_MIN_NGRAM])
            if ngram in corpus:
                return True
        return False
