"""
Document loaders for the ingestion pipeline.

Three loader classes with a uniform interface:
    PDFLoader      — page-by-page extraction via pdfplumber
    MarkdownLoader — single-document load for .md files
    TextLoader     — single-document load for .txt files

Every loader follows:
    __init__(file_path: str)
    .load() -> list[Document]
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber


# ---------------------------------------------------------------------------
# Core data structure — used across the entire pipeline
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """Single unit of content flowing through the RAG pipeline.

    Attributes:
        page_content: The extracted text content.
        metadata:     Dict carrying source, page number, timestamps, etc.
                      Keys vary by loader but always include:
                        source     — full absolute path to the file
                        file_name  — just the filename
                        loaded_at  — UTC ISO timestamp
    """
    page_content: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PDF Loader
# ---------------------------------------------------------------------------

class PDFLoader:
    """Load a PDF file page-by-page using pdfplumber.

    Returns one Document per page.  Pages with fewer than `min_chars`
    characters of extracted text are skipped (blank / decorative pages).

    Usage:
        loader = PDFLoader("path/to/file.pdf")
        docs = loader.load()
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self, file_path: str, min_chars: int = 50) -> None:
        self.path = Path(file_path).resolve()
        self.min_chars = min_chars
        self._validate()

    def _validate(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        if self.path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{self.path.suffix}'. "
                f"PDFLoader accepts: {self.SUPPORTED_EXTENSIONS}"
            )

    def load(self) -> list[Document]:
        loaded_at = datetime.now(timezone.utc).isoformat()
        documents: list[Document] = []

        with pdfplumber.open(self.path) as pdf:
            total_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()

                if len(text) < self.min_chars:
                    continue

                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": str(self.path),
                        "file_name": self.path.name,
                        "page": page_num,
                        "total_pages": total_pages,
                        "loaded_at": loaded_at,
                    },
                ))

        return documents


# ---------------------------------------------------------------------------
# Markdown Loader
# ---------------------------------------------------------------------------

class MarkdownLoader:
    """Load a Markdown file as a single Document.

    Usage:
        loader = MarkdownLoader("path/to/file.md")
        docs = loader.load()
    """

    SUPPORTED_EXTENSIONS = {".md", ".markdown"}

    def __init__(self, file_path: str, min_chars: int = 50) -> None:
        self.path = Path(file_path).resolve()
        self.min_chars = min_chars
        self._validate()

    def _validate(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        if self.path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{self.path.suffix}'. "
                f"MarkdownLoader accepts: {self.SUPPORTED_EXTENSIONS}"
            )

    def load(self) -> list[Document]:
        loaded_at = datetime.now(timezone.utc).isoformat()
        text = self.path.read_text(encoding="utf-8").strip()

        if len(text) < self.min_chars:
            return []

        return [Document(
            page_content=text,
            metadata={
                "source": str(self.path),
                "file_name": self.path.name,
                "page": 1,
                "total_pages": 1,
                "loaded_at": loaded_at,
            },
        )]


# ---------------------------------------------------------------------------
# Text Loader
# ---------------------------------------------------------------------------

class TextLoader:
    """Load a plain text file as a single Document.

    Usage:
        loader = TextLoader("path/to/file.txt")
        docs = loader.load()
    """

    SUPPORTED_EXTENSIONS = {".txt"}

    def __init__(self, file_path: str, min_chars: int = 50) -> None:
        self.path = Path(file_path).resolve()
        self.min_chars = min_chars
        self._validate()

    def _validate(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        if self.path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{self.path.suffix}'. "
                f"TextLoader accepts: {self.SUPPORTED_EXTENSIONS}"
            )

    def load(self) -> list[Document]:
        loaded_at = datetime.now(timezone.utc).isoformat()
        text = self.path.read_text(encoding="utf-8").strip()

        if len(text) < self.min_chars:
            return []

        return [Document(
            page_content=text,
            metadata={
                "source": str(self.path),
                "file_name": self.path.name,
                "page": 1,
                "total_pages": 1,
                "loaded_at": loaded_at,
            },
        )]
