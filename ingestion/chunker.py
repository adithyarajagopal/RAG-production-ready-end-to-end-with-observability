"""
Recursive character text chunker for the ingestion pipeline.

Splits documents into ~600-token chunks with 100-token overlap.
Tries clean boundaries in order: paragraph → sentence → word → char.

Tokens are approximated as len(text) // 4.

Usage:
    from ingestion.chunker import RecursiveChunker
    chunker = RecursiveChunker()
    chunks = chunker.chunk(documents)
"""

from __future__ import annotations

from ingestion.loader import Document
from config import CHUNK_TOKENS, OVERLAP_TOKENS


# ---------------------------------------------------------------------------
# Separator hierarchy — tried in order, coarsest to finest
# ---------------------------------------------------------------------------

DEFAULT_SEPARATORS = [
    "\n\n",   # paragraph breaks
    "\n",     # line breaks
    ". ",     # sentence boundaries
    ", ",     # clause boundaries
    " ",      # words
    "",       # character-level fallback (guaranteed to split)
]


# ---------------------------------------------------------------------------
# Recursive Character Chunker
# ---------------------------------------------------------------------------

class RecursiveChunker:
    """Split Documents into token-bounded chunks with overlap.

    Args:
        chunk_tokens:  Target chunk size in tokens (default 600).
        overlap_tokens: Overlap between consecutive chunks (default 100).
        separators:     Ordered list of split boundaries to try.

    Token approximation: 1 token ≈ 4 characters.
    """

    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        chunk_tokens: int = CHUNK_TOKENS,
        overlap_tokens: int = OVERLAP_TOKENS,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_tokens * self.CHARS_PER_TOKEN      # 2400 chars
        self.chunk_overlap = overlap_tokens * self.CHARS_PER_TOKEN  # 400 chars
        self.separators = separators or DEFAULT_SEPARATORS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Chunk a list of Documents.  Returns new Documents with updated metadata."""
        all_chunks: list[Document] = []

        for doc in documents:
            text = doc.page_content
            if not text.strip():
                continue

            pieces = self._split_text(text, self.separators)
            merged = self._merge_with_overlap(pieces)

            for idx, (chunk_text, char_start, char_end) in enumerate(merged):
                chunk_meta = {
                    **doc.metadata,
                    "chunk_index": idx,
                    "chunk_total": len(merged),
                    "char_start": char_start,
                    "char_end": char_end,
                }
                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata=chunk_meta,
                ))

        return all_chunks

    # ------------------------------------------------------------------
    # Recursive splitting
    # ------------------------------------------------------------------

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the separator hierarchy.

        Picks the first separator that actually appears in the text,
        splits on it, then recurses on any piece that's still too long
        using the remaining (finer) separators.
        """
        # Base case — text fits in one chunk, return as-is
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Find the best (coarsest) separator that exists in this text
        chosen_sep = ""
        remaining_seps = []
        for i, sep in enumerate(separators):
            if sep == "":
                # Character-level fallback is always available
                chosen_sep = sep
                remaining_seps = []
                break
            if sep in text:
                chosen_sep = sep
                remaining_seps = separators[i + 1:]
                break

        # Split on the chosen separator
        if chosen_sep == "":
            # Hard split at chunk_size boundary (last resort)
            parts = [
                text[i:i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]
        else:
            raw_parts = text.split(chosen_sep)
            # Re-attach the separator to the end of each piece (except last)
            # so we don't lose paragraph/sentence boundaries in the output
            parts = []
            for j, part in enumerate(raw_parts):
                if j < len(raw_parts) - 1:
                    parts.append(part + chosen_sep)
                else:
                    parts.append(part)

        # Recurse on any piece that is still too long
        final: list[str] = []
        for part in parts:
            if not part.strip():
                continue
            if len(part) <= self.chunk_size:
                final.append(part)
            else:
                final.extend(self._split_text(part, remaining_seps))

        return final

    # ------------------------------------------------------------------
    # Merge small pieces into chunks with overlap
    # ------------------------------------------------------------------

    def _merge_with_overlap(
        self, pieces: list[str]
    ) -> list[tuple[str, int, int]]:
        """Merge small pieces into chunks respecting size and overlap.

        Returns list of (chunk_text, char_start, char_end) tuples.
        char_start and char_end are offsets into the original page_content.
        """
        if not pieces:
            return []

        chunks: list[tuple[str, int, int]] = []
        current_parts: list[str] = []
        current_len = 0

        # Track position in original text
        # Each piece maps to a contiguous region of the source text
        piece_offsets: list[int] = []
        offset = 0
        for p in pieces:
            piece_offsets.append(offset)
            offset += len(p)

        # Index tracking which pieces are in the current buffer
        current_piece_indices: list[int] = []

        for i, piece in enumerate(pieces):
            # Would adding this piece exceed chunk_size?
            candidate_len = current_len + len(piece)

            if candidate_len > self.chunk_size and current_parts:
                # Flush current buffer as a chunk
                chunk_text = "".join(current_parts)
                char_start = piece_offsets[current_piece_indices[0]]
                char_end = char_start + len(chunk_text)
                chunks.append((chunk_text.strip(), char_start, char_end))

                # Build overlap: walk backwards keeping pieces until we
                # fill the overlap budget
                overlap_parts: list[str] = []
                overlap_len = 0
                overlap_indices: list[int] = []
                for idx in reversed(current_piece_indices):
                    p = pieces[idx]
                    if overlap_len + len(p) > self.chunk_overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_indices.insert(0, idx)
                    overlap_len += len(p)

                current_parts = overlap_parts
                current_len = overlap_len
                current_piece_indices = overlap_indices

            current_parts.append(piece)
            current_piece_indices.append(i)
            current_len += len(piece)

        # Flush remaining buffer
        if current_parts:
            chunk_text = "".join(current_parts)
            char_start = piece_offsets[current_piece_indices[0]]
            char_end = char_start + len(chunk_text)
            chunks.append((chunk_text.strip(), char_start, char_end))

        return chunks
