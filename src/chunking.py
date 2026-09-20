from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Split after terminal punctuation so the punctuation remains attached
        # to its sentence. Newlines also cover the documented ``".\n"`` case.
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])(?:[ \t]+|\r?\n+)", text.strip())
            if sentence.strip()
        ]

        return [
            " ".join(sentences[start : start + self.max_sentences_per_chunk])
            for start in range(0, len(sentences), self.max_sentences_per_chunk)
        ]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = max(1, chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return [chunk for chunk in self._split(text.strip(), self.separators) if chunk]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text.strip()]

        # With no useful separator left, fall back to a hard character split.
        if not remaining_separators:
            return [
                current_text[start : start + self.chunk_size].strip()
                for start in range(0, len(current_text), self.chunk_size)
                if current_text[start : start + self.chunk_size].strip()
            ]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]
        if separator == "":
            return self._split(current_text, [])
        if separator not in current_text:
            return self._split(current_text, next_separators)

        raw_parts = current_text.split(separator)
        pieces: list[str] = []
        for index, part in enumerate(raw_parts):
            if not part:
                continue
            # Retain separators where possible so splitting does not silently
            # alter the original prose.
            piece = part + (separator if index < len(raw_parts) - 1 else "")
            if len(piece) > self.chunk_size:
                pieces.extend(self._split(piece, next_separators))
            else:
                pieces.append(piece.strip())

        # Recombine adjacent small pieces up to the target size. Without this
        # step paragraph/word splitting would produce many tiny chunks.
        merged: list[str] = []
        buffer = ""
        for piece in pieces:
            if not piece:
                continue
            joiner = "" if not buffer or buffer.endswith((" ", "\n")) else " "
            candidate = f"{buffer}{joiner}{piece}" if buffer else piece
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    merged.append(buffer.strip())
                buffer = piece
        if buffer:
            merged.append(buffer.strip())
        return merged


class HeadingChunker:
    """Split Markdown by headings while preserving hierarchical context.

    Short sections remain intact. Oversized sections are split recursively,
    with their parent and current headings prepended to every child chunk.
    """

    HEADING_BOUNDARY = re.compile(r"(?m)(?=^#{1,6}\s+)")
    HEADING_LINE = re.compile(r"^(#{1,6})\s+(.+)$")

    def __init__(self, max_size: int = 400) -> None:
        self.max_size = max(1, max_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        clean_text = text.strip()
        sections = [
            section.strip()
            for section in self.HEADING_BOUNDARY.split(clean_text)
            if section.strip()
        ]

        if not any(self.HEADING_LINE.match(section.split("\n", 1)[0]) for section in sections):
            return RecursiveChunker(chunk_size=self.max_size).chunk(clean_text)

        chunks: list[str] = []
        heading_stack: list[tuple[int, str]] = []
        for section in sections:
            first_line, separator, body = section.partition("\n")
            heading_match = self.HEADING_LINE.match(first_line.strip())
            if not heading_match:
                chunks.extend(RecursiveChunker(chunk_size=self.max_size).chunk(section))
                continue

            level = len(heading_match.group(1))
            heading = first_line.strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading))

            heading_context = "\n".join(item[1] for item in heading_stack)
            body = body.strip() if separator else ""
            contextual_section = f"{heading_context}\n{body}".strip()

            if len(contextual_section) <= self.max_size:
                chunks.append(contextual_section)
                continue

            if not body or len(heading_context) + 1 >= self.max_size:
                chunks.extend(RecursiveChunker(chunk_size=self.max_size).chunk(contextual_section))
                continue

            body_budget = self.max_size - len(heading_context) - 1
            for child in RecursiveChunker(chunk_size=body_budget).chunk(body):
                chunks.append(f"{heading_context}\n{child}".strip())

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    magnitude_a = math.sqrt(_dot(vec_a, vec_a))
    magnitude_b = math.sqrt(_dot(vec_b, vec_b))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        safe_chunk_size = max(1, chunk_size)
        strategies = {
            "fixed_size": FixedSizeChunker(
                chunk_size=safe_chunk_size,
                overlap=min(50, safe_chunk_size - 1),
            ),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=safe_chunk_size),
        }

        comparison = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            comparison[name] = {
                "count": count,
                "avg_length": sum(map(len, chunks)) / count if count else 0.0,
                "chunks": chunks,
            }
        return comparison
