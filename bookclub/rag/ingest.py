"""Ingest the SNBP guidelines PDF into MongoDB Atlas (chunks + embeddings + vector index).

Chunks follow the PDF's own table of contents: each page is split at the section
headings that start on it, and long sections are split with overlap. Every chunk
keeps page, chapter and section so answers can cite them.

Run: python -m bookclub.rag.ingest
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from bookclub.config import RAW_DIR
from bookclub.data.sources import GUIDELINES_FILE
from bookclub.embeddings import get_embedder
from bookclub.mongo import GUIDELINES, ensure_vector_index, get_collection

SOURCE = "Orientaciones para mediar y gestionar clubes de lectura (SNBP, 2025)"
MAX_CHARS = 1200
OVERLAP = 200
MIN_CHARS = 80
# Front matter (cover, credits, index) ends before page 9; back matter (notes,
# bibliography) only holds references.
FIRST_CONTENT_PAGE = 9
LAST_CONTENT_PAGE = 63

_RUNNING_HEADER = "ORIENTACIONES PARA MEDIAR Y GESTIONAR CLUBES DE LECTURA"


@dataclass
class Chunk:
    page: int
    chapter: str
    section: str
    text: str


def clean_page_text(text: str, running_headers: frozenset[str] = frozenset()) -> str:
    text = text.replace("\u00ad\n", "").replace("\u00ad", "")  # soft hyphens at line ends
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.isdigit() or stripped in (_RUNNING_HEADER, "•") or stripped in running_headers:
            continue
        if stripped.startswith("Escanea este código QR"):
            continue
        lines.append(stripped)
    text = "\n".join(lines)
    text = re.sub(r"(?<![.:;?!»])\n(?=[a-záéíóúñ(«])", " ", text)  # re-join wrapped sentences
    return text


def split_long(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + max_chars // 2:
                end = cut + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        space = text.find(" ", start, end)
        if space != -1:
            start = space + 1  # do not start the overlap mid-word
    return parts


def _toc_by_page(doc: pymupdf.Document) -> list[tuple[int, int, str]]:
    return [(page, level, title.strip()) for level, title, page in doc.get_toc() if not title.startswith("_")]


def build_chunks(pdf_path: Path) -> list[Chunk]:
    doc = pymupdf.open(pdf_path)
    toc = _toc_by_page(doc)
    chapter_titles = {title for _, level, title in toc if level == 1}
    chapter, section = "Presentación", "Presentación"
    chunks: list[Chunk] = []

    for page_number in range(FIRST_CONTENT_PAGE, min(len(doc), LAST_CONTENT_PAGE) + 1):
        headings = [(level, title) for page, level, title in toc if page == page_number]
        # Chapter titles repeat as running headers on later pages; keep them only where the chapter starts.
        running = frozenset(chapter_titles - {title for level, title in headings if level == 1})
        text = clean_page_text(doc[page_number - 1].get_text(), running)

        located = sorted((text.find(title), level, title) for level, title in headings)
        # Headings not found in the page text (e.g. split across lines) are assumed to open the page.
        for pos, level, title in located:
            if pos == -1:
                chapter, section = (title, title) if level == 1 else (chapter, title)

        # Split the page at the headings that appear on it, in reading order.
        segments: list[tuple[str, str, str]] = []
        cursor = 0
        for pos, level, title in located:
            if pos == -1:
                continue
            if pos > cursor:
                segments.append((chapter, section, text[cursor:pos]))
            chapter, section = (title, title) if level == 1 else (chapter, title)
            cursor = pos
        segments.append((chapter, section, text[cursor:]))

        for seg_chapter, seg_section, seg_text in segments:
            for part in split_long(seg_text.strip()):
                if len(part) >= MIN_CHARS:
                    chunks.append(Chunk(page_number, seg_chapter, seg_section, part))
    return chunks


def chunk_document(number: int, chunk: Chunk, embedding: list[float]) -> dict:
    return {
        "_id": f"p{chunk.page:03d}-{number:04d}",
        "source": SOURCE,
        "page": chunk.page,
        "chapter": chunk.chapter,
        "section": chunk.section,
        "text": chunk.text,
        "embedding": embedding,
    }


def ingest(pdf_path: Path = RAW_DIR / GUIDELINES_FILE) -> int:
    chunks = build_chunks(pdf_path)
    embedder = get_embedder()
    # The heading context is embedded with the text to improve retrieval of short sections.
    vectors = embedder.embed_documents([f"{c.chapter}. {c.section}\n{c.text}" for c in chunks])

    collection = get_collection(GUIDELINES)
    collection.delete_many({})
    collection.insert_many(
        [chunk_document(n, c, v) for n, (c, v) in enumerate(zip(chunks, vectors, strict=True))]
    )
    ensure_vector_index(collection, embedder.dimensions, filter_fields=("page", "chapter"))
    return len(chunks)


if __name__ == "__main__":
    print(f"ingested {ingest()} chunks into '{GUIDELINES}'")
