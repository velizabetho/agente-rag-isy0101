"""Semantic search over the ingested guidelines."""

from bookclub.embeddings import get_embedder
from bookclub.mongo import GUIDELINES, get_collection, vector_search

DEFAULT_K = 5


def search_guidelines(query: str, k: int = DEFAULT_K) -> list[dict]:
    """Return the k most similar chunks with page, chapter, section, text and score."""
    vector = get_embedder().embed_query(query)
    return vector_search(get_collection(GUIDELINES), vector, k=k)


def format_citation(chunk: dict) -> str:
    return f"[Orientaciones SNBP 2025, p. {chunk['page']} — {chunk['section']}]"
