"""Long-term memory per club: facts the agent should remember across conversations.

Stored in MongoDB Atlas with embeddings and recalled by semantic similarity to the
current request, filtered by club.
"""

from datetime import UTC, datetime

from bookclub.embeddings import get_embedder
from bookclub.mongo import MEMORIES, ensure_vector_index, get_collection, vector_search

MEMORY_KINDS = ("preference", "agreement", "progress", "note")


def save_memory(club_id: str, text: str, kind: str = "note") -> str:
    if kind not in MEMORY_KINDS:
        raise ValueError(f"kind must be one of {MEMORY_KINDS}")
    document = {
        "club_id": club_id,
        "kind": kind,
        "text": text.strip(),
        "created_at": datetime.now(UTC),
        "embedding": get_embedder().embed_documents([text])[0],
    }
    return str(get_collection(MEMORIES).insert_one(document).inserted_id)


def recall(club_id: str, query: str, k: int = 5, min_score: float = 0.8) -> list[dict]:
    """Memories of this club most related to the query (above a similarity threshold)."""
    vector = get_embedder().embed_query(query)
    hits = vector_search(get_collection(MEMORIES), vector, k=k, filters={"club_id": club_id})
    return [h for h in hits if h["score"] >= min_score]


def recent_memories(club_id: str, limit: int = 20) -> list[dict]:
    cursor = get_collection(MEMORIES).find({"club_id": club_id}, {"embedding": 0}).sort("created_at", -1).limit(limit)
    return list(cursor)


def ensure_memory_index() -> None:
    ensure_vector_index(get_collection(MEMORIES), get_embedder().dimensions, filter_fields=("club_id",))
