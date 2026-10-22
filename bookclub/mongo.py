"""MongoDB Atlas client, collections and vector search indexes."""

import time
from functools import lru_cache

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.operations import SearchIndexModel

from bookclub.config import get_settings

GUIDELINES = "guideline_chunks"
MEMORIES = "club_memories"
VECTOR_INDEX = "vector_index"


@lru_cache
def get_client() -> MongoClient:
    settings = get_settings()
    if not settings.mongodb_uri:
        raise RuntimeError("MONGODB_URI is not set (see .env.example)")
    return MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=20000, appname="bookclub")


def get_db() -> Database:
    return get_client()[get_settings().mongodb_db]


def get_collection(name: str) -> Collection:
    return get_db()[name]


def ensure_vector_index(collection: Collection, dimensions: int, filter_fields: tuple[str, ...] = ()) -> None:
    """Create the Atlas vector index if missing (or recreate it if dimensions changed) and wait until queryable."""
    if collection.name not in collection.database.list_collection_names():
        collection.database.create_collection(collection.name)

    fields = [{"type": "vector", "path": "embedding", "numDimensions": dimensions, "similarity": "cosine"}]
    fields += [{"type": "filter", "path": f} for f in filter_fields]

    existing = {idx["name"]: idx for idx in collection.list_search_indexes()}
    current = existing.get(VECTOR_INDEX)
    if current is not None:
        current_fields = current.get("latestDefinition", {}).get("fields", [])
        if current_fields == fields:
            _wait_queryable(collection)
            return
        collection.drop_search_index(VECTOR_INDEX)
        while VECTOR_INDEX in {i["name"] for i in collection.list_search_indexes()}:
            time.sleep(2)

    collection.create_search_index(
        SearchIndexModel(definition={"fields": fields}, name=VECTOR_INDEX, type="vectorSearch")
    )
    _wait_queryable(collection)


def _wait_queryable(collection: Collection, timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for idx in collection.list_search_indexes(VECTOR_INDEX):
            if idx.get("queryable"):
                return
        time.sleep(3)
    raise TimeoutError(f"Vector index on {collection.name} not queryable after {timeout_s}s")


def vector_search(
    collection: Collection, query_vector: list[float], k: int, filters: dict | None = None, num_candidates: int = 100
) -> list[dict]:
    stage: dict = {
        "index": VECTOR_INDEX,
        "path": "embedding",
        "queryVector": query_vector,
        "numCandidates": max(num_candidates, k * 10),
        "limit": k,
    }
    if filters:
        stage["filter"] = filters
    pipeline = [
        {"$vectorSearch": stage},
        {"$project": {"embedding": 0}},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
    ]
    return list(collection.aggregate(pipeline))
