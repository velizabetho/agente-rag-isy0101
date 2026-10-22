"""Prepare everything the app needs, from scratch and idempotently.

1. Download the public SNBP sources (guidelines PDF, 16 regional catalogs, BPDigital spreadsheet).
2. Build the SQLite database (catalogs, ebooks, clubs).
3. Ingest the guidelines into MongoDB Atlas (chunks, embeddings, vector index).
4. Create the vector index for the clubs' long-term memory.

Run: python -m bookclub.setup
"""

from bookclub.data.build_db import build
from bookclub.data.sources import download_all
from bookclub.memory import ensure_memory_index
from bookclub.rag.ingest import ingest


def main() -> None:
    print("1/4 downloading sources")
    download_all()
    print("2/4 building SQLite database")
    for table, count in build().items():
        print(f"    {table}: {count}")
    print("3/4 ingesting guidelines into Atlas")
    print(f"    chunks: {ingest()}")
    print("4/4 memory index")
    ensure_memory_index()
    print("done")


if __name__ == "__main__":
    main()
