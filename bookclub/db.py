"""SQLite database: catalogs and clubs (read-only reference data) plus the agent's records."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from bookclub.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS catalog_titles (
    id INTEGER PRIMARY KEY,
    region TEXT NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    publisher TEXT,
    isbn TEXT,
    year INTEGER,
    pages INTEGER,
    copies INTEGER,
    genre TEXT,
    source_page INTEGER
);
CREATE INDEX IF NOT EXISTS idx_catalog_region ON catalog_titles(region);

CREATE TABLE IF NOT EXISTS ebooks (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT,
    isbn TEXT,
    publisher TEXT,
    year INTEGER,
    subjects TEXT,
    access TEXT NOT NULL CHECK (access IN ('free', 'licensed')),
    copies INTEGER,
    url TEXT
);

CREATE TABLE IF NOT EXISTS clubs (
    club_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL,
    library TEXT,
    participants INTEGER,
    frequency_text TEXT,
    sessions_per_month REAL,
    titles_read_2024 INTEGER,
    url TEXT
);

CREATE TABLE IF NOT EXISTS reservations (
    id INTEGER PRIMARY KEY,
    club_id TEXT NOT NULL REFERENCES clubs(club_id),
    catalog_title_id INTEGER NOT NULL REFERENCES catalog_titles(id),
    copies INTEGER NOT NULL CHECK (copies > 0),
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'returned', 'cancelled')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS reading_plans (
    id INTEGER PRIMARY KEY,
    club_id TEXT NOT NULL REFERENCES clubs(club_id),
    book_title TEXT NOT NULL,
    catalog_title_id INTEGER REFERENCES catalog_titles(id),
    ebook_id INTEGER REFERENCES ebooks(id),
    total_pages INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'finished', 'cancelled')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS plan_sessions (
    id INTEGER PRIMARY KEY,
    plan_id INTEGER NOT NULL REFERENCES reading_plans(id),
    session_number INTEGER NOT NULL,
    date TEXT NOT NULL,
    from_page INTEGER NOT NULL,
    to_page INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'done', 'cancelled'))
);

CREATE TABLE IF NOT EXISTS session_logs (
    id INTEGER PRIMARY KEY,
    club_id TEXT NOT NULL REFERENCES clubs(club_id),
    plan_id INTEGER REFERENCES reading_plans(id),
    date TEXT NOT NULL,
    attendance INTEGER,
    pages_read_to INTEGER,
    summary TEXT NOT NULL,
    agreements TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# Tables the text-to-SQL tool may read.
QUERYABLE_TABLES = ("catalog_titles", "ebooks", "clubs", "reservations", "reading_plans", "plan_sessions")


def db_path() -> Path:
    return get_settings().sqlite_path


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path or db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


@contextmanager
def connect_readonly(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(f"file:{(path or db_path()).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def schema_description(conn: sqlite3.Connection) -> str:
    """CREATE statements of the queryable tables, used in the text-to-SQL prompt."""
    rows = conn.execute(
        f"SELECT sql FROM sqlite_master WHERE type = 'table' AND name IN ({','.join('?' * len(QUERYABLE_TABLES))})",
        QUERYABLE_TABLES,
    ).fetchall()
    return "\n\n".join(row["sql"] for row in rows)
