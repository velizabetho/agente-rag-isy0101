"""Build the SQLite database from the downloaded sources and the clubs CSV.

The database is rebuilt from scratch: reference tables (catalogs, ebooks, clubs)
are reloaded and the agent's records (reservations, plans, session logs) are cleared.

Run: python -m bookclub.data.build_db
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

from bookclub.config import DATA_DIR, RAW_DIR
from bookclub.data.catalog_parser import parse_all
from bookclub.data.sources import BPDIGITAL_FILE
from bookclub.db import connect, init_schema

CLUBS_CSV = DATA_DIR / "clubs_enriched.csv"
AGENT_TABLES = ("session_logs", "plan_sessions", "reading_plans", "reservations")
REFERENCE_TABLES = ("catalog_titles", "ebooks", "clubs")
# A few BPDigital rows carry obviously corrupted copy counts (e.g. 19 212 362).
MAX_PLAUSIBLE_COPIES = 10_000


def _clean(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip().rstrip(" /,")
    return text or None


def _int(value) -> int | None:
    match = re.search(r"\d+", str(value)) if _clean(value) else None
    return int(match.group()) if match else None


def _isbn(value) -> str | None:
    digits = re.sub(r"[^0-9Xx]", "", str(value)) if _clean(value) else ""
    return digits.upper() if len(digits) in (10, 13) else None


def load_catalog(conn: sqlite3.Connection, raw_dir: Path) -> int:
    entries = parse_all(raw_dir)
    conn.executemany(
        "INSERT INTO catalog_titles (region, title, author, publisher, isbn, year, pages, copies, genre, source_page)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (e.region, e.title, e.author, e.publisher, e.isbn, e.year, e.pages, e.copies, e.genre, e.source_page)
            for e in entries
        ],
    )
    return len(entries)


def load_ebooks(conn: sqlite3.Connection, xlsx: Path) -> int:
    free = pd.read_excel(xlsx, sheet_name="DESCARGA LIBRE", header=1)
    licensed = pd.read_excel(xlsx, sheet_name="LICENCIAS", header=1)
    rows = []
    for r in free.itertuples(index=False):
        rows.append(
            (_clean(r.Título), _clean(r.Autor), _isbn(r.ISBN), _clean(r.Editor), _int(r[4]), _clean(r.Materia),
             "free", None, _clean(r.Link))
        )
    for r in licensed.itertuples(index=False):
        publisher = _clean(r.Editorial)
        year = None
        if publisher and (m := re.search(r"\s(\d{4})$", publisher)):
            publisher, year = publisher[: m.start()].strip(), int(m.group(1))
        copies = _int(r.Cantidad)
        rows.append(
            (_clean(r.Título), _clean(r.Autor), _isbn(r.ISBN), publisher, year, _clean(r.Materia), "licensed",
             copies if copies is not None and copies <= MAX_PLAUSIBLE_COPIES else None, _clean(r.Link))
        )
    rows = [row for row in rows if row[0]]
    conn.executemany(
        "INSERT INTO ebooks (title, author, isbn, publisher, year, subjects, access, copies, url)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def load_clubs(conn: sqlite3.Connection, csv_path: Path) -> int:
    df = pd.read_csv(csv_path)
    rows = [
        (
            r.club_id, _clean(r.club), _clean(r.region), _clean(r.biblioteca_asociada), _int(r.participantes),
            _clean(r.frecuencia_texto) or _clean(r.frecuencia),
            None if pd.isna(r.sesiones_por_mes) else float(r.sesiones_por_mes),
            _int(r.titulos_leidos_2024), _clean(r.url_ficha),
        )
        for r in df.itertuples(index=False)
    ]
    conn.executemany(
        "INSERT INTO clubs (club_id, name, region, library, participants, frequency_text, sessions_per_month,"
        " titles_read_2024, url) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def build(raw_dir: Path = RAW_DIR, clubs_csv: Path = CLUBS_CSV) -> dict[str, int]:
    with connect() as conn:
        init_schema(conn)
        for table in AGENT_TABLES + REFERENCE_TABLES:
            conn.execute(f"DELETE FROM {table}")
        return {
            "catalog_titles": load_catalog(conn, raw_dir),
            "ebooks": load_ebooks(conn, raw_dir / BPDIGITAL_FILE),
            "clubs": load_clubs(conn, clubs_csv),
        }


if __name__ == "__main__":
    for table, count in build().items():
        print(f"{table:15} {count}")
