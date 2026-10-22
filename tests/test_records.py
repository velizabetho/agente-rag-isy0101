import sqlite3

import pytest

from bookclub import records
from bookclub.db import init_schema
from bookclub.records import RecordError
from bookclub.sql import SQLError, run_query


@pytest.fixture
def db_file(tmp_path):
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    conn.executemany(
        "INSERT INTO catalog_titles (id, region, title, author, pages, copies) VALUES (?, ?, ?, ?, ?, ?)",
        [(1, "Metropolitana", "Sub terra", "Baldomero Lillo", 262, 10),
         (2, "Valparaíso", "Martín Rivas", "Alberto Blest Gana", 252, 11)],
    )
    conn.executemany(
        "INSERT INTO clubs (club_id, name, region, participants, sessions_per_month) VALUES (?, ?, ?, ?, ?)",
        [("club-rm", "Club RM", "Metropolitana", 8, 2), ("club-rm-2", "Otro club RM", "Metropolitana", 5, 1)],
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def conn(db_file):
    connection = sqlite3.connect(db_file)
    connection.row_factory = sqlite3.Row
    yield connection
    connection.close()


def test_reservation_defaults_to_one_copy_per_member(conn):
    result = records.reserve_copies(conn, "club-rm", 1, None, "2026-11-01", "2026-12-15")

    assert result["copies"] == 8
    assert result["available_copies"] == 2


def test_overlapping_reservations_cannot_exceed_copies(conn):
    records.reserve_copies(conn, "club-rm", 1, 8, "2026-11-01", "2026-12-15")

    with pytest.raises(RecordError, match="Solo hay 2 copias"):
        records.reserve_copies(conn, "club-rm-2", 1, 5, "2026-12-01", "2027-01-15")
    # after the first loan is returned the copies are free again
    assert records.availability(conn, 1, "2026-12-15", "2027-01-31")["available_copies"] == 10


def test_cannot_reserve_from_another_region(conn):
    with pytest.raises(RecordError, match="catálogo de Valparaíso"):
        records.reserve_copies(conn, "club-rm", 2, 5, "2026-11-01", "2026-12-01")


def test_only_one_active_plan_per_club(conn):
    records.create_reading_plan(conn, "club-rm", "Sub terra", 262, "2026-11-02", catalog_title_id=1)

    with pytest.raises(RecordError, match="ya tiene un plan activo"):
        records.create_reading_plan(conn, "club-rm", "Otro", 100, "2026-11-02")


def test_log_detects_group_behind_schedule_and_replan_moves_pages(conn):
    plan = records.create_reading_plan(conn, "club-rm", "Sub terra", 262, "2026-11-02", catalog_title_id=1)
    first = plan["sessions"][0]

    logged = records.log_session(conn, "club-rm", first["date"], "Conversamos los primeros cuentos", 7, 60)

    assert logged["expected_page"] == first["to_page"]
    assert logged["pages_behind"] == first["to_page"] - 60

    replanned = records.replan(conn, "club-rm", 60, first["date"])
    assert replanned["sessions"][0]["from_page"] == 61
    assert replanned["sessions"][-1]["to_page"] == 262
    status = records.club_status(conn, "club-rm")
    pending = [s for s in status["active_plan"]["sessions"] if s["status"] == "pending"]
    assert pending[0]["from_page"] == 61


def test_logging_last_page_finishes_plan(conn):
    records.create_reading_plan(conn, "club-rm", "Sub terra", 262, "2026-11-02")

    result = records.log_session(conn, "club-rm", "2026-12-28", "Cerramos el libro", pages_read_to=262)

    assert result["plan_finished"] is True
    assert records.club_status(conn, "club-rm")["active_plan"] is None


def test_sql_tool_reads_allowed_tables(db_file):
    result = run_query("SELECT title, copies FROM catalog_titles WHERE region = 'Metropolitana'", db_file)

    assert result["rows"] == [{"title": "Sub terra", "copies": 10}]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM catalog_titles",
        "SELECT 1; DROP TABLE clubs",
        "SELECT * FROM session_logs",
        "SELECT name FROM sqlite_master",
    ],
)
def test_sql_tool_rejects_writes_and_hidden_tables(db_file, sql):
    with pytest.raises(SQLError):
        run_query(sql, db_file)
