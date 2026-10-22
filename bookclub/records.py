"""Business rules for the agent's write operations: reservations, reading plans and session logs.

Every function validates its input against the database and raises RecordError with a
message the agent can show to the user; nothing here trusts the LLM's arguments.
"""

import sqlite3
from datetime import date

from bookclub.planning import DEFAULT_PAGES_PER_DAY, PlannedSession, plan_reading, replan_reading

# The guidelines recommend groups of 8-12 people; clubs without a count get one copy per expected member.
DEFAULT_GROUP_SIZE = 12
MAX_LOAN_DAYS = 180


class RecordError(ValueError):
    """Invalid request; the message explains why in Spanish for the end user."""


def _club(conn: sqlite3.Connection, club_id: str) -> sqlite3.Row:
    club = conn.execute("SELECT * FROM clubs WHERE club_id = ?", (club_id,)).fetchone()
    if club is None:
        raise RecordError(f"No existe el club '{club_id}'.")
    return club


def _parse_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise RecordError(f"{field} debe tener formato AAAA-MM-DD (recibido: '{value}').") from exc


def availability(conn: sqlite3.Connection, catalog_title_id: int, start_date: str, end_date: str) -> dict:
    """Copies of a regional catalog title still free between two dates."""
    title = conn.execute("SELECT * FROM catalog_titles WHERE id = ?", (catalog_title_id,)).fetchone()
    if title is None:
        raise RecordError(f"No existe el título con id {catalog_title_id} en los catálogos regionales.")
    start, end = _parse_date(start_date, "start_date"), _parse_date(end_date, "end_date")
    if end <= start:
        raise RecordError("La fecha de devolución debe ser posterior a la de inicio.")
    reserved = conn.execute(
        "SELECT COALESCE(SUM(copies), 0) FROM reservations"
        " WHERE catalog_title_id = ? AND status = 'active' AND start_date < ? AND end_date > ?",
        (catalog_title_id, end.isoformat(), start.isoformat()),
    ).fetchone()[0]
    total = title["copies"] or 0
    return {
        "catalog_title_id": catalog_title_id,
        "title": title["title"],
        "author": title["author"],
        "region": title["region"],
        "pages": title["pages"],
        "total_copies": total,
        "reserved_copies": reserved,
        "available_copies": max(total - reserved, 0),
    }


def reserve_copies(
    conn: sqlite3.Connection, club_id: str, catalog_title_id: int, copies: int | None, start_date: str, end_date: str
) -> dict:
    club = _club(conn, club_id)
    status = availability(conn, catalog_title_id, start_date, end_date)
    if status["region"] != club["region"]:
        raise RecordError(
            f"'{status['title']}' pertenece al catálogo de {status['region']}, pero el club es de {club['region']}."
            " Solo se pueden reservar títulos del catálogo de la propia región."
        )
    if (_parse_date(end_date, "end_date") - _parse_date(start_date, "start_date")).days > MAX_LOAN_DAYS:
        raise RecordError(f"El préstamo no puede superar {MAX_LOAN_DAYS} días.")
    copies = copies or club["participants"] or DEFAULT_GROUP_SIZE
    if copies <= 0:
        raise RecordError("La cantidad de copias debe ser positiva.")
    if copies > status["available_copies"]:
        raise RecordError(
            f"Solo hay {status['available_copies']} copias disponibles de '{status['title']}' en esas fechas"
            f" (se pidieron {copies})."
        )
    cursor = conn.execute(
        "INSERT INTO reservations (club_id, catalog_title_id, copies, start_date, end_date) VALUES (?, ?, ?, ?, ?)",
        (club_id, catalog_title_id, copies, start_date, end_date),
    )
    return {"reservation_id": cursor.lastrowid, "copies": copies, **status,
            "available_copies": status["available_copies"] - copies}


def _active_plan(conn: sqlite3.Connection, club_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM reading_plans WHERE club_id = ? AND status = 'active' ORDER BY id DESC LIMIT 1", (club_id,)
    ).fetchone()


def _insert_sessions(conn: sqlite3.Connection, plan_id: int, sessions: list[PlannedSession], offset: int = 0) -> None:
    conn.executemany(
        "INSERT INTO plan_sessions (plan_id, session_number, date, from_page, to_page) VALUES (?, ?, ?, ?, ?)",
        [(plan_id, s.number + offset, s.date.isoformat(), s.from_page, s.to_page) for s in sessions],
    )


def _sessions_per_month(club: sqlite3.Row) -> float:
    if not club["sessions_per_month"]:
        raise RecordError(
            "El club no tiene una frecuencia de sesiones registrada; indica cada cuánto se reúne."
        )
    return club["sessions_per_month"]


def create_reading_plan(
    conn: sqlite3.Connection,
    club_id: str,
    book_title: str,
    total_pages: int,
    start_date: str,
    catalog_title_id: int | None = None,
    ebook_id: int | None = None,
    sessions_per_month: float | None = None,
    pages_per_day: int = DEFAULT_PAGES_PER_DAY,
) -> dict:
    club = _club(conn, club_id)
    if (current := _active_plan(conn, club_id)) is not None:
        raise RecordError(
            f"El club ya tiene un plan activo para '{current['book_title']}' (plan {current['id']})."
            " Finalízalo o reprograma ese plan antes de crear otro."
        )
    if total_pages <= 0:
        raise RecordError("El número de páginas debe ser positivo.")
    per_month = sessions_per_month or _sessions_per_month(club)
    sessions = plan_reading(total_pages, _parse_date(start_date, "start_date"), per_month, pages_per_day)
    plan_id = conn.execute(
        "INSERT INTO reading_plans (club_id, book_title, catalog_title_id, ebook_id, total_pages)"
        " VALUES (?, ?, ?, ?, ?)",
        (club_id, book_title, catalog_title_id, ebook_id, total_pages),
    ).lastrowid
    _insert_sessions(conn, plan_id, sessions)
    return {"plan_id": plan_id, "book_title": book_title, "sessions": [_session_dict(s) for s in sessions]}


def _session_dict(s: PlannedSession) -> dict:
    return {"session": s.number, "date": s.date.isoformat(), "from_page": s.from_page, "to_page": s.to_page}


def log_session(
    conn: sqlite3.Connection,
    club_id: str,
    session_date: str,
    summary: str,
    attendance: int | None = None,
    pages_read_to: int | None = None,
    agreements: str | None = None,
) -> dict:
    """Record a meeting. If the group read less than planned, report how far behind it is."""
    _club(conn, club_id)
    day = _parse_date(session_date, "session_date")
    plan = _active_plan(conn, club_id)
    conn.execute(
        "INSERT INTO session_logs (club_id, plan_id, date, attendance, pages_read_to, summary, agreements)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (club_id, plan["id"] if plan else None, day.isoformat(), attendance, pages_read_to, summary, agreements),
    )
    result: dict = {"logged": True, "plan_id": plan["id"] if plan else None}
    if plan is None or pages_read_to is None:
        return result

    conn.execute(
        "UPDATE plan_sessions SET status = 'done' WHERE plan_id = ? AND status = 'pending' AND to_page <= ?",
        (plan["id"], pages_read_to),
    )
    expected = conn.execute(
        "SELECT MAX(to_page) FROM plan_sessions WHERE plan_id = ? AND status != 'cancelled' AND date <= ?",
        (plan["id"], day.isoformat()),
    ).fetchone()[0] or 0
    result["expected_page"] = expected
    result["pages_behind"] = max(expected - pages_read_to, 0)
    if pages_read_to >= plan["total_pages"]:
        conn.execute("UPDATE reading_plans SET status = 'finished' WHERE id = ?", (plan["id"],))
        result["plan_finished"] = True
    return result


def replan(
    conn: sqlite3.Connection,
    club_id: str,
    pages_read_to: int,
    from_date: str,
    sessions_per_month: float | None = None,
    pages_per_day: int = DEFAULT_PAGES_PER_DAY,
) -> dict:
    """Cancel the pending sessions of the active plan and schedule the remaining pages again."""
    club = _club(conn, club_id)
    plan = _active_plan(conn, club_id)
    if plan is None:
        raise RecordError("El club no tiene un plan de lectura activo para reprogramar.")
    if not 0 <= pages_read_to <= plan["total_pages"]:
        raise RecordError(f"pages_read_to debe estar entre 0 y {plan['total_pages']}.")
    per_month = sessions_per_month or _sessions_per_month(club)
    sessions = replan_reading(plan["total_pages"], pages_read_to, _parse_date(from_date, "from_date"), per_month,
                              pages_per_day)
    conn.execute("UPDATE plan_sessions SET status = 'cancelled' WHERE plan_id = ? AND status = 'pending'", (plan["id"],))
    last_number = conn.execute(
        "SELECT COALESCE(MAX(session_number), 0) FROM plan_sessions WHERE plan_id = ?", (plan["id"],)
    ).fetchone()[0]
    _insert_sessions(conn, plan["id"], sessions, offset=last_number)
    return {"plan_id": plan["id"], "book_title": plan["book_title"],
            "sessions": [_session_dict(s) | {"session": s.number + last_number} for s in sessions]}


def club_status(conn: sqlite3.Connection, club_id: str) -> dict:
    club = dict(_club(conn, club_id))
    plan = _active_plan(conn, club_id)
    sessions = []
    if plan is not None:
        sessions = [dict(r) for r in conn.execute(
            "SELECT session_number, date, from_page, to_page, status FROM plan_sessions"
            " WHERE plan_id = ? AND status != 'cancelled' ORDER BY date", (plan["id"],))]
    reservations = [dict(r) for r in conn.execute(
        "SELECT r.id, t.title, r.copies, r.start_date, r.end_date FROM reservations r"
        " JOIN catalog_titles t ON t.id = r.catalog_title_id WHERE r.club_id = ? AND r.status = 'active'",
        (club_id,))]
    logs = [dict(r) for r in conn.execute(
        "SELECT date, attendance, pages_read_to, summary, agreements FROM session_logs"
        " WHERE club_id = ? ORDER BY date DESC LIMIT 3", (club_id,))]
    return {
        "club": club,
        "active_plan": dict(plan) | {"sessions": sessions} if plan else None,
        "active_reservations": reservations,
        "recent_sessions": logs,
    }
