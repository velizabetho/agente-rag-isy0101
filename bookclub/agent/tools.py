"""Tools available to the agent.

Query tools: guidelines RAG, catalog text-to-SQL, availability, Open Library, Wikipedia, club status.
Write tools: reservations, reading plans, session logs, re-planning, long-term memory, session guides.

The current club is injected from the graph state, so the model never chooses which club it acts on.
Tools return (content, artifact): content goes to the LLM, artifact to the UI trace.
"""

import json
import re
from datetime import date
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from bookclub import external, memory, records
from bookclub.config import DATA_DIR
from bookclub.db import connect
from bookclub.rag.retriever import format_citation, search_guidelines
from bookclub.sql import SQLError, run_query

GUIDES_DIR = DATA_DIR / "guides"
ClubId = Annotated[str, InjectedState("club_id")]


def _json(data) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _run(action, *args, **kwargs) -> tuple[str, dict]:
    """Execute a records operation in a transaction and turn domain errors into tool feedback."""
    try:
        with connect() as conn:
            result = action(conn, *args, **kwargs)
        return _json(result), result
    except records.RecordError as exc:
        return f"ERROR: {exc}", {"error": str(exc)}


# ---------------------------------------------------------------- query tools

@tool(response_format="content_and_artifact")
def search_club_guidelines(query: str) -> tuple[str, list[dict]]:
    """Search the official SNBP guide "Orientaciones para mediar y gestionar clubes de lectura" (2025).

    Use it for methodology and rules: how to run a session, group size, session frequency and cycle length,
    first-session structure, open questions and dynamics, how regional catalogs and loans work, virtual clubs.
    Write the query in Spanish with the concrete topic. Returns passages with page citations.
    """
    chunks = search_guidelines(query)
    content = "\n\n".join(f"{format_citation(c)}\n{c['text']}" for c in chunks)
    return content, [{"page": c["page"], "section": c["section"], "score": round(c["score"], 3),
                      "text": c["text"]} for c in chunks]


@tool(response_format="content_and_artifact")
def query_database(sql: str) -> tuple[str, dict]:
    """Run ONE read-only SQLite SELECT over the book-club database. Returns at most 50 rows.

    Tables:
    - catalog_titles(id, region, title, author, publisher, isbn, year, pages, copies, genre): printed books of
      the 16 regional catalogs for book clubs; `copies` is the total owned by that region. Regions: 'Arica y
      Parinacota', 'Tarapacá', 'Antofagasta', 'Atacama', 'Coquimbo', 'Valparaíso', 'Metropolitana',
      "O'Higgins", 'Maule', 'Ñuble', 'Biobío', 'La Araucanía', 'Los Ríos', 'Los Lagos', 'Aysén', 'Magallanes'.
      genre values: 'Novela', 'Literatura infantil y juvenil', 'Cuento', 'Ensayo', 'Narrativa gráfica',
      'Ciencias sociales y humanidades', 'Biografía', 'Generalidades', 'Poesía', 'Ciencias básicas y aplicadas',
      'Crónica', 'Epistolar', 'Dramaturgia'.
    - ebooks(id, title, author, isbn, publisher, year, subjects, access, copies, url): Biblioteca Pública
      Digital titles for clubs; access 'free' (no limit) or 'licensed' (`copies` simultaneous loans).
    - clubs(club_id, name, region, library, participants, frequency_text, sessions_per_month,
      titles_read_2024): real SNBP book clubs.
    - reservations(id, club_id, catalog_title_id, copies, start_date, end_date, status)
    - reading_plans(id, club_id, book_title, catalog_title_id, ebook_id, total_pages, status)
    - plan_sessions(id, plan_id, session_number, date, from_page, to_page, status)

    Tips: compare text with LIKE and lower() (e.g. lower(title) LIKE '%sub terra%'); always filter
    catalog_titles by the club's region when the user wants books they can borrow; select only needed columns.
    """
    try:
        result = run_query(sql)
    except SQLError as exc:
        return f"ERROR: {exc}", {"sql": sql, "error": str(exc)}
    return _json(result), {"sql": sql, **result}


@tool(response_format="content_and_artifact")
def check_availability(catalog_title_id: int, start_date: str, end_date: str) -> tuple[str, dict]:
    """Copies of a regional catalog title (catalog_titles.id) still free for loan between two dates (YYYY-MM-DD),
    discounting active reservations of other clubs."""
    return _run(records.availability, catalog_title_id, start_date, end_date)


@tool(response_format="content_and_artifact")
def get_club_status(club_id: ClubId) -> tuple[str, dict]:
    """Profile of the current club (region, members, frequency) plus its active reading plan with sessions,
    active reservations and the latest session logs. Use it before planning, reserving or logging."""
    return _run(records.club_status, club_id)


@tool(response_format="content_and_artifact")
def search_book_info(title: str, author: str | None = None) -> tuple[str, list[dict]]:
    """External source (Open Library): synopsis, subjects, first publication year and typical page count of a
    book. Use it to present a book or to suggest discussion topics; it does NOT tell loan availability."""
    try:
        results = external.search_open_library(title, author)
    except Exception as exc:  # network errors must not break the conversation
        return f"ERROR: Open Library no respondió ({exc}).", []
    # Only the best match keeps its description; alternatives are listed briefly to save tokens.
    compact = results[:1] + [{k: r[k] for k in ("title", "authors", "first_publish_year")} for r in results[1:]]
    return (_json(compact) if results else "Sin resultados en Open Library."), results


@tool(response_format="content_and_artifact")
def search_author_context(query: str) -> tuple[str, dict]:
    """External source (Spanish Wikipedia): short biography or context about an author, work or historical
    period, with its URL. Use it to prepare the presentation of a book."""
    try:
        result = external.wikipedia_summary(query)
    except Exception as exc:
        return f"ERROR: Wikipedia no respondió ({exc}).", {}
    return (_json(result) if result else "Sin resultados en Wikipedia."), result or {}


# ---------------------------------------------------------------- write tools

@tool(response_format="content_and_artifact")
def reserve_copies(
    catalog_title_id: int, start_date: str, end_date: str, club_id: ClubId, copies: int | None = None
) -> tuple[str, dict]:
    """Reserve printed copies of a regional catalog title for the current club between start_date and end_date
    (YYYY-MM-DD). copies defaults to the club's number of participants. Only titles of the club's own region can
    be reserved and never more copies than available. Confirm the title and dates with the user first."""
    return _run(records.reserve_copies, club_id, catalog_title_id, copies, start_date, end_date)


@tool(response_format="content_and_artifact")
def create_reading_plan(
    book_title: str,
    total_pages: int,
    start_date: str,
    club_id: ClubId,
    catalog_title_id: int | None = None,
    ebook_id: int | None = None,
    sessions_per_month: float | None = None,
    pages_per_day: int = 10,
) -> tuple[str, dict]:
    """Create the reading calendar of a book for the current club: splits total_pages into sessions according to
    the club's meeting frequency (or sessions_per_month) and a reading pace of pages_per_day, starting to read on
    start_date (YYYY-MM-DD). Fails if the club already has an active plan."""
    return _run(records.create_reading_plan, club_id, book_title, total_pages, start_date, catalog_title_id,
                ebook_id, sessions_per_month, pages_per_day)


@tool(response_format="content_and_artifact")
def log_session(
    session_date: str,
    summary: str,
    club_id: ClubId,
    attendance: int | None = None,
    pages_read_to: int | None = None,
    agreements: str | None = None,
) -> tuple[str, dict]:
    """Record a meeting of the current club: date (YYYY-MM-DD), short summary, attendance, last page the group
    read and agreements. Returns pages_behind when the group is behind its reading plan."""
    return _run(records.log_session, club_id, session_date, summary, attendance, pages_read_to, agreements)


@tool(response_format="content_and_artifact")
def replan_reading(
    pages_read_to: int, from_date: str, club_id: ClubId, sessions_per_month: float | None = None,
    pages_per_day: int = 10,
) -> tuple[str, dict]:
    """Re-schedule the remaining pages of the club's active plan from from_date (YYYY-MM-DD), e.g. when the group
    is behind or the meeting frequency changes. Pending sessions are replaced."""
    return _run(records.replan, club_id, pages_read_to, from_date, sessions_per_month, pages_per_day)


@tool(response_format="content_and_artifact")
def cancel_reading_plan(club_id: ClubId) -> tuple[str, dict]:
    """Cancel the current club's active reading plan (e.g. the group changed the book). Ask the user first."""
    return _run(records.cancel_plan, club_id)


@tool(response_format="content_and_artifact")
def remember_club_fact(fact: str, kind: str, club_id: ClubId) -> tuple[str, dict]:
    """Save a durable fact about the current club in long-term memory so it is remembered in future
    conversations. kind: 'preference' (tastes, constraints), 'agreement' (group decisions), 'progress'
    or 'note'. Write the fact as a short, self-contained sentence in Spanish."""
    try:
        memory_id = memory.save_memory(club_id, fact, kind)
    except ValueError as exc:
        return f"ERROR: {exc}", {"error": str(exc)}
    return "Guardado en la memoria del club.", {"memory_id": memory_id, "fact": fact, "kind": kind}


@tool(response_format="content_and_artifact")
def save_session_guide(title: str, content_markdown: str, club_id: ClubId) -> tuple[str, dict]:
    """Save a session guide (Markdown) for the current club: presentation of the book, open questions,
    opening/closing dynamics and sources. Returns the file path."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "guia"
    folder = GUIDES_DIR / club_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{date.today().isoformat()}-{slug}.md"
    path.write_text(f"# {title}\n\n{content_markdown.strip()}\n", encoding="utf-8")
    return f"Guía guardada en {path.relative_to(DATA_DIR.parent)}", {"path": str(path)}


QUERY_TOOLS = [search_club_guidelines, query_database, check_availability, get_club_status, search_book_info,
               search_author_context]
WRITE_TOOLS = [reserve_copies, create_reading_plan, log_session, replan_reading, cancel_reading_plan,
               remember_club_fact, save_session_guide]
ALL_TOOLS = QUERY_TOOLS + WRITE_TOOLS
