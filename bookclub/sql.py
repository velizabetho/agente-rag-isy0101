"""Read-only execution of LLM-written SQL (text-to-SQL) against the SQLite database.

Defense in depth: the connection is opened read-only, a single SELECT/WITH statement is
accepted, and an authorizer denies access to any table outside QUERYABLE_TABLES.
"""

import re
import sqlite3
from pathlib import Path

from bookclub.db import QUERYABLE_TABLES, connect_readonly

MAX_ROWS = 50


class SQLError(ValueError):
    pass


def _authorizer(action: int, arg1: str | None, arg2: str | None, db_name: str | None, trigger: str | None) -> int:
    if action == sqlite3.SQLITE_SELECT or action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        # arg1 is the table being read; it is None for columns of CTEs and subqueries.
        return sqlite3.SQLITE_OK if arg1 in QUERYABLE_TABLES or arg1 is None else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_RECURSIVE:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def run_query(sql: str, path: Path | None = None) -> dict:
    statement = sql.strip().rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", statement):
        raise SQLError("Solo se permiten consultas SELECT.")
    if ";" in statement:
        raise SQLError("Solo se permite una consulta por llamada.")
    with connect_readonly(path) as conn:
        conn.set_authorizer(_authorizer)
        try:
            cursor = conn.execute(statement)
        except sqlite3.DatabaseError as exc:
            raise SQLError(f"Error SQL: {exc}") from exc
        rows = cursor.fetchmany(MAX_ROWS + 1)
    truncated = len(rows) > MAX_ROWS
    return {"rows": [dict(r) for r in rows[:MAX_ROWS]], "truncated": truncated}
