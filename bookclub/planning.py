"""Deterministic reading-schedule calculator used by the agent's planning tools.

A club meets `sessions_per_month` times a month. Between two meetings members read
for `interval` days at `pages_per_day`, which fixes how many pages each session can
cover. The book is split evenly across the resulting number of sessions.
"""

import math
from dataclasses import dataclass
from datetime import date, timedelta

DEFAULT_PAGES_PER_DAY = 10


@dataclass(frozen=True)
class PlannedSession:
    number: int
    date: date
    from_page: int
    to_page: int

    @property
    def pages(self) -> int:
        return self.to_page - self.from_page + 1


def interval_days(sessions_per_month: float) -> int:
    """Days between meetings, in whole weeks so sessions keep the same weekday (monthly=28, biweekly=14, weekly=7)."""
    if sessions_per_month <= 0:
        raise ValueError("sessions_per_month must be positive")
    return 7 * max(1, round(4 / sessions_per_month))


def plan_reading(
    total_pages: int,
    start_date: date,
    sessions_per_month: float,
    pages_per_day: int = DEFAULT_PAGES_PER_DAY,
    first_page: int = 1,
) -> list[PlannedSession]:
    """Split pages first_page..total_pages into sessions; reading starts on start_date.

    The first meeting is one interval after start_date, so members always get a full
    interval of reading before each session.
    """
    if total_pages < first_page:
        raise ValueError("first_page is beyond total_pages")
    if pages_per_day <= 0:
        raise ValueError("pages_per_day must be positive")

    interval = interval_days(sessions_per_month)
    remaining = total_pages - first_page + 1
    n_sessions = math.ceil(remaining / (pages_per_day * interval))

    sessions = []
    for i in range(n_sessions):
        from_page = first_page + round(i * remaining / n_sessions)
        to_page = first_page + round((i + 1) * remaining / n_sessions) - 1
        sessions.append(PlannedSession(i + 1, start_date + timedelta(days=interval * (i + 1)), from_page, to_page))
    return sessions


def replan_reading(
    total_pages: int,
    pages_read_to: int,
    today: date,
    sessions_per_month: float,
    pages_per_day: int = DEFAULT_PAGES_PER_DAY,
) -> list[PlannedSession]:
    """Re-plan the rest of a book when the group is behind (or ahead) of the original schedule."""
    if pages_read_to >= total_pages:
        return []
    return plan_reading(total_pages, today, sessions_per_month, pages_per_day, first_page=pages_read_to + 1)
