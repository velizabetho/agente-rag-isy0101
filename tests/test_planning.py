from datetime import date

import pytest

from bookclub.planning import interval_days, plan_reading, replan_reading


@pytest.mark.parametrize(("per_month", "days"), [(1, 28), (2, 14), (4, 7), (0.5, 56)])
def test_interval_keeps_weekday(per_month, days):
    assert interval_days(per_month) == days


def test_plan_covers_every_page_once_without_gaps():
    sessions = plan_reading(total_pages=453, start_date=date(2026, 11, 2), sessions_per_month=4)

    assert sessions[0].from_page == 1
    assert sessions[-1].to_page == 453
    for prev, nxt in zip(sessions, sessions[1:]):
        assert nxt.from_page == prev.to_page + 1
    # weekly club reading 10 pages/day -> at most 70 pages per session
    assert all(s.pages <= 70 for s in sessions)
    assert len(sessions) == 7


def test_first_session_leaves_a_full_interval_to_read():
    sessions = plan_reading(total_pages=120, start_date=date(2026, 11, 2), sessions_per_month=2)

    assert sessions[0].date == date(2026, 11, 16)
    assert [s.date.weekday() for s in sessions] == [date(2026, 11, 2).weekday()] * len(sessions)


def test_short_book_fits_one_monthly_session():
    sessions = plan_reading(total_pages=98, start_date=date(2026, 11, 2), sessions_per_month=1)

    assert len(sessions) == 1
    assert (sessions[0].from_page, sessions[0].to_page) == (1, 98)


def test_replan_starts_after_pages_already_read():
    sessions = replan_reading(total_pages=300, pages_read_to=120, today=date(2026, 12, 1), sessions_per_month=4)

    assert sessions[0].from_page == 121
    assert sessions[-1].to_page == 300
    assert sessions[0].date == date(2026, 12, 8)


def test_replan_when_book_is_finished_is_empty():
    assert replan_reading(total_pages=300, pages_read_to=300, today=date(2026, 12, 1), sessions_per_month=1) == []


def test_invalid_frequency_is_rejected():
    with pytest.raises(ValueError):
        plan_reading(total_pages=100, start_date=date(2026, 11, 2), sessions_per_month=0)
