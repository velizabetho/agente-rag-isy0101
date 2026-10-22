"""Exercise the real graph, tools and SQLite with a scripted chat model (no LLM or network calls)."""

import sqlite3

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from bookclub import db
from bookclub.agent.graph import TaskPlan, build_graph


class ScriptedChatModel(BaseChatModel):
    """Returns the scripted AI messages in order and records the prompts it received."""

    responses: list[AIMessage]
    prompts: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.prompts.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


class FixedPlanner:
    def __init__(self, steps):
        self.steps = steps

    def with_structured_output(self, schema):
        return RunnableLambda(lambda _: TaskPlan(steps=self.steps))


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    path = tmp_path / "agent.db"
    conn = sqlite3.connect(path)
    db.init_schema(conn)
    conn.execute(
        "INSERT INTO clubs (club_id, name, region, participants, sessions_per_month) VALUES (?, ?, ?, ?, ?)",
        ("club-test", "Club de prueba", "Metropolitana", 9, 4),
    )
    conn.execute(
        "INSERT INTO catalog_titles (id, region, title, author, pages, copies) VALUES (1, 'Metropolitana', ?, ?, ?, ?)",
        ("Sub terra", "Baldomero Lillo", 262, 10),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "db_path", lambda: path)
    return path


def test_agent_runs_write_tools_for_the_club_in_state_and_keeps_thread_memory(temp_db):
    model = ScriptedChatModel(responses=[
        AIMessage("", tool_calls=[
            {"id": "c1", "name": "reserve_copies",
             "args": {"catalog_title_id": 1, "start_date": "2026-11-02", "end_date": "2027-01-15"}},
            # the model tries to act on another club; the injected state must win
            {"id": "c2", "name": "create_reading_plan",
             "args": {"book_title": "Sub terra", "total_pages": 262, "start_date": "2026-11-02",
                      "catalog_title_id": 1, "club_id": "otro-club"}},
        ]),
        AIMessage("Listo: reservé 9 copias y armé el calendario."),
        AIMessage("Hablamos de Sub terra."),
    ])
    agent = build_graph(model, planner_llm=FixedPlanner(["get_club_status", "reserve_copies"]),
                        club_context_fn=lambda club_id: f"Club {club_id}", recall_fn=lambda c, q: ["(preference) cuentos"],
                        checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    state = agent.invoke({"messages": [("user", "Reserva Sub terra y arma el calendario")], "club_id": "club-test"},
                         config)

    assert [t["tool"] for t in state["trace"]] == ["reserve_copies", "create_reading_plan"]
    assert all(t["status"] == "success" for t in state["trace"])
    assert state["trace"][0]["artifact"]["copies"] == 9  # defaults to club participants
    with sqlite3.connect(temp_db) as conn:
        assert conn.execute("SELECT club_id FROM reading_plans").fetchall() == [("club-test",)]
        # weekly club, 10 pages/day -> 70 pages per session
        assert conn.execute("SELECT COUNT(*) FROM plan_sessions").fetchone()[0] == 4
    system_prompt = model.prompts[0][0].content
    assert "1. get_club_status" in system_prompt and "(preference) cuentos" in system_prompt

    second = agent.invoke({"messages": [("user", "¿De qué libro hablamos?")], "club_id": "club-test"}, config)

    assert second["trace"] == []  # per-turn trace was reset
    contents = [m.content for m in model.prompts[-1]]
    assert "Reserva Sub terra y arma el calendario" in contents  # thread history reached the model


def test_tool_errors_are_returned_to_the_model_not_raised(temp_db):
    model = ScriptedChatModel(responses=[
        AIMessage("", tool_calls=[{"id": "c1", "name": "reserve_copies",
                                   "args": {"catalog_title_id": 1, "copies": 50, "start_date": "2026-11-02",
                                            "end_date": "2026-12-01"}}]),
        AIMessage("No hay suficientes copias."),
    ])
    agent = build_graph(model, planner_llm=FixedPlanner([]))

    state = agent.invoke({"messages": [("user", "Reserva 50 copias")], "club_id": "club-test"})

    assert state["trace"][0]["content"].startswith("ERROR: Solo hay 10 copias")
    assert state["messages"][-1].content == "No hay suficientes copias."
