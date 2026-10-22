"""Production wiring of the agent: Groq LLM, SQLite club profile, Atlas memory and Mongo checkpointer."""

from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.mongodb import MongoDBSaver

from bookclub import memory
from bookclub.agent.graph import build_graph
from bookclub.config import get_settings
from bookclub.db import connect
from bookclub.llm import get_llm, get_planner_llm
from bookclub.mongo import get_client

RECURSION_LIMIT = 30


def club_context(club_id: str) -> str:
    with connect() as conn:
        club = conn.execute("SELECT * FROM clubs WHERE club_id = ?", (club_id,)).fetchone()
    if club is None:
        return f"club_id={club_id} (no registrado)"
    return (
        f"{club['name']} — región {club['region']}, biblioteca: {club['library'] or 'sin dato'}; "
        f"participantes: {club['participants'] or 'sin dato'}; frecuencia: {club['frequency_text'] or 'sin dato'} "
        f"(sesiones por mes: {club['sessions_per_month'] or 'sin dato'}); club_id={club_id}"
    )


def recall_memories(club_id: str, query: str) -> list[str]:
    return [f"({m['kind']}) {m['text']}" for m in memory.recall(club_id, query)] if query else []


@lru_cache
def get_agent():
    checkpointer = MongoDBSaver(get_client(), db_name=get_settings().mongodb_db)
    return build_graph(get_llm(), planner_llm=get_planner_llm(), club_context_fn=club_context,
                       recall_fn=recall_memories, checkpointer=checkpointer)


@dataclass
class AgentReply:
    answer: str
    plan: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    memories: list[str] = field(default_factory=list)


def ask(club_id: str, thread_id: str, message: str, agent=None, metadata: dict | None = None) -> AgentReply:
    agent = agent or get_agent()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT,
              "metadata": {"club_id": club_id, **(metadata or {})}}
    state = agent.invoke({"messages": [HumanMessage(message)], "club_id": club_id}, config)
    last = state["messages"][-1]
    answer = last.content if isinstance(last, AIMessage) else ""
    return AgentReply(answer, state.get("plan", []), state.get("trace", []), state.get("memories", []))
