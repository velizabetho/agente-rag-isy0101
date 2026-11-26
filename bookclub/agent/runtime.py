"""Production wiring of the agent: Groq LLM, guardrails, SQLite club profile, Atlas memory, Mongo checkpointer
and per-turn telemetry (JSONL log + OpenTelemetry)."""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.mongodb import MongoDBSaver

from bookclub import guardrails, memory
from bookclub.agent.graph import build_graph
from bookclub.config import get_settings
from bookclub.db import connect
from bookclub.embeddings import get_embedder
from bookclub.llm import get_llm, get_planner_llm
from bookclub.mongo import get_client
from bookclub.observability import otel, store
from bookclub.observability.recorder import observe_turn

RECURSION_LIMIT = 30
LOGGED_INPUT_CHARS = 500
LOGGED_ANSWER_CHARS = 1500


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


def build_agent(checkpointer=None, llm: BaseChatModel | None = None):
    """Production agent. Without a checkpointer the caller provides persistence (LangGraph Studio / `langgraph dev`)."""
    return build_graph(llm or get_llm(), planner_llm=get_planner_llm(), club_context_fn=club_context,
                       recall_fn=recall_memories, checkpointer=checkpointer,
                       input_guard=guardrails.check_input, output_guard=guardrails.check_output)


@lru_cache
def get_agent():
    return build_agent(MongoDBSaver(get_client(), db_name=get_settings().mongodb_db))


def warm_up() -> None:
    """Load the embedding model before the first turn; otherwise the first user pays its load time (cold start)."""
    get_embedder()


@dataclass
class AgentReply:
    answer: str
    plan: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    memories: list[str] = field(default_factory=list)
    guardrail: dict = field(default_factory=dict)
    telemetry: dict = field(default_factory=dict)


def ask(club_id: str, thread_id: str, message: str, agent=None, metadata: dict | None = None,
        log_path: Path | None = store.APP_LOG, log_fields: dict | None = None) -> AgentReply:
    """Run one turn and log its telemetry (log_path=None skips the JSONL; extra log_fields are added to the record).

    Errors are logged and re-raised."""
    agent = agent or get_agent()
    settings = get_settings()
    meta = {"source": "app", "club_id": club_id, "thread_id": thread_id, "agent_model": settings.llm_model,
            "planner_model": settings.planner_model, **(log_fields or {})}
    with observe_turn(**meta) as recorder:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT,
                  "metadata": {"club_id": club_id, **(metadata or {})}, "callbacks": [recorder]}
        masked_input = guardrails.mask_pii(message)[0][:LOGGED_INPUT_CHARS]
        try:
            state = agent.invoke({"messages": [HumanMessage(message)], "club_id": club_id}, config)
        except Exception as exc:
            record = recorder.finish(error=exc, input=masked_input, answer="")
            _publish(record, recorder.start_ns, log_path)
            exc.telemetry = record  # lets batch runners keep the failed turn
            raise
    last = state["messages"][-1]
    answer = last.content if isinstance(last, AIMessage) else ""
    reply = AgentReply(answer, state.get("plan", []), state.get("trace", []), state.get("memories", []),
                       state.get("guardrail", {}))
    status = "blocked" if reply.guardrail.get("allowed") is False else "ok"
    reply.telemetry = recorder.finish(
        status, input=masked_input, answer=answer[:LOGGED_ANSWER_CHARS], plan_steps=len(reply.plan),
        tools_called=sorted({step["tool"] for step in reply.trace}), memories_recalled=len(reply.memories),
        guardrail=reply.guardrail,
    )
    _publish(reply.telemetry, recorder.start_ns, log_path)
    return reply


def _publish(record: dict, start_ns: int, log_path: Path | None) -> None:
    if log_path is not None:
        store.write_turn(record, log_path)
    otel.export_turn(record, start_ns)
