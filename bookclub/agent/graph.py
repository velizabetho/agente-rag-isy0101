"""LangGraph agent, built explicitly with StateGraph.

    START -> prepare -> planner -> agent --(tool calls?)--> tools -> agent
                                       \\--> END

- prepare: resets the per-turn trace, loads the club profile and recalls long-term memories.
- planner: decomposes multi-step requests into an ordered task list (plan-and-execute).
- agent:   LLM with tools; follows the plan and the system rules.
- tools:   executes tool calls and records them (arguments + artifacts) for traceability.

Short-term memory: a checkpointer keyed by thread_id (MongoDB in the app, in-memory in tests).
"""

from collections.abc import Callable
from datetime import date
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field

from bookclub.agent.prompts import AGENT_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT
from bookclub.agent.tools import ALL_TOOLS

MAX_HISTORY_MESSAGES = 24


def _is_malformed_tool_call(exc: Exception) -> bool:
    """Groq rejects generations whose tool-call JSON is invalid ('tool_use_failed'); a new sample usually works."""
    return "tool_use_failed" in str(exc)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    club_id: str
    club_context: str
    memories: list[str]
    plan: list[str]
    trace: list[dict]  # tool calls of the current turn (reset by prepare)


class TaskPlan(BaseModel):
    steps: list[str] = Field(default_factory=list, description="Ordered steps; empty for simple requests")


def _last_human(messages: list[AnyMessage]) -> str:
    return next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")


def build_graph(
    llm: BaseChatModel,
    planner_llm: BaseChatModel | None = None,
    club_context_fn: Callable[[str], str] | None = None,
    recall_fn: Callable[[str, str], list[str]] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Assemble the agent. Context and memory providers are injected so tests can run without databases."""
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    planner = (planner_llm or llm).with_structured_output(TaskPlan)
    tool_node = ToolNode(ALL_TOOLS, handle_tool_errors=True)

    def prepare(state: AgentState) -> dict:
        club_id = state["club_id"]
        context = club_context_fn(club_id) if club_context_fn else club_id
        memories = recall_fn(club_id, _last_human(state["messages"])) if recall_fn else []
        return {"club_context": context, "memories": memories, "plan": [], "trace": []}

    def plan_step(state: AgentState) -> dict:
        system = PLANNER_SYSTEM_PROMPT.format(today=date.today().isoformat(), club_context=state["club_context"])
        result = planner.invoke([SystemMessage(system), HumanMessage(_last_human(state["messages"]))])
        return {"plan": result.steps if isinstance(result, TaskPlan) else []}

    def agent(state: AgentState) -> dict:
        system = AGENT_SYSTEM_PROMPT.format(
            today=date.today().isoformat(),
            club_context=state["club_context"],
            memories="\n".join(f"- {m}" for m in state["memories"]) or "(sin recuerdos relevantes)",
            plan="\n".join(f"{i}. {s}" for i, s in enumerate(state["plan"], 1)) or "(solicitud simple, sin plan)",
        )
        history = trim_messages(
            state["messages"], strategy="last", token_counter=len, max_tokens=MAX_HISTORY_MESSAGES,
            start_on="human", include_system=False,
        )
        response = llm_with_tools.invoke([SystemMessage(system), *history])
        return {"messages": [response]}

    def tools(state: AgentState) -> dict:
        last = state["messages"][-1]
        output = tool_node.invoke(state)
        tool_messages: list[ToolMessage] = output["messages"]
        by_id = {m.tool_call_id: m for m in tool_messages}
        trace = []
        for call in last.tool_calls:
            message = by_id.get(call["id"])
            trace.append({
                "tool": call["name"],
                "args": {k: v for k, v in call["args"].items() if k != "club_id"},
                "status": getattr(message, "status", "success"),
                "artifact": getattr(message, "artifact", None),
                "content": str(message.content)[:2000] if message else None,
            })
        return {"messages": tool_messages, "trace": state["trace"] + trace}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare)
    graph.add_node("planner", plan_step)
    graph.add_node("agent", agent, retry_policy=RetryPolicy(max_attempts=3, retry_on=_is_malformed_tool_call))
    graph.add_node("tools", tools)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "planner")
    graph.add_edge("planner", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
