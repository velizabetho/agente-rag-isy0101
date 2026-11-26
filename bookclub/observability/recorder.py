"""Per-turn telemetry: every agent turn becomes one structured record with its spans, tokens, errors and resources.

Sources of data:
- LangChain/LangGraph callbacks (TurnRecorder): graph nodes, LLM calls (model, tokens) and tool calls.
- `span()` context manager: sub-steps that are not LangChain runnables (embeddings, vector search, HTTP to
  Open Library / Wikipedia, Prompt Guard).
- Logs of the Groq and httpx clients: HTTP 429 responses and automatic retries (time spent waiting).
- psutil: CPU time and resident memory of the process.

The current recorder lives in a context variable, so helpers deep in the call stack can add spans without
receiving it as a parameter; LangGraph copies the context into the threads that run tools.
"""

import logging
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import psutil
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langchain_core.outputs import LLMResult

# USD per 1M tokens (input, output). Groq on-demand prices: https://groq.com/pricing
PRICES_PER_M = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}
GRAPH_NODES = {"guard_input", "prepare", "planner", "agent", "tools", "guard_output"}

_current: ContextVar["TurnRecorder | None"] = ContextVar("turn_recorder", default=None)
_process = psutil.Process()


def _short_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def classify_error(text: str) -> str:
    """Group raw error messages into the categories used by the dashboard."""
    lowered = text.lower()
    if "tool_use_failed" in lowered:
        return "tool_use_failed"
    if "request too large" in lowered or "413" in lowered:
        return "request_too_large"
    if "rate limit" in lowered or "429" in lowered or "ratelimit" in lowered:
        return "rate_limit"
    if "recursion" in lowered:
        return "recursion_limit"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    return "other"


class TurnRecorder(BaseCallbackHandler):
    """Collects the spans of one agent turn. Use it with `observe_turn()`."""

    raise_error = False

    def __init__(self, **meta: Any):
        self.turn_id = str(uuid.uuid4())
        self.meta = meta
        self.spans: list[dict] = []
        self.events: list[dict] = []
        self._open: dict[UUID, dict] = {}
        self._root: UUID | None = None
        self._lock = threading.Lock()
        self.started_at = datetime.now(UTC)
        self.start_ns = time.time_ns()
        self._t0 = time.perf_counter()
        self._cpu0 = sum(_process.cpu_times()[:2])
        self._rss0 = _process.memory_info().rss

    # ------------------------------------------------------------ helpers
    def _ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def _start(self, run_id: UUID, **fields) -> None:
        with self._lock:
            self._open[run_id] = {"start_ms": round(self._ms(), 1), "status": "ok", **fields}

    def _end(self, run_id: UUID, exc: BaseException | None = None, **fields) -> dict | None:
        with self._lock:
            entry = self._open.pop(run_id, None)
            if entry is None:
                return None
            entry["duration_ms"] = round(self._ms() - entry["start_ms"], 1)
            entry.update(fields)
            if exc is not None:
                entry["status"] = "error"
                entry["error"] = _short_error(exc)
                entry["error_type"] = classify_error(entry["error"])
            self.spans.append(entry)
            return entry

    def add_event(self, kind: str, **fields) -> None:
        with self._lock:
            self.events.append({"type": kind, "at_ms": round(self._ms(), 1), **fields})

    # ------------------------------------------------------------ graph nodes
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, **kwargs):
        if parent_run_id is None and self._root is None:
            self._root = run_id
            return
        name = kwargs.get("name") or ""
        if parent_run_id == self._root and name in GRAPH_NODES:
            self._start(run_id, kind="node", name=name, node=name)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._end(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error)

    # ------------------------------------------------------------ LLM calls
    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        metadata = metadata or {}
        params = kwargs.get("invocation_params") or {}
        model = metadata.get("ls_model_name") or params.get("model") or params.get("model_name") or "unknown"
        self._start(run_id, kind="llm", name=f"llm:{metadata.get('langgraph_node', '?')}",
                    node=metadata.get("langgraph_node"), model=model,
                    input_messages=sum(len(batch) for batch in messages))

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs):
        usage: dict = {}
        try:
            usage = response.generations[0][0].message.usage_metadata or {}
        except (IndexError, AttributeError):
            pass
        raw = (response.llm_output or {}).get("token_usage") or {}
        if not usage:
            usage = {"input_tokens": raw.get("prompt_tokens", 0), "output_tokens": raw.get("completion_tokens", 0)}
        details = usage.get("output_token_details") or {}
        cached = ((usage.get("input_token_details") or {}).get("cache_read")
                  or (raw.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        # Groq reports how long the request took on its side; the rest of the span is network and retry waits.
        server_s = (raw.get("total_time") or 0) + (raw.get("queue_time") or 0)
        self._end(run_id, input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                  reasoning_tokens=details.get("reasoning", 0), cached_tokens=cached,
                  server_ms=round(server_s * 1000, 1))

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error)

    # ------------------------------------------------------------ tools
    def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name") or "tool"
        self._start(run_id, kind="tool", name=name, node=(metadata or {}).get("langgraph_node"))

    def on_tool_end(self, output, *, run_id, **kwargs):
        content = output.content if isinstance(output, ToolMessage) else output
        text = str(content)
        if text.startswith("ERROR"):  # domain error returned to the model (no stock, other region, ...)
            self._end(run_id, status="error", error=text[:300], error_type="tool_domain_error")
        else:
            self._end(run_id, output_chars=len(text))

    def on_tool_error(self, error, *, run_id, **kwargs):
        entry = self._end(run_id, error)
        if entry:
            entry["error_type"] = "tool_exception"

    # ------------------------------------------------------------ result
    def finish(self, status: str = "ok", error: BaseException | None = None, **fields) -> dict:
        llm = [s for s in self.spans if s["kind"] == "llm"]
        tokens: dict[str, dict] = {}
        for s in llm:
            t = tokens.setdefault(s["model"], {"calls": 0, "input": 0, "cached": 0, "output": 0, "reasoning": 0})
            t["calls"] += 1
            t["input"] += s.get("input_tokens", 0)
            t["cached"] += s.get("cached_tokens", 0)
            t["output"] += s.get("output_tokens", 0)
            t["reasoning"] += s.get("reasoning_tokens", 0)
        # Cached input tokens cost 50% and do not count towards Groq's rate limits.
        cost = sum((t["input"] - t["cached"] / 2) * PRICES_PER_M.get(m, (0, 0))[0]
                   + t["output"] * PRICES_PER_M.get(m, (0, 0))[1] for m, t in tokens.items()) / 1_000_000
        # Root-cause errors only: a failed LLM call also fails its node and the turn, those are not counted again.
        errors = [{"source": s["kind"], "name": s["name"], "type": s.get("error_type", "other"), "error": s["error"]}
                  for s in self.spans if s["status"] == "error" and s["kind"] != "node"]
        failure = None
        if error is not None:
            failure = {"type": classify_error(_short_error(error)), "error": _short_error(error)}
        retries = [e for e in self.events if e["type"] == "provider_retry"]
        return {
            "turn_id": self.turn_id,
            "timestamp": self.started_at.isoformat(timespec="seconds"),
            **self.meta,
            "status": "error" if error is not None else status,
            "latency_ms": round(self._ms(), 1),
            "spans": sorted(self.spans, key=lambda s: s["start_ms"]),
            "events": self.events,
            "tokens": tokens,
            "total_tokens": sum(t["input"] + t["output"] for t in tokens.values()),
            "cached_tokens": sum(t["cached"] for t in tokens.values()),
            "llm_calls": len(llm),
            "tool_calls": sum(1 for s in self.spans if s["kind"] == "tool"),
            "cost_usd": round(cost, 6),
            "errors": errors,
            "failure": failure,
            "http_429": sum(1 for e in self.events if e["type"] == "http_status" and e["code"] == 429),
            "provider_retries": len(retries),
            "retry_wait_s": round(sum(e.get("wait_s", 0) for e in retries), 2),
            "cpu_ms": round((sum(_process.cpu_times()[:2]) - self._cpu0) * 1000, 1),
            "rss_mb": round(_process.memory_info().rss / 2**20, 1),
            "rss_delta_mb": round((_process.memory_info().rss - self._rss0) / 2**20, 1),
            **fields,
        }


@contextmanager
def observe_turn(**meta: Any):
    """Activate a TurnRecorder for the code inside the block (pass it as callback to the graph)."""
    recorder = TurnRecorder(**meta)
    token = _current.set(recorder)
    try:
        yield recorder
    finally:
        _current.reset(token)


@contextmanager
def span(name: str, kind: str):
    """Time a sub-step of the current turn (no-op outside a turn). Yields a dict for extra attributes."""
    recorder = _current.get()
    extra: dict = {}
    if recorder is None:
        yield extra
        return
    start = recorder._ms()
    error: BaseException | None = None
    try:
        yield extra
    except BaseException as exc:
        error = exc
        raise
    finally:
        entry = {"kind": kind, "name": name, "start_ms": round(start, 1),
                 "duration_ms": round(recorder._ms() - start, 1), "status": "ok", **extra}
        if error is not None or "error" in extra:
            entry["status"] = "error"
            entry.setdefault("error", _short_error(error) if error else "")
            entry["error_type"] = classify_error(entry["error"])
        with recorder._lock:
            recorder.spans.append(entry)


class _ProviderLogHandler(logging.Handler):
    """Turns Groq/httpx client logs into events of the current turn (429 responses and retry waits)."""

    _status = re.compile(r'"HTTP/[\d.]+ (\d{3})')
    _retry = re.compile(r"Retrying request to (\S+) in ([\d.]+) seconds")

    def emit(self, record: logging.LogRecord) -> None:
        recorder = _current.get()
        if recorder is None:
            return
        message = record.getMessage()
        if record.name.startswith("groq") and (match := self._retry.search(message)):
            recorder.add_event("provider_retry", wait_s=float(match.group(2)))
        elif "api.groq.com" in message and (match := self._status.search(message)):
            code = int(match.group(1))
            if code >= 400:
                recorder.add_event("http_status", code=code)


def install_log_capture() -> None:
    handler = _ProviderLogHandler(level=logging.INFO)
    for name in ("groq._base_client", "httpx"):
        logger = logging.getLogger(name)
        if not any(isinstance(h, _ProviderLogHandler) for h in logger.handlers):
            logger.addHandler(handler)
        logger.setLevel(min(logger.level or logging.INFO, logging.INFO))


install_log_capture()
