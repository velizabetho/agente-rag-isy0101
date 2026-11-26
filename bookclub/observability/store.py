"""Structured execution logs (JSON Lines) and the metrics computed from them.

One line per agent turn (see recorder.TurnRecorder.finish). The app appends to logs/agent_turns.jsonl and the
scenario runner writes one file per run in evaluation/observability/. The Streamlit dashboard and the CLI
analysis (python -m evaluation.analyze) read the same files through these functions.
"""

import itertools
import json
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from bookclub.config import ROOT_DIR

APP_LOG = ROOT_DIR / "logs" / "agent_turns.jsonl"
RUNS_DIR = ROOT_DIR / "evaluation" / "observability"
STEP_GROUPS = {"guardrail": "guardrails", "embedding": "embeddings", "vector_search": "búsqueda vectorial",
               "http": "APIs externas"}


def write_turn(record: dict, path: Path = APP_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_turns(paths: Iterable[Path]) -> list[dict]:
    turns = []
    for path in paths:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                turns += [json.loads(line) for line in f if line.strip()]
    return turns


def log_files() -> list[Path]:
    files = sorted(RUNS_DIR.glob("*.jsonl"))
    return files + ([APP_LOG] if APP_LOG.exists() else [])


# ---------------------------------------------------------------- tables
def turns_frame(turns: list[dict]) -> pd.DataFrame:
    rows = []
    for t in turns:
        nodes = {s["name"]: s["duration_ms"] for s in t["spans"] if s["kind"] == "node"}
        llm_ms = sum(s["duration_ms"] for s in t["spans"] if s["kind"] == "llm")
        tool_ms = sum(s["duration_ms"] for s in t["spans"] if s["kind"] == "tool")
        rows.append({
            "turn_id": t["turn_id"], "timestamp": pd.to_datetime(t["timestamp"]), "source": t.get("source", "app"),
            "run": t.get("run_tag") or t.get("source", "app"), "scenario_id": t.get("scenario_id"),
            "category": t.get("category") or "app", "variant": t.get("variant") or "-",
            "repetition": t.get("repetition"), "model": t.get("agent_model", "?"), "status": t["status"],
            "latency_s": t["latency_ms"] / 1000, "llm_s": llm_ms / 1000, "tools_s": tool_ms / 1000,
            "planner_s": nodes.get("planner", 0) / 1000, "prepare_s": nodes.get("prepare", 0) / 1000,
            "guard_s": nodes.get("guard_input", 0) / 1000,
            "tokens": t["total_tokens"], "llm_calls": t["llm_calls"], "tool_calls": t["tool_calls"],
            "cost_usd": t["cost_usd"], "http_429": t["http_429"], "retry_wait_s": t["retry_wait_s"],
            "cpu_s": t["cpu_ms"] / 1000, "rss_mb": t["rss_mb"], "errors": len(t["errors"]),
            "tool_errors": sum(1 for e in t["errors"] if e["source"] == "tool"),
            "blocked": bool((t.get("guardrail") or {}).get("reason")),
            "tool_correct": t.get("tool_correct"), "answer_correct": t.get("answer_correct"),
            "retrieval_hit": t.get("retrieval_hit"), "tools_called": ",".join(t.get("tools_called", [])),
            "input": t.get("input", ""), "answer": t.get("answer", ""),
        })
    return pd.DataFrame(rows)


def spans_frame(turns: list[dict]) -> pd.DataFrame:
    rows = []
    for t in turns:
        for s in t["spans"]:
            rows.append({"turn_id": t["turn_id"], "run": t.get("run_tag") or t.get("source", "app"),
                         "category": t.get("category") or "app", "kind": s["kind"], "name": s["name"],
                         "node": s.get("node"), "model": s.get("model"), "start_s": s["start_ms"] / 1000,
                         "duration_s": s["duration_ms"] / 1000, "status": s["status"],
                         "error_type": s.get("error_type"), "input_tokens": s.get("input_tokens"),
                         "output_tokens": s.get("output_tokens")})
    return pd.DataFrame(rows)


def _root_errors(turn: dict) -> list[dict]:
    """Errors of a turn; a failed turn without an attributed span error counts as one error of the turn."""
    errors = list(turn["errors"])
    if turn.get("failure") and not errors:
        errors.append({"source": "turn", "name": "invoke", **turn["failure"]})
    errors += [{"source": "provider", "name": "groq", "type": "http_429", "error": "HTTP 429 Too Many Requests"}
               for _ in range(turn["http_429"])]
    return errors


def errors_frame(turns: list[dict]) -> pd.DataFrame:
    rows = [{"turn_id": t["turn_id"], "timestamp": pd.to_datetime(t["timestamp"]),
             "run": t.get("run_tag") or t.get("source", "app"), "category": t.get("category") or "app",
             "source": e["source"], "name": e["name"], "type": e["type"], "error": e["error"]}
            for t in turns for e in _root_errors(t)]
    return pd.DataFrame(rows, columns=["turn_id", "timestamp", "run", "category", "source", "name", "type", "error"])


def step_breakdown(turns: list[dict]) -> pd.DataFrame:
    """Seconds of each turn spent in each component (time inside tools minus their own sub-steps).

    An LLM span is split into the time Groq reports for the request (server_ms) and the rest, which is
    waiting: HTTP 429 retries with back-off plus network."""
    rows = []
    for t in turns:
        parts: Counter = Counter()
        for s in t["spans"]:
            if s["kind"] == "llm":
                server = min(s.get("server_ms") or s["duration_ms"], s["duration_ms"])
                parts[f"LLM {s.get('node') or ''}".strip()] += server
                parts["espera proveedor (429 y red)"] += s["duration_ms"] - server
            elif s["kind"] == "tool":
                parts["herramientas (resto)"] += s["duration_ms"]
            elif s["kind"] in STEP_GROUPS:
                parts[STEP_GROUPS[s["kind"]]] += s["duration_ms"]
                if s["kind"] != "guardrail" and any(
                        o["kind"] == "tool" and o["start_ms"] <= s["start_ms"] <= o["start_ms"] + o["duration_ms"]
                        for o in t["spans"]):
                    parts["herramientas (resto)"] -= s["duration_ms"]
        accounted = sum(parts.values())
        parts["otros (grafo, memoria, BD)"] = max(t["latency_ms"] - accounted, 0)
        rows.append({"turn_id": t["turn_id"], "run": t.get("run_tag") or t.get("source", "app"),
                     **{k: v / 1000 for k, v in parts.items()}})
    return pd.DataFrame(rows).fillna(0)


# ---------------------------------------------------------------- metrics
def percentile(values, q: float) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.percentile(values, q)) if values else None


def _rate(series: pd.Series) -> float | None:
    values = series.dropna()
    return float(values.astype(bool).mean()) if len(values) else None


def summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return {
        "turns": len(df),
        "answer_accuracy": _rate(df["answer_correct"]),
        "tool_accuracy": _rate(df["tool_correct"]),
        "retrieval_hit": _rate(df["retrieval_hit"]),
        "error_rate": float((df["status"] == "error").mean()),
        "turns_with_errors": float((df["errors"] > 0).mean()),
        "tool_error_rate": float(df["tool_errors"].sum() / max(df["tool_calls"].sum(), 1)),
        "turns_with_429": float((df["http_429"] > 0).mean()),
        "blocked_rate": float(df["blocked"].mean()),
        "latency_p50_s": percentile(df["latency_s"], 50),
        "latency_p95_s": percentile(df["latency_s"], 95),
        "latency_max_s": float(df["latency_s"].max()),
        "tokens_mean": float(df["tokens"].mean()),
        "llm_calls_mean": float(df["llm_calls"].mean()),
        "cost_per_1000_turns_usd": float(df["cost_usd"].mean() * 1000),
        "cpu_s_mean": float(df["cpu_s"].mean()),
        "rss_mb_max": float(df["rss_mb"].max()),
    }


def consistency(df: pd.DataFrame, embed: Callable[[list[str]], list[list[float]]] | None = None) -> pd.DataFrame:
    """Per scenario with repetitions: agreement of tool choice, of the judge verdict and answer similarity."""
    rows = []
    scenario_turns = df[df["scenario_id"].notna()]
    for (run, scenario), group in scenario_turns.groupby(["run", "scenario_id"]):
        if len(group) < 2:
            continue
        tool_sets = Counter(group["tools_called"])
        verdicts = Counter(group["answer_correct"].dropna())
        row = {"run": run, "scenario_id": scenario, "category": group["category"].iloc[0],
               "variant": group["variant"].iloc[0], "reps": len(group),
               "tool_consistency": tool_sets.most_common(1)[0][1] / len(group),
               "verdict_consistency": verdicts.most_common(1)[0][1] / sum(verdicts.values()) if verdicts else None,
               "accuracy": _rate(group["answer_correct"]),
               "latency_cv": float(group["latency_s"].std() / group["latency_s"].mean()),
               "tokens_cv": float(group["tokens"].std() / group["tokens"].mean()) if group["tokens"].mean() else 0.0,
               "tool_sets": " | ".join(f"{k or '(ninguna)'}×{v}" for k, v in tool_sets.most_common())}
        answers = [a for a in group["answer"] if a]
        if embed and len(answers) >= 2:
            vectors = np.array(embed(answers))
            sims = [float(vectors[i] @ vectors[j]) for i, j in itertools.combinations(range(len(vectors)), 2)]
            row["answer_similarity"] = float(np.mean(sims))
        rows.append(row)
    return pd.DataFrame(rows)
