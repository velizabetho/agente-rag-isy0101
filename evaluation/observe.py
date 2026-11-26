"""Run the observability scenarios (evaluation/scenarios.json) and log every turn with its telemetry.

Each scenario covers a kind of data variability (paraphrase with typos, another region, multi-step action,
no tool needed, prompt injection, personal data, benign text that looks like an injection) and is repeated
`--reps` times to measure consistency. Every repetition starts from a fresh copy of the SQLite database and a
new thread, so repetitions are independent. Each turn is judged (answer accuracy) and written as one JSON line
to evaluation/observability/<tag>.jsonl; an interrupted run is resumed by running the same command again.

Run:
    python -m evaluation.observe --tag baseline --reps 2
    python -m evaluation.observe --tag modelo-20b --reps 1 --model openai/gpt-oss-20b
"""

import argparse
import json
import tempfile
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from bookclub.agent.runtime import ask, build_agent, warm_up
from bookclub.config import get_settings
from bookclub.llm import get_llm
from bookclub.mongo import MEMORIES, get_collection
from bookclub.observability import store
from evaluation.evaluate import EVAL_CLUB, isolated_database, judge, retrieval_hit

SCENARIOS = Path(__file__).resolve().parent / "scenarios.json"


def load_scenarios(ids: list[str] | None = None) -> list[dict]:
    items = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    return [s for s in items if not ids or s["id"] in ids]


def score(scenario: dict, reply) -> dict:
    """Evaluation fields of a turn: tool choice, answer (LLM judge), retrieval and guardrail outcome."""
    called = set(reply.telemetry["tools_called"])
    expected = set(scenario["expected_tools"])
    blocked = reply.guardrail.get("allowed") is False
    result = {"expected_tools": scenario["expected_tools"], "blocked": blocked,
              "tool_correct": expected <= called if expected else not called}
    if "expected_blocked" in scenario:
        result["guardrail_correct"] = blocked == scenario["expected_blocked"]
    if blocked or scenario.get("expected_blocked"):
        result.update(answer_correct=blocked == bool(scenario.get("expected_blocked")),
                      judge_reason="bloqueado por guardrail" if blocked else "no fue bloqueado")
    else:
        verdict = judge(scenario["question"], scenario["expected_answer"], reply.answer)
        result.update(answer_correct=verdict.correct, judge_reason=verdict.reason)
    if scenario.get("expected_pages"):
        result["retrieval_hit"] = retrieval_hit(reply.trace, scenario["expected_pages"])
    if scenario.get("pii"):  # raw personal data must not reach tools (memory, logs) nor the answer
        written = json.dumps([step["args"] for step in reply.trace], ensure_ascii=False) + reply.answer
        result["pii_leak"] = any(value in written for value in scenario["pii"])
    return result


def is_daily_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "tokens per day" in text or "(tpd)" in text or "requests per day" in text


def run(tag: str, reps: int, model: str | None = None, ids: list[str] | None = None) -> Path:
    out = store.RUNS_DIR / f"{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(t["scenario_id"], t["repetition"]) for t in store.read_turns([out])}
    agent_model = model or get_settings().llm_model
    agent = build_agent(InMemorySaver(), llm=get_llm(agent_model))
    warm_up()  # same as the app: the embedding model is loaded before the first turn
    scenarios = load_scenarios(ids)
    pending = [(s, r) for r in range(1, reps + 1) for s in scenarios if (s["id"], r) not in done]
    print(f"{tag}: {len(pending)} turns pending ({len(done)} already logged) -> {out}", flush=True)

    for i, (scenario, rep) in enumerate(pending, 1):
        fields = {"source": "scenario", "run_tag": tag, "scenario_id": scenario["id"],
                  "category": scenario["category"], "variant": scenario["variant"], "repetition": rep,
                  "agent_model": agent_model}
        target = Path(tempfile.gettempdir()) / f"bookclub_observe_{tag}.db"
        target.unlink(missing_ok=True)
        get_collection(MEMORIES).delete_many({"club_id": EVAL_CLUB})
        failure: Exception | None = None
        with isolated_database(target):
            try:
                reply = ask(scenario["club_id"], f"obs-{uuid.uuid4()}", scenario["question"], agent=agent,
                            metadata={"eval": True}, log_path=None, log_fields=fields)
            except Exception as exc:
                if not hasattr(exc, "telemetry"):  # failure outside the agent turn
                    raise
                failure = exc
            finally:
                target.unlink(missing_ok=True)
        if failure is None:
            record = {**reply.telemetry, **score(scenario, reply)}
        else:
            record = {**failure.telemetry, "expected_tools": scenario["expected_tools"], "tool_correct": False,
                      "answer_correct": False, "judge_reason": f"error: {str(failure)[:200]}"}
        store.write_turn(record, out)
        print(f"[{i}/{len(pending)}] {scenario['id']} rep={rep} {record['status']} "
              f"{record['latency_ms'] / 1000:.1f}s tokens={record['total_tokens']} 429={record['http_429']} "
              f"tool={record['tool_correct']} answer={record['answer_correct']}", flush=True)
        if failure is not None and is_daily_limit(failure):
            print(f"Stopped by the provider's daily limit: {failure}", flush=True)
            break
    get_collection(MEMORIES).delete_many({"club_id": EVAL_CLUB})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="name of the run (output file evaluation/observability/<tag>.jsonl)")
    parser.add_argument("--reps", type=int, default=2, help="repetitions per scenario (consistency)")
    parser.add_argument("--model", help="agent model (default LLM_MODEL)")
    parser.add_argument("ids", nargs="*", help="only these scenario ids")
    args = parser.parse_args()
    run(args.tag, args.reps, args.model, args.ids or None)
