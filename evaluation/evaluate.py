"""Evaluate the agent on evaluation/dataset.json.

Metrics
- Tool accuracy: the expected tools were called (subset match); for no_tool/falso_positivo, no tool at all.
- Answer accuracy: an LLM judge compares the answer with the expected answer -> {correct, reason}.
- Retrieval hit@5 (rag questions): a retrieved guideline chunk comes from one of the expected pages.

Each question runs in a new thread. Write actions run against a throwaway copy of the SQLite
database so the real records are never modified; questions run in dataset order because some
actions depend on previous ones (reserve -> plan -> log -> re-plan).

Run: python -m evaluation.evaluate
"""

import json
import shutil
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from bookclub import db
from bookclub.agent.graph import build_graph
from bookclub.agent.prompts import JUDGE_SYSTEM_PROMPT, PROMPT_VERSION
from bookclub.agent.runtime import ask, club_context, recall_memories
from bookclub.config import get_settings
from bookclub.llm import get_judge_llm, get_llm, get_planner_llm
from bookclub.mongo import MEMORIES, get_collection

EVAL_DIR = Path(__file__).resolve().parent
DATASET = EVAL_DIR / "dataset.json"
RESULTS_DIR = EVAL_DIR / "results"
EVAL_CLUB = "club-de-lectura-gente-de-palabra"  # Metropolitana, 8 participants, weekly


class Judgement(BaseModel):
    correct: bool
    reason: str


def load_dataset() -> list[dict]:
    return json.loads(DATASET.read_text(encoding="utf-8"))


@contextmanager
def isolated_database(target: Path) -> Iterator[Path]:
    """Point the app at a copy of the SQLite database for the duration of the run."""
    shutil.copyfile(db.db_path(), target)
    original = db.db_path
    db.db_path = lambda: target
    try:
        yield target
    finally:
        db.db_path = original


def judge(question: str, expected: str, answer: str) -> Judgement:
    llm = get_judge_llm().with_structured_output(Judgement)
    prompt = f"Pregunta: {question}\n\nRespuesta esperada: {expected}\n\nRespuesta del asistente: {answer}"
    return llm.invoke([SystemMessage(JUDGE_SYSTEM_PROMPT), HumanMessage(prompt)])


def retrieval_hit(trace: list[dict], expected_pages: list[int]) -> bool:
    pages = {chunk["page"] for step in trace if step["tool"] == "search_club_guidelines"
             for chunk in (step["artifact"] or [])}
    return bool(pages & set(expected_pages))


def evaluate_item(item: dict, agent) -> dict:
    reply = ask(item.get("club_id", EVAL_CLUB), f"eval-{uuid.uuid4()}", item["question"], agent=agent,
                metadata={"eval": True})
    called = sorted({step["tool"] for step in reply.trace})
    expected = set(item["expected_tools"])
    tool_correct = expected <= set(called) if expected else not called
    verdict = judge(item["question"], item["expected_answer"], reply.answer)
    return {
        **item,
        "answer": reply.answer,
        "plan": reply.plan,
        "tools_called": called,
        "tool_correct": tool_correct,
        "answer_correct": verdict.correct,
        "judge_reason": verdict.reason,
        "retrieval_hit": retrieval_hit(reply.trace, item["expected_pages"]) if item.get("expected_pages") else None,
        "trace": [{k: step[k] for k in ("tool", "args", "status")} for step in reply.trace],
    }


def summarize(results: list[dict]) -> dict:
    def rate(rows, key):
        values = [r[key] for r in rows if r[key] is not None]
        return round(sum(values) / len(values), 3) if values else None

    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)
    settings = get_settings()
    return {
        "n": len(results),
        "tool_accuracy": rate(results, "tool_correct"),
        "answer_accuracy": rate(results, "answer_correct"),
        "retrieval_hit_rate": rate(results, "retrieval_hit"),
        "by_category": {
            cat: {"n": len(rows), "tool_accuracy": rate(rows, "tool_correct"),
                  "answer_accuracy": rate(rows, "answer_correct")}
            for cat, rows in by_category.items()
        },
        "prompt_version": PROMPT_VERSION,
        "models": {"agent": settings.llm_model, "planner": settings.planner_model, "judge": settings.judge_model,
                   "embeddings": settings.embedding_model},
    }


def run_evaluation(on_progress: Callable[[int, int], None] | None = None, ids: list[str] | None = None) -> dict:
    dataset = [item for item in load_dataset() if not ids or item["id"] in ids]
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    get_collection(MEMORIES).delete_many({"club_id": EVAL_CLUB})
    agent = build_graph(get_llm(), planner_llm=get_planner_llm(), club_context_fn=club_context,
                        recall_fn=recall_memories, checkpointer=InMemorySaver())

    results = []
    with isolated_database(RESULTS_DIR / f"eval_{stamp}.db") as db_copy:
        for i, item in enumerate(dataset, 1):
            results.append(evaluate_item(item, agent))
            if on_progress:
                on_progress(i, len(dataset))
            print(f"[{i}/{len(dataset)}] {item['id']} tool={results[-1]['tool_correct']} "
                  f"answer={results[-1]['answer_correct']} tools={results[-1]['tools_called']}", flush=True)
    db_copy.unlink(missing_ok=True)
    get_collection(MEMORIES).delete_many({"club_id": EVAL_CLUB})

    report = {"timestamp": stamp, "summary": summarize(results), "results": results}
    (RESULTS_DIR / f"run_{stamp}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                                                  encoding="utf-8")
    return report


if __name__ == "__main__":
    import sys

    summary = run_evaluation(ids=sys.argv[1:] or None)["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
