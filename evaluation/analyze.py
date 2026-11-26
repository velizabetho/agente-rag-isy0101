"""Analysis of the execution logs: prints Markdown tables used in docs/observabilidad.md.

Run:
    python -m evaluation.analyze                  # all runs in evaluation/observability/
    python -m evaluation.analyze base optimizado  # only these runs (tags)
"""

import sys

import pandas as pd

from bookclub.embeddings import get_embedder
from bookclub.observability import store


def _md(frame: pd.DataFrame, floatfmt: str = ".2f") -> str:
    return frame.to_markdown(floatfmt=floatfmt)


def main(tags: list[str]) -> None:
    files = [store.RUNS_DIR / f"{t}.jsonl" for t in tags] if tags else sorted(store.RUNS_DIR.glob("*.jsonl"))
    turns = store.read_turns(files)
    df = store.turns_frame(turns)
    spans = store.spans_frame(turns)

    print("## Resumen por corrida\n")
    print(_md(pd.DataFrame({run: store.summary(g) for run, g in df.groupby("run")}), ".3f"))

    print("\n## Precisión y latencia por categoría\n")
    by_cat = df.groupby(["run", "category"]).agg(
        turnos=("turn_id", "count"), answer_acc=("answer_correct", "mean"), tool_acc=("tool_correct", "mean"),
        latencia_p50=("latency_s", "median"), latencia_max=("latency_s", "max"), tokens=("tokens", "mean"),
        llamadas_llm=("llm_calls", "mean"), con_429=("http_429", lambda s: (s > 0).mean()))
    print(_md(by_cat))

    print("\n## Tiempo promedio por componente (s por turno)\n")
    steps = store.step_breakdown(turns).drop(columns="turn_id").groupby("run").mean().T
    print(_md(steps))

    print("\n## Latencia por paso (s)\n")
    step_stats = spans[spans["kind"] != "node"].groupby(["run", "kind", "name"])["duration_s"].describe(
        percentiles=[0.5, 0.95])[["count", "mean", "50%", "95%", "max"]]
    print(_md(step_stats))

    print("\n## Errores por tipo y componente\n")
    errors = store.errors_frame(turns)
    if errors.empty:
        print("Sin errores.")
    else:
        print(_md(errors.groupby(["run", "type", "name"]).size().rename("ocurrencias").to_frame(), ".0f"))
        print("\nTurnos fallidos:\n")
        failed = [(t.get("run_tag"), t.get("scenario_id"), t.get("repetition"), t["failure"]["type"],
                   t["failure"]["error"][:140]) for t in turns if t.get("failure")]
        print(_md(pd.DataFrame(failed, columns=["run", "escenario", "rep", "tipo", "error"]), ".0f")
              if failed else "Ninguno.")

    print("\n## Consistencia entre repeticiones\n")
    cons = store.consistency(df, embed=get_embedder().embed_documents)
    if not cons.empty:
        print(_md(cons.drop(columns=["tool_sets"]).set_index(["run", "scenario_id"]), ".2f"))
        print("\nHerramientas elegidas en cada repetición:\n")
        print(_md(cons[["run", "scenario_id", "tool_sets"]].set_index(["run", "scenario_id"])))

    print("\n## Anomalías: turnos sobre el p95 de latencia de su corrida\n")
    p95 = df.groupby("run")["latency_s"].transform(lambda s: s.quantile(0.95))
    slow = df[df["latency_s"] >= p95][["run", "scenario_id", "repetition", "latency_s", "tokens", "http_429",
                                         "retry_wait_s", "llm_s", "tools_s", "prepare_s", "status"]]
    print(_md(slow.set_index(["run", "scenario_id"])))

    print("\n## Seguridad\n")
    sec = [{"run": t.get("run_tag"), "escenario": t.get("scenario_id"), "rep": t.get("repetition"),
            "bloqueado": (t.get("guardrail") or {}).get("allowed") is False,
            "score_injection": (t.get("guardrail") or {}).get("injection_score"),
            "pii_detectada": ",".join((t.get("guardrail") or {}).get("pii", [])),
            "pii_filtrada_a_tools": t.get("pii_leak"), "guardrail_ok": t.get("guardrail_correct")}
           for t in turns if t.get("category") == "seguridad"]
    print(_md(pd.DataFrame(sec), ".4f") if sec else "Sin escenarios de seguridad.")


if __name__ == "__main__":
    main(sys.argv[1:])
