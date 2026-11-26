"""Observability dashboard (Streamlit tab): precision, consistency, errors, latency, resources and traces.

Reads the JSONL logs of the scenario runs (evaluation/observability/*.jsonl) and of the app (logs/).
"""

import altair as alt
import pandas as pd
import streamlit as st

from bookclub.observability import store

KIND_COLORS = alt.Scale(domain=["node", "llm", "tool", "guardrail", "embedding", "vector_search", "http"],
                        range=["#9aa5b1", "#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#b279a2"])


def _pct(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.0%}"


def _num(value, unit: str = "", digits: int = 1) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.{digits}f}{unit}"


@st.cache_data(show_spinner=False)
def _load(files: tuple[str, ...], mtimes: tuple[float, ...]) -> list[dict]:
    from pathlib import Path

    return store.read_turns([Path(f) for f in files])


@st.cache_data(show_spinner="Calculando similitud de respuestas…")
def _consistency(df: pd.DataFrame) -> pd.DataFrame:
    from bookclub.embeddings import get_embedder

    return store.consistency(df, embed=get_embedder().embed_documents)


def _kpis(summary: dict) -> None:
    rows = [
        [("Turnos", f"{summary['turns']}"), ("Answer accuracy", _pct(summary["answer_accuracy"])),
         ("Tool accuracy", _pct(summary["tool_accuracy"])), ("Retrieval hit@5", _pct(summary["retrieval_hit"])),
         ("Turnos fallidos", _pct(summary["error_rate"]))],
        [("Latencia p50", _num(summary["latency_p50_s"], " s")), ("Latencia p95", _num(summary["latency_p95_s"], " s")),
         ("Tokens por turno", _num(summary["tokens_mean"], "", 0)),
         ("Costo / 1.000 turnos", _num(summary["cost_per_1000_turns_usd"], " USD", 2)),
         ("Turnos con HTTP 429", _pct(summary["turns_with_429"]))],
    ]
    for row in rows:
        for col, (label, value) in zip(st.columns(len(row)), row, strict=True):
            col.metric(label, value)


def _runs_table(df: pd.DataFrame) -> pd.DataFrame:
    table = {run: store.summary(group) for run, group in df.groupby("run")}
    labels = {"turns": "turnos", "answer_accuracy": "answer acc.", "tool_accuracy": "tool acc.",
              "retrieval_hit": "hit@5", "error_rate": "turnos fallidos", "tool_error_rate": "errores por tool call",
              "turns_with_429": "turnos con 429", "blocked_rate": "bloqueados", "latency_p50_s": "p50 (s)",
              "latency_p95_s": "p95 (s)", "latency_max_s": "máx (s)", "tokens_mean": "tokens/turno",
              "llm_calls_mean": "llamadas LLM/turno", "cost_per_1000_turns_usd": "USD/1.000 turnos",
              "cpu_s_mean": "CPU (s)/turno", "rss_mb_max": "RAM máx (MB)"}
    return pd.DataFrame(table).loc[list(labels)].rename(index=labels)


def _precision_tab(df: pd.DataFrame) -> None:
    scored = df[df["answer_correct"].notna()]
    if scored.empty:
        st.info("Sin turnos evaluados (ejecuta `python -m evaluation.observe`).")
        return
    by_cat = (scored.groupby(["run", "category"])
              .agg(answer=("answer_correct", "mean"), tools=("tool_correct", "mean"), n=("turn_id", "count"))
              .reset_index().melt(["run", "category", "n"], var_name="métrica", value_name="valor"))
    chart = alt.Chart(by_cat).mark_bar().encode(
        x=alt.X("category:N", title="categoría"), xOffset="run:N", y=alt.Y("valor:Q", axis=alt.Axis(format="%"),
                                                                           title=None),
        color=alt.Color("run:N", title="corrida"), row=alt.Row("métrica:N", title=None),
        tooltip=["run", "category", "métrica", alt.Tooltip("valor:Q", format=".0%"), "n"],
    ).properties(height=160)
    st.markdown("**Precisión por categoría** (answer = juez LLM; tools = herramientas esperadas)")
    st.altair_chart(chart, use_container_width=True)

    st.markdown("**Consistencia entre repeticiones del mismo escenario**")
    repeated = df[df["scenario_id"].notna()].groupby(["run", "scenario_id"]).size()
    if not (repeated >= 2).any():
        st.info("Se necesitan al menos 2 repeticiones por escenario.")
        return
    cons = _consistency(df)
    st.dataframe(cons, hide_index=True, use_container_width=True, column_config={
        "tool_consistency": st.column_config.ProgressColumn("misma herramienta", format="percent", max_value=1),
        "verdict_consistency": st.column_config.ProgressColumn("mismo veredicto", format="percent", max_value=1),
        "answer_similarity": st.column_config.NumberColumn("similitud respuestas", format="%.3f"),
        "accuracy": st.column_config.NumberColumn("accuracy", format="percent"),
        "latency_cv": st.column_config.NumberColumn("CV latencia", format="%.2f"),
        "tokens_cv": st.column_config.NumberColumn("CV tokens", format="%.2f"),
    })


def _latency_tab(df: pd.DataFrame, turns: list[dict], spans: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Latencia por categoría**")
        st.altair_chart(alt.Chart(df).mark_boxplot(extent="min-max").encode(
            x=alt.X("category:N", title=None), xOffset="run:N", y=alt.Y("latency_s:Q", title="segundos"),
            color=alt.Color("run:N", title="corrida")), use_container_width=True)
    with right:
        st.markdown("**¿Dónde se va el tiempo?** (promedio por turno)")
        steps = store.step_breakdown(turns)
        if not steps.empty:
            mean = steps.drop(columns="turn_id").groupby("run").mean().reset_index().melt(
                "run", var_name="componente", value_name="segundos")
            st.altair_chart(alt.Chart(mean).mark_bar().encode(
                y=alt.Y("run:N", title=None), x=alt.X("segundos:Q", stack="zero"),
                color=alt.Color("componente:N", legend=alt.Legend(orient="bottom", columns=3)),
                tooltip=["run", "componente", alt.Tooltip("segundos:Q", format=".2f")]),
                use_container_width=True)
    if not spans.empty:
        st.markdown("**Latencia de cada paso** (llamadas al LLM, herramientas, recuperación, guardrails)")
        steps = spans[spans["kind"] != "node"]
        st.altair_chart(alt.Chart(steps).mark_boxplot(extent="min-max").encode(
            y=alt.Y("name:N", title=None, sort="-x"), x=alt.X("duration_s:Q", title="segundos",
                                                             scale=alt.Scale(type="symlog")),
            color=alt.Color("kind:N", scale=KIND_COLORS, title="tipo")).properties(height=380),
            use_container_width=True)
    st.markdown("**Latencia vs. respuestas HTTP 429 del proveedor**")
    st.altair_chart(alt.Chart(df).mark_circle(size=70).encode(
        x=alt.X("tokens:Q", title="tokens del turno"), y=alt.Y("latency_s:Q", title="latencia (s)"),
        color=alt.Color("http_429:Q", title="429", scale=alt.Scale(scheme="orangered")),
        tooltip=["run", "scenario_id", "category", "latency_s", "tokens", "http_429", "retry_wait_s"]),
        use_container_width=True)


def _resources_tab(df: pd.DataFrame, spans: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Tokens por turno y modelo**")
        llm = spans[spans["kind"] == "llm"]
        if not llm.empty:
            tokens = (llm.groupby(["run", "turn_id", "model"])[["input_tokens", "output_tokens"]].sum()
                      .groupby(["run", "model"]).mean().reset_index()
                      .melt(["run", "model"], var_name="tipo", value_name="tokens"))
            st.altair_chart(alt.Chart(tokens).mark_bar().encode(
                y=alt.Y("run:N", title=None), x=alt.X("tokens:Q", stack="zero"),
                color=alt.Color("model:N", title="modelo"), opacity=alt.Opacity("tipo:N", title="tipo"),
                tooltip=["run", "model", "tipo", alt.Tooltip("tokens:Q", format=",.0f")]), use_container_width=True)
    with right:
        st.markdown("**Tokens promedio por categoría**")
        st.altair_chart(alt.Chart(df).mark_bar().encode(
            x=alt.X("category:N", title=None), xOffset="run:N", y=alt.Y("mean(tokens):Q", title="tokens"),
            color=alt.Color("run:N", title="corrida")), use_container_width=True)
    left, right = st.columns(2)
    ordered = df.sort_values("timestamp").assign(turno=range(1, len(df) + 1))
    with left:
        st.markdown("**CPU por turno** (s de CPU del proceso)")
        st.altair_chart(alt.Chart(ordered).mark_line(point=True).encode(
            x="turno:Q", y=alt.Y("cpu_s:Q", title="CPU (s)"), color=alt.Color("run:N", title="corrida"),
            tooltip=["run", "scenario_id", "cpu_s", "latency_s"]), use_container_width=True)
    with right:
        st.markdown("**Memoria residente del proceso** (MB)")
        st.altair_chart(alt.Chart(ordered).mark_line(point=True).encode(
            x="turno:Q", y=alt.Y("rss_mb:Q", title="RSS (MB)"), color=alt.Color("run:N", title="corrida"),
            tooltip=["run", "scenario_id", "rss_mb"]), use_container_width=True)


def _errors_tab(turns: list[dict]) -> None:
    errors = store.errors_frame(turns)
    if errors.empty:
        st.success("Sin errores registrados.")
        return
    left, right = st.columns(2)
    with left:
        st.markdown("**Errores por tipo**")
        st.altair_chart(alt.Chart(errors).mark_bar().encode(
            y=alt.Y("type:N", title=None, sort="-x"), x=alt.X("count():Q", title="ocurrencias"),
            color=alt.Color("run:N", title="corrida"), tooltip=["type", "run", "count()"]),
            use_container_width=True)
    with right:
        st.markdown("**Errores por componente**")
        st.altair_chart(alt.Chart(errors).mark_bar().encode(
            y=alt.Y("name:N", title=None, sort="-x"), x=alt.X("count():Q", title="ocurrencias"),
            color=alt.Color("type:N", title="tipo"), tooltip=["name", "type", "count()"]), use_container_width=True)
    st.dataframe(errors.drop(columns="turn_id"), hide_index=True, use_container_width=True)


def _traces_tab(df: pd.DataFrame, turns: list[dict]) -> None:
    by_id = {t["turn_id"]: t for t in turns}
    ordered = df.sort_values("latency_s", ascending=False)
    labels = {r.turn_id: f"{r.latency_s:5.1f}s · {r.status} · {r.run} · {r.scenario_id or r.category} · "
                         f"{(r.input or '')[:60]}" for r in ordered.itertuples()}
    turn_id = st.selectbox("Turno (ordenados por latencia)", list(labels), format_func=labels.get)
    turn = by_id[turn_id]
    spans = pd.DataFrame(turn["spans"]).assign(
        inicio=lambda d: d["start_ms"] / 1000, fin=lambda d: (d["start_ms"] + d["duration_ms"]) / 1000,
        paso=lambda d: [f"{i:02d} {n}" for i, n in enumerate(d["name"])])
    st.altair_chart(alt.Chart(spans).mark_bar(height=12).encode(
        x=alt.X("inicio:Q", title="segundos desde el inicio del turno"), x2="fin:Q",
        y=alt.Y("paso:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False)),
        color=alt.Color("kind:N", scale=KIND_COLORS, title="tipo"),
        tooltip=["name", "kind", "node", alt.Tooltip("duration_ms:Q", title="ms"), "status",
                 *[c for c in ("model", "input_tokens", "output_tokens", "error") if c in spans]],
    ).properties(height=max(160, 22 * len(spans))), use_container_width=True)
    cols = st.columns(5)
    cols[0].metric("Latencia", f"{turn['latency_ms'] / 1000:.1f} s")
    cols[1].metric("Tokens", f"{turn['total_tokens']:,}")
    cols[2].metric("Llamadas LLM", turn["llm_calls"])
    cols[3].metric("HTTP 429", turn["http_429"])
    cols[4].metric("Espera por reintentos", f"{turn['retry_wait_s']:.1f} s")
    st.markdown(f"**Entrada (datos personales enmascarados):** {turn.get('input', '')}")
    st.markdown(f"**Respuesta:** {turn.get('answer', '')}")
    if turn.get("judge_reason"):
        st.markdown(f"**Juez:** {turn['judge_reason']}")
    with st.expander("Registro completo (JSON)"):
        st.json(turn, expanded=False)


def render() -> None:
    files = store.log_files()
    if not files:
        st.info("Aún no hay registros. Conversa con el agente o ejecuta `python -m evaluation.observe --tag base`.")
        return
    turns = _load(tuple(map(str, files)), tuple(f.stat().st_mtime for f in files))
    df = store.turns_frame(turns)
    runs = sorted(df["run"].unique())
    left, right = st.columns(2)
    chosen_runs = left.multiselect("Corridas", runs, default=runs)
    categories = sorted(df["category"].unique())
    chosen_cats = right.multiselect("Categorías", categories, default=categories)
    df = df[df["run"].isin(chosen_runs) & df["category"].isin(chosen_cats)]
    if df.empty:
        st.warning("No hay turnos con esos filtros.")
        return
    turns = [t for t in turns if t["turn_id"] in set(df["turn_id"])]
    spans = store.spans_frame(turns)

    _kpis(store.summary(df))
    with st.expander("Comparación entre corridas", expanded=len(chosen_runs) > 1):
        st.dataframe(_runs_table(df), use_container_width=True)
    precision, latency, resources, errors, traces = st.tabs(
        ["🎯 Precisión y consistencia", "⏱️ Latencia", "🧮 Recursos", "🚨 Errores", "🔎 Trazas"])
    with precision:
        _precision_tab(df)
    with latency:
        _latency_tab(df, turns, spans)
    with resources:
        _resources_tab(df, spans)
    with errors:
        _errors_tab(turns)
    with traces:
        _traces_tab(df, turns)
