"""Streamlit UI: chat with the agent, club records and evaluation results.

Run: streamlit run app.py
"""

import json
import uuid

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from bookclub import memory, records  # noqa: E402  (load .env before importing settings users)
from bookclub.agent.runtime import ask  # noqa: E402
from bookclub.db import connect  # noqa: E402
from evaluation.evaluate import RESULTS_DIR, load_dataset, run_evaluation  # noqa: E402

st.set_page_config(page_title="Asistente de clubes de lectura", page_icon="📚", layout="wide")


@st.cache_data
def load_clubs() -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute("SELECT club_id, name, region, library FROM clubs ORDER BY region, name").fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def show_trace(reply: dict) -> None:
    with st.expander(f"🔧 Trazabilidad ({len(reply['trace'])} herramientas)"):
        if reply["plan"]:
            st.markdown("**Plan de tareas**")
            st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(reply["plan"], 1)))
        if reply["memories"]:
            st.markdown("**Memoria recuperada del club**")
            st.markdown("\n".join(f"- {m}" for m in reply["memories"]))
        for step in reply["trace"]:
            icon = "✅" if step["status"] == "success" and not str(step["content"]).startswith("ERROR") else "⚠️"
            st.markdown(f"{icon} `{step['tool']}`")
            st.json(step["args"], expanded=False)
            if step["artifact"]:
                st.json(step["artifact"], expanded=False)


# ---------------------------------------------------------------- sidebar
clubs = load_clubs()
with st.sidebar:
    st.header("📚 Club")
    region = st.selectbox("Región", sorted(clubs["region"].unique()))
    options = clubs[clubs["region"] == region]
    club_id = st.selectbox("Club", options["club_id"], format_func=lambda c: options.set_index("club_id").loc[c, "name"])
    if st.session_state.get("club_id") != club_id:
        st.session_state.club_id = club_id
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
    if st.button("Nueva conversación"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
    st.caption(f"thread: `{st.session_state.thread_id[:8]}`")

chat_tab, club_tab, eval_tab = st.tabs(["💬 Chat", "🗂️ Registro del club", "📊 Evaluación"])

# ---------------------------------------------------------------- chat
with chat_tab:
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                show_trace(turn)
    if prompt := st.chat_input("Pregunta o pide algo para tu club…"):
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"), st.spinner("Pensando…"):
            reply = ask(club_id, st.session_state.thread_id, prompt)
            turn = {"role": "assistant", "content": reply.answer, "plan": reply.plan, "trace": reply.trace,
                    "memories": reply.memories}
            st.markdown(reply.answer)
            show_trace(turn)
        st.session_state.history.append(turn)

# ---------------------------------------------------------------- club records
with club_tab:
    with connect() as conn:
        status = records.club_status(conn, club_id)
    club = status["club"]
    st.subheader(club["name"])
    cols = st.columns(4)
    cols[0].metric("Región", club["region"])
    cols[1].metric("Participantes", club["participants"] or "—")
    cols[2].metric("Sesiones por mes", club["sessions_per_month"] or "—")
    cols[3].metric("Títulos leídos 2024", club["titles_read_2024"] or "—")
    st.caption(f"{club['library'] or ''} · {club['frequency_text'] or ''}")

    st.markdown("#### Plan de lectura activo")
    if status["active_plan"]:
        st.write(f"**{status['active_plan']['book_title']}** ({status['active_plan']['total_pages']} páginas)")
        st.dataframe(pd.DataFrame(status["active_plan"]["sessions"]), hide_index=True)
    else:
        st.info("Sin plan activo.")
    st.markdown("#### Reservas activas")
    st.dataframe(pd.DataFrame(status["active_reservations"]), hide_index=True)
    st.markdown("#### Últimas sesiones registradas")
    st.dataframe(pd.DataFrame(status["recent_sessions"]), hide_index=True)
    st.markdown("#### Memoria de largo plazo")
    notes = memory.recent_memories(club_id)
    st.dataframe(pd.DataFrame([{"tipo": m["kind"], "recuerdo": m["text"], "fecha": m["created_at"]} for m in notes]),
                 hide_index=True)

# ---------------------------------------------------------------- evaluation
with eval_tab:
    dataset = load_dataset()
    st.write(f"Dataset: {len(dataset)} preguntas. Cada pregunta corre en un thread nuevo.")
    if st.button("Ejecutar evaluación (toma varios minutos por los límites de Groq)"):
        progress = st.progress(0.0)
        run_evaluation(on_progress=lambda done, total: progress.progress(done / total))
    runs = sorted(RESULTS_DIR.glob("run_*.json"), reverse=True)
    if runs:
        chosen = st.selectbox("Resultados", runs, format_func=lambda p: p.stem)
        report = json.loads(chosen.read_text(encoding="utf-8"))
        summary = report["summary"]
        cols = st.columns(3)
        cols[0].metric("Tool accuracy", f"{summary['tool_accuracy']:.0%}")
        cols[1].metric("Answer accuracy", f"{summary['answer_accuracy']:.0%}")
        cols[2].metric("Retrieval hit@5", f"{summary['retrieval_hit_rate']:.0%}" if summary["retrieval_hit_rate"]
                       is not None else "—")
        st.dataframe(pd.DataFrame(summary["by_category"]).T, use_container_width=True)
        table = pd.DataFrame([{k: r[k] for k in ("id", "category", "tool_correct", "answer_correct", "tools_called",
                                                   "expected_tools")} for r in report["results"]])
        st.dataframe(table, hide_index=True, use_container_width=True)
        for r in report["results"]:
            with st.expander(f"{r['id']} · {r['question']}"):
                st.markdown(f"**Esperado:** {r['expected_answer']}")
                st.markdown(f"**Respuesta:** {r['answer']}")
                st.markdown(f"**Juez:** {r['judge_reason']}")
