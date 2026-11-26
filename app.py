"""Streamlit UI: conversation with the agent, club records, evaluation and observability.

Run: streamlit run app.py
"""

import json
import uuid

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from bookclub import memory, records  # noqa: E402  (load .env before importing settings users)
from bookclub.agent.runtime import ask, warm_up  # noqa: E402
from bookclub.db import connect  # noqa: E402
from bookclub.observability import dashboard  # noqa: E402
from evaluation.evaluate import RESULTS_DIR, load_dataset, run_evaluation  # noqa: E402

st.set_page_config(page_title="Clubes de lectura", page_icon=":material/menu_book:", layout="wide")
st.logo("assets/logo.svg", size="large")

SUGGESTIONS = [
    "¿Con cuántas personas se recomienda trabajar en un club?",
    "¿Qué libros de Baldomero Lillo hay en nuestro catálogo?",
    "Arma el calendario de lectura de Sub terra desde el 2026-12-07",
    "¿Cómo va nuestro club?",
]
TOOL_LABELS = {
    "search_club_guidelines": "Manual del SNBP", "query_database": "Base de datos", "check_availability": "Disponibilidad",
    "get_club_status": "Estado del club", "search_book_info": "Open Library", "search_author_context": "Wikipedia",
    "reserve_copies": "Reserva", "create_reading_plan": "Plan de lectura", "log_session": "Registro de sesión",
    "replan_reading": "Reprogramación", "cancel_reading_plan": "Cancelar plan", "remember_club_fact": "Memoria del club",
    "save_session_guide": "Guía de sesión",
}


@st.cache_resource(show_spinner="Cargando el modelo de embeddings…")
def warm_up_once() -> bool:
    warm_up()
    return True


@st.cache_data
def load_clubs() -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute("SELECT club_id, name, region, library, participants, frequency_text FROM clubs "
                            "ORDER BY region, name").fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def new_thread() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []


def show_trace(reply: dict) -> None:
    telemetry = reply.get("telemetry") or {}
    summary = (f" · {telemetry['latency_ms'] / 1000:.1f} s · {telemetry['total_tokens']:,} tokens".replace(",", ".")
               if telemetry else "")
    with st.expander(f"Trazabilidad: {len(reply['trace'])} herramientas{summary}", icon=":material/account_tree:"):
        guard = reply.get("guardrail") or {}
        if guard.get("reason"):
            st.warning("El mensaje no se envió al modelo.", title=f"Bloqueado por guardrail ({guard['reason']})",
                       icon=":material/shield:")
        if guard.get("pii"):
            st.info(f"Se enmascararon antes de enviarlos al modelo: {', '.join(guard['pii'])}.",
                    title="Datos personales protegidos", icon=":material/lock:")
        if reply["plan"]:
            st.markdown("**Plan de tareas**")
            st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(reply["plan"], 1)))
        if reply["memories"]:
            st.markdown("**Memoria recuperada del club**")
            st.markdown("\n".join(f"- {m}" for m in reply["memories"]))
        for step in reply["trace"]:
            ok = step["status"] == "success" and not str(step["content"]).startswith("ERROR")
            icon = ":green[:material/check_circle:]" if ok else ":orange[:material/error:]"
            st.markdown(f"{icon} **{TOOL_LABELS.get(step['tool'], step['tool'])}** · `{step['tool']}`")
            st.json(step["args"], expanded=False)
            if step["artifact"]:
                st.json(step["artifact"], expanded=False)


# ---------------------------------------------------------------- shared state and sidebar
warm_up_once()
clubs = load_clubs()
if "thread_id" not in st.session_state:
    new_thread()

with st.sidebar:
    st.caption("Club activo")
    region = st.selectbox("Región", sorted(clubs["region"].unique()))
    options = clubs[clubs["region"] == region].set_index("club_id")
    club_id = st.selectbox("Club", options.index, format_func=lambda c: options.loc[c, "name"])
    if st.session_state.get("club_id") != club_id:
        st.session_state.club_id = club_id
        new_thread()
    club = options.loc[club_id]
    participants = f"{int(club['participants'])} participantes" if pd.notna(club["participants"]) else "sin dato"
    st.caption(f":material/local_library: {club['library'] or 'Biblioteca sin dato'}  \n"
               f":material/groups: {participants}  \n"
               f":material/event_repeat: {club['frequency_text'] or 'frecuencia sin dato'}")


# ---------------------------------------------------------------- pages
def conversation_page() -> None:
    header, action = st.columns([4, 1], vertical_alignment="bottom")
    header.title("Conversación")
    header.caption(f"{club['name']} · {club['region']}")
    action.button("Nueva conversación", icon=":material/add_comment:", on_click=new_thread, width="stretch")
    for turn in st.session_state.history:
        avatar = ":material/person:" if turn["role"] == "user" else ":material/auto_stories:"
        with st.chat_message(turn["role"], avatar=avatar):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                show_trace(turn)

    suggestion = None
    if not st.session_state.history:
        suggestion = st.pills("Puedes empezar con", SUGGESTIONS, key=f"suggestion-{st.session_state.thread_id}")
    prompt = st.chat_input("Pregunta o pide algo para tu club") or suggestion
    if not prompt:
        return
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=":material/auto_stories:"), st.spinner("Pensando…"):
        reply = ask(club_id, st.session_state.thread_id, prompt)
        turn = {"role": "assistant", "content": reply.answer, "plan": reply.plan, "trace": reply.trace,
                "memories": reply.memories, "guardrail": reply.guardrail, "telemetry": reply.telemetry}
        st.markdown(reply.answer)
        show_trace(turn)
    st.session_state.history.append(turn)
    st.rerun()


def club_page() -> None:
    with connect() as conn:
        status = records.club_status(conn, club_id)
    info = status["club"]
    st.title(info["name"])
    st.caption(f"{info['library'] or ''} · {info['frequency_text'] or ''}")
    cols = st.columns(4)
    cols[0].metric("Región", info["region"], icon=":material/map:", border=True)
    cols[1].metric("Participantes", info["participants"] or "—", icon=":material/groups:", border=True)
    per_month = info["sessions_per_month"]
    cols[2].metric("Sesiones por mes", f"{per_month:g}" if per_month else "—", icon=":material/event_repeat:",
                   border=True)
    cols[3].metric("Títulos leídos 2024", info["titles_read_2024"] or "—", icon=":material/library_books:", border=True)

    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader(":material/calendar_month: Plan de lectura activo")
        plan = status["active_plan"]
        if plan:
            st.markdown(f"**{plan['book_title']}** · {plan['total_pages']} páginas")
            st.dataframe(pd.DataFrame(plan["sessions"]), hide_index=True, width="stretch")
        else:
            st.caption("Sin plan activo.")
    with right.container(border=True):
        st.subheader(":material/bookmark_added: Reservas activas")
        if status["active_reservations"]:
            st.dataframe(pd.DataFrame(status["active_reservations"]), hide_index=True, width="stretch")
        else:
            st.caption("Sin reservas activas.")
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader(":material/history_edu: Últimas sesiones")
        if status["recent_sessions"]:
            st.dataframe(pd.DataFrame(status["recent_sessions"]), hide_index=True, width="stretch")
        else:
            st.caption("Aún no hay sesiones registradas.")
    with right.container(border=True):
        st.subheader(":material/psychology: Memoria del club")
        notes = memory.recent_memories(club_id)
        if notes:
            st.dataframe(pd.DataFrame([{"tipo": m["kind"], "recuerdo": m["text"], "fecha": m["created_at"]}
                                       for m in notes]), hide_index=True, width="stretch")
        else:
            st.caption("El agente todavía no guarda recuerdos de este club.")


def evaluation_page() -> None:
    st.title("Evaluación")
    dataset = load_dataset()
    st.caption(f"Dataset de {len(dataset)} preguntas con respuesta verificada. Cada pregunta corre en una conversación "
               "nueva y las acciones se prueban sobre una copia de la base.")
    if st.button("Ejecutar evaluación completa", icon=":material/play_arrow:", type="primary",
                 help="Toma varios minutos por los límites de Groq"):
        progress = st.progress(0.0)
        run_evaluation(on_progress=lambda done, total: progress.progress(done / total))
    runs = sorted(RESULTS_DIR.glob("run_*.json"), reverse=True)
    if not runs:
        st.info("Todavía no hay resultados. Ejecuta `python -m evaluation.evaluate`.", icon=":material/info:")
        return
    chosen = st.selectbox("Resultados", runs, format_func=lambda p: p.stem)
    report = json.loads(chosen.read_text(encoding="utf-8"))
    summary = report["summary"]
    if not report.get("complete", True):
        st.warning(f"Se evaluaron {summary['n']} preguntas. Se reanuda con `python -m evaluation.evaluate --resume`.",
                   title="Evaluación incompleta", icon=":material/pending:")
    cols = st.columns(3)
    cols[0].metric("Tool accuracy", f"{summary['tool_accuracy']:.0%}", icon=":material/build:", border=True)
    cols[1].metric("Answer accuracy", f"{summary['answer_accuracy']:.0%}", icon=":material/task_alt:", border=True)
    hit = summary["retrieval_hit_rate"]
    cols[2].metric("Retrieval hit@5", f"{hit:.0%}" if hit is not None else "—", icon=":material/menu_book:", border=True)
    st.dataframe(pd.DataFrame(summary["by_category"]).T, width="stretch")
    table = pd.DataFrame([{k: r[k] for k in ("id", "category", "tool_correct", "answer_correct", "tools_called",
                                               "expected_tools")} for r in report["results"]])
    st.dataframe(table, hide_index=True, width="stretch")
    for r in report["results"]:
        icon = ":material/check_circle:" if r["answer_correct"] else ":material/cancel:"
        with st.expander(f"{r['id']} · {r['question']}", icon=icon):
            st.markdown(f"**Esperado:** {r['expected_answer']}")
            st.markdown(f"**Respuesta:** {r['answer']}")
            st.markdown(f"**Juez:** {r['judge_reason']}")


def observability_page() -> None:
    st.title("Observabilidad")
    st.caption("Turnos registrados por la app y por las corridas de escenarios en evaluation/observability.")
    dashboard.render()


page = st.navigation([
    st.Page(conversation_page, title="Conversación", icon=":material/forum:", default=True),
    st.Page(club_page, title="Registro del club", icon=":material/folder_shared:", url_path="club"),
    st.Page(evaluation_page, title="Evaluación", icon=":material/fact_check:", url_path="evaluacion"),
    st.Page(observability_page, title="Observabilidad", icon=":material/monitoring:", url_path="observabilidad"),
], position="sidebar")
page.run()
