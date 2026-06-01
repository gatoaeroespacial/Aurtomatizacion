"""
Interfaz Streamlit — módulos experimentales N-Back y Stroop.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

from cognitive_tasks import (
    STROOP_COLORS,
    STROOP_WORDS,
    ExperimentRun,
    ExperimentStore,
    PhysioSnapshot,
    compute_nback_metrics,
    compute_stroop_metrics_dict,
    compute_task_grade,
    grade_label,
    generate_nback_sequence,
    generate_stroop_trials,
    new_run_id,
    physio_from_state,
    score_nback_response,
    score_stroop_trial,
    summarize_physio,
    validate_nback_logic_demo,
)
from config import (
    EXP_MUSIC_CONDITIONS,
    EXP_NBACK_ISI_SEC,
    EXP_NBACK_STIM_SEC,
    EXP_NBACK_TARGET_RATE,
    EXP_NBACK_TRIALS,
    EXP_STROOP_ISI_SEC,
    EXP_STROOP_STIM_SEC,
    EXP_STROOP_TRIALS,
)

_EXP_KEY = "exp_active"
_PHYSIO_MIN_INTERVAL = 1.0

CONDITION_LABELS = {
    "sin_musica": "1 — Sin música",
    "musica_fija": "2 — Música fija",
    "musica_adaptativa": "3 — Música adaptativa",
}

TASK_LABELS = {
    "nback_2": "N-Back 2-back",
    "stroop": "Stroop",
}


def _sample_physio(
    exp: dict, load_state: Callable[[], dict]
) -> None:
    now = time.time()
    if now - exp.get("last_physio_ts", 0) < _PHYSIO_MIN_INTERVAL:
        return
    s = load_state()
    if not s:
        return
    snap = physio_from_state(s)
    exp.setdefault("physio", []).append(snap)
    exp["last_physio_ts"] = now


def _finish_run(exp: dict, store: ExperimentStore) -> ExperimentRun:
    from dataclasses import asdict

    if exp["task"] == "nback_2":
        metrics = compute_nback_metrics(exp["responses"])
    else:
        metrics = compute_stroop_metrics_dict(exp["responses"])
    physio: List[PhysioSnapshot] = exp.get("physio", [])
    run = ExperimentRun(
        run_id=exp["run_id"],
        user_id=exp["user_id"],
        task=exp["task"],
        music_condition=exp["condition"],
        started_at=exp["started_at"],
        finished_at=time.time(),
        metrics=metrics if isinstance(metrics, dict) else asdict(metrics),
        physio_timeseries=[asdict(p) for p in physio],
        physio_summary=summarize_physio(physio),
        trials=[asdict(t) for t in exp["responses"]],
    )
    store.save_run(run)
    return run


def _render_physio_sidebar(exp: dict, load_state: Callable[[], dict]) -> None:
    s = load_state() or {}
    st.markdown("**Fisiología en vivo**")
    c1, c2 = st.columns(2)
    c1.metric("BPM", f"{s.get('bpm', 0):.0f}")
    c2.metric("RMSSD", f"{s.get('rmssd', 0):.0f} ms")
    c1.metric("GSR", f"{s.get('scl', 0):.2f} µS")
    c2.metric("Resp.", f"{s.get('resp_rate', 0):.1f} rpm")
    n = len(exp.get("physio", []))
    st.caption(f"Muestras sincronizadas: **{n}**")


def _nback_advance_after_response(exp: dict) -> None:
    exp["answered"] = True
    exp["phase"] = "isi"
    exp["isi_start"] = time.time()


def _tick_nback(exp: dict, load_state: Callable[[], dict]) -> None:
    _sample_physio(exp, load_state)
    now = time.time()
    idx = exp["idx"]
    trials = exp["trials"]
    trial = trials[idx]

    if exp["phase"] == "stim":
        elapsed = now - exp["stim_start"]
        if not exp.get("answered") and elapsed >= EXP_NBACK_STIM_SEC:
            resp = "omission" if trial.get("scorable") else "omission"
            exp["responses"].append(
                score_nback_response(trial, resp, None))
            _nback_advance_after_response(exp)
    elif exp["phase"] == "isi":
        if now - exp["isi_start"] >= EXP_NBACK_ISI_SEC:
            exp["idx"] += 1
            if exp["idx"] >= len(trials):
                exp["phase"] = "done"
            else:
                exp["phase"] = "stim"
                exp["stim_start"] = now
                exp["answered"] = False


def _render_nback_running(
    exp: dict, load_state: Callable[[], dict], send_cmd: Callable
) -> None:
    _tick_nback(exp, load_state)
    idx = exp["idx"]
    total = len(exp["trials"])

    if exp["phase"] == "done":
        return

    trial = exp["trials"][idx]
    pos = trial["position"]
    st.progress(idx / total)
    st.caption(
        f"Estímulo **{pos}** / {total} · "
        f"{CONDITION_LABELS.get(exp['condition'], '')}"
    )

    if exp["phase"] == "stim":
        letter = trial["stimulus"]
        st.markdown(
            f"<div style='text-align:center;font-size:6rem;font-weight:800;"
            f"padding:36px 0;color:#cdd6f4;letter-spacing:0.08em'>{letter}</div>",
            unsafe_allow_html=True,
        )

        if not trial.get("scorable"):
            st.info(
                f"Posición {pos}: memoriza la letra. "
                f"A partir de la posición 3 aplica la regla 2-back."
            )
        else:
            ref = trial.get("reference_2back", "—")
            st.markdown(
                f"<p style='text-align:center;color:#a6adc8;font-size:1rem'>"
                f"¿La letra <b>{letter}</b> coincide con la de hace "
                f"<b>2 posiciones?</b></p>",
                unsafe_allow_html=True,
            )
            b1, b2 = st.columns(2)
            disabled = exp.get("answered", False)
            if b1.button(
                "✓ SÍ — MATCH",
                key=f"nback_yes_{idx}",
                use_container_width=True,
                type="primary",
                disabled=disabled,
            ):
                rt = (time.time() - exp["stim_start"]) * 1000.0
                exp["responses"].append(
                    score_nback_response(trial, "match", rt))
                _nback_advance_after_response(exp)
                st.rerun()
            if b2.button(
                "✗ NO — NO MATCH",
                key=f"nback_no_{idx}",
                use_container_width=True,
                disabled=disabled,
            ):
                rt = (time.time() - exp["stim_start"]) * 1000.0
                exp["responses"].append(
                    score_nback_response(trial, "no_match", rt))
                _nback_advance_after_response(exp)
                st.rerun()

        if not exp.get("answered"):
            remain = max(0.0, EXP_NBACK_STIM_SEC - (time.time() - exp["stim_start"]))
            st.progress(1.0 - remain / EXP_NBACK_STIM_SEC)
            time.sleep(0.15)
            st.rerun()
    else:
        st.markdown(
            "<p style='text-align:center;color:#45475a;padding:72px 0;"
            "font-size:1.2rem'>· · ·</p>",
            unsafe_allow_html=True,
        )
        time.sleep(0.3)
        st.rerun()


def _tick_stroop(exp: dict, load_state: Callable[[], dict]) -> None:
    _sample_physio(exp, load_state)
    now = time.time()
    if exp["phase"] != "stim" or exp.get("answered"):
        if exp["phase"] == "isi":
            if now - exp["isi_start"] >= EXP_STROOP_ISI_SEC:
                exp["idx"] += 1
                if exp["idx"] >= len(exp["trials"]):
                    exp["phase"] = "done"
                else:
                    exp["phase"] = "stim"
                    exp["stim_start"] = now
                    exp["answered"] = False
        return

    elapsed = now - exp["stim_start"]
    if elapsed >= EXP_STROOP_STIM_SEC:
        trial = exp["trials"][exp["idx"]]
        rec = score_stroop_trial(trial, "", 0.0)
        rec.error_type = "timeout"
        rec.correct = False
        exp["responses"].append(rec)
        exp["answered"] = True
        exp["phase"] = "isi"
        exp["isi_start"] = now


def _render_stroop_running(
    exp: dict, load_state: Callable[[], dict], send_cmd: Callable
) -> None:
    _tick_stroop(exp, load_state)
    idx = exp["idx"]
    total = len(exp["trials"])

    if exp["phase"] == "done":
        return

    st.progress((idx + 1) / total)
    trial = exp["trials"][idx]

    if exp["phase"] == "stim" and not exp.get("answered"):
        st.markdown(
            f"<div style='text-align:center;font-size:3rem;font-weight:800;"
            f"padding:30px 0;color:{trial['hex']}'>{trial['word']}</div>",
            unsafe_allow_html=True,
        )
        st.caption("Selecciona el **color de la tinta** (no el significado de la palabra).")
        cols = st.columns(4)
        for i, word in enumerate(STROOP_WORDS):
            if cols[i].button(
                word,
                key=f"stroop_{idx}_{word}",
                use_container_width=True,
            ):
                rt = (time.time() - exp["stim_start"]) * 1000.0
                rec = score_stroop_trial(trial, word, rt)
                exp["responses"].append(rec)
                exp["answered"] = True
                exp["phase"] = "isi"
                exp["isi_start"] = time.time()
                st.rerun()
    elif exp["phase"] == "isi":
        st.markdown("<p style='text-align:center;padding:40px;color:#444'>···</p>",
                    unsafe_allow_html=True)
        time.sleep(0.25)
        st.rerun()


def _render_results(run: ExperimentRun) -> None:
    m = run.metrics
    st.success("Ensayo guardado.")
    if run.task == "nback_2":
        c1, c2, c3, c4, c5 = st.columns(5)
        acc = m.get("accuracy", 0)
        c1.metric("Ensayos", m.get("n_trials", 0))
        c2.metric("Aciertos", m.get("n_correct", 0))
        c3.metric("Errores", m.get("n_errors", 0))
        c4.metric("Precisión", f"{acc * 100:.1f}%")
        c5.metric("Omisiones", m.get("omissions", 0))
        st.metric("RT medio (respuestas)", f"{m.get('mean_rt_ms', 0):.0f} ms")
        st.info(f"**Calificación:** {m.get('display', '—')}  \n{m.get('detail', '')}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m.get('accuracy', 0) * 100:.1f}%")
        c2.metric("RT medio", f"{m.get('mean_rt_ms', 0):.0f} ms")
        c3.metric("Errores", m.get("errors", 0))
        c4.metric("Ensayos", m.get("n_trials", 0))
        st.info(f"**Calificación:** {m.get('display', '—')}  \n{m.get('detail', '')}")

    ps = run.physio_summary
    if ps:
        st.markdown("**Fisiología durante la tarea (promedio)**")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("BPM", f"{ps.get('bpm_mean', 0):.1f}")
        f2.metric("RMSSD", f"{ps.get('rmssd_mean', 0):.1f} ms")
        f3.metric("GSR", f"{ps.get('gsr_mean', 0):.2f} µS")
        f4.metric("Resp.", f"{ps.get('resp_mean', 0):.1f} rpm")


def _row_from_run(r: dict) -> dict:
    m, p = r.get("metrics", {}), r.get("physio_summary", {})
    task = r.get("task", "")
    if "score" not in m:
        m = {**m, **compute_task_grade(task, m)}
    err = m.get("n_errors", m.get("errors", 0))
    return {
        "Tarea": TASK_LABELS.get(task, task),
        "Música": CONDITION_LABELS.get(r.get("music_condition"), ""),
        "Accuracy %": round(float(m.get("accuracy", 0) or 0) * 100, 1),
        "RT ms": m.get("mean_rt_ms"),
        "Errores": err,
        "Puntuación": m.get("score"),
        "Calificación": m.get("display", ""),
        "BPM": p.get("bpm_mean"),
        "RMSSD": p.get("rmssd_mean"),
        "GSR": p.get("gsr_mean"),
    }


def _append_summary_row(df: pd.DataFrame, label: str = "📊 Promedio") -> pd.DataFrame:
    if df.empty:
        return df
    num_cols = ["Accuracy %", "RT ms", "Errores", "Puntuación", "BPM", "RMSSD", "GSR"]
    summary = {c: "" for c in df.columns}
    summary["Tarea"] = label
    for c in num_cols:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            summary[c] = round(vals.mean(), 2) if len(vals) else ""
    scores = pd.to_numeric(df.get("Puntuación", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(scores):
        avg = scores.mean()
        summary["Calificación"] = f"{avg:.1f}/100 — {grade_label(avg)}"
    return pd.concat([df, pd.DataFrame([summary])], ignore_index=True)


def _render_comparison_table(store: ExperimentStore) -> None:
    runs = store.list_runs()
    if not runs:
        st.info("Aún no hay ensayos guardados.")
        return

    rows = [_row_from_run(r) for r in runs[-30:]]
    df = pd.DataFrame(rows)
    st.markdown("**Historial de ensayos**")
    st.caption(
        "Puntuación: N-Back = precisión (− omisiones). "
        "Stroop = 65 % precisión + 35 % velocidad (RT)."
    )
    st.dataframe(_append_summary_row(df), use_container_width=True, hide_index=True)

    st.markdown("**Comparativa por condición musical**")
    if df.empty:
        return
    agg = df.groupby("Música", as_index=False).agg({
        "Accuracy %": "mean",
        "RT ms": "mean",
        "Errores": "mean",
        "Puntuación": "mean",
        "BPM": "mean",
        "RMSSD": "mean",
        "GSR": "mean",
    }).round(2)
    agg["Calificación"] = agg["Puntuación"].apply(
        lambda s: f"{s:.1f}/100 — {grade_label(float(s))}" if pd.notna(s) else ""
    )
    st.dataframe(_append_summary_row(agg, "📊 Promedio global"), use_container_width=True, hide_index=True)


def render_experiments(
    connected: bool,
    load_state: Callable[[], dict],
    send_cmd: Callable[[dict], None],
) -> None:
    st.markdown("### 🧪 Módulos experimentales")
    st.caption(
        "N-Back 2-back y Stroop con registro de accuracy, RT y errores, "
        "sincronizados con BPM, RMSSD, GSR y respiración desde `main.py`."
    )

    if not connected:
        st.warning("Inicia `python main.py --demo` (o con Arduino) antes de correr ensayos.")

    user_id = st.text_input("ID participante", value="estudiante", key="exp_user_id")
    store = ExperimentStore(user_id)

    c1, c2 = st.columns(2)
    with c1:
        task = st.selectbox(
            "Tarea",
            options=["nback_2", "stroop"],
            format_func=lambda x: TASK_LABELS[x],
            key="exp_task_select",
        )
    with c2:
        condition = st.selectbox(
            "Condición musical",
            options=list(EXP_MUSIC_CONDITIONS),
            format_func=lambda x: CONDITION_LABELS[x],
            key="exp_cond_select",
        )

    fixed_state = st.selectbox(
        "Estado para música fija",
        ["alta_conc", "media_conc", "baja_conc", "estres"],
        index=1,
        key="exp_fixed_state",
    )

    exp = st.session_state.get(_EXP_KEY)

    side = st.sidebar
    with side:
        st.markdown("#### Ensayo activo")
        if exp:
            _render_physio_sidebar(exp, load_state)
            if st.button("Cancelar ensayo", key="exp_cancel"):
                st.session_state.pop(_EXP_KEY, None)
                st.rerun()
        else:
            st.caption("Sin ensayo en curso.")

    if exp is None:
        n_trials = EXP_NBACK_TRIALS if task == "nback_2" else EXP_STROOP_TRIALS
        if task == "nback_2":
            with st.expander("Validación lógica 2-back (ejemplo A-B-A-F-F)"):
                for step in validate_nback_logic_demo():
                    st.markdown(
                        f"**Posición {step['position']}** → **{step['letter']}** · "
                        f"ref. 2-back: {step['reference_2back']} · "
                        f"Respuesta correcta: **{step['correct_answer']}**"
                    )
            st.caption(
                f"Secuencia: ~{EXP_NBACK_TARGET_RATE*100:.0f}% MATCH / "
                f"{(1-EXP_NBACK_TARGET_RATE)*100:.0f}% NO MATCH · "
                f"Estímulo {EXP_NBACK_STIM_SEC}s + pausa {EXP_NBACK_ISI_SEC}s"
            )
        st.info(
            f"Duración aproximada: "
            f"{n_trials * (EXP_NBACK_STIM_SEC + EXP_NBACK_ISI_SEC if task == 'nback_2' else EXP_STROOP_STIM_SEC + EXP_STROOP_ISI_SEC) / 60:.1f} min"
        )
        if st.button("▶ Iniciar ensayo", type="primary", disabled=not connected):
            send_cmd({
                "action": "experiment_condition",
                "condition": condition,
                "fixed_state": fixed_state,
            })
            st.session_state[_EXP_KEY] = {
                "task": task,
                "condition": condition,
                "user_id": user_id,
                "run_id": new_run_id(),
                "started_at": time.time(),
                "idx": 0,
                "phase": "stim",
                "stim_start": time.time(),
                "answered": False,
                "responses": [],
                "physio": [],
                "last_physio_ts": 0.0,
                "trials": (
                    generate_nback_sequence(EXP_NBACK_TRIALS, EXP_NBACK_TARGET_RATE)
                    if task == "nback_2"
                    else generate_stroop_trials(EXP_STROOP_TRIALS)
                ),
            }
            st.rerun()
    else:
        if exp["phase"] == "done" or exp.get("finished"):
            if not exp.get("saved"):
                run = _finish_run(exp, store)
                exp["saved"] = True
                exp["finished"] = True
                _render_results(run)
            if st.button("Nuevo ensayo"):
                st.session_state.pop(_EXP_KEY, None)
                st.rerun()
        else:
            if exp["task"] == "nback_2":
                _render_nback_running(exp, load_state, send_cmd)
            else:
                _render_stroop_running(exp, load_state, send_cmd)

            if exp["phase"] == "done":
                exp["finished"] = True
                st.rerun()

    st.markdown("---")
    st.markdown("#### Historial y comparativa (sin música / fija / adaptativa)")
    _render_comparison_table(store)
