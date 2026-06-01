"""
================================================================
DASHBOARD_WEB.PY — Interfaz web del Sistema Adaptativo
================================================================
Ejecutar:
    streamlit run dashboard_web.py

Requiere que main.py esté corriendo en otra terminal.
Lee state.json (escrito por main.py cada ~10 s).
Escribe sim_cmd.json para enviar comandos al simulador.

No duplica lógica del pipeline. Solo visualización + control.
================================================================
"""

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# ── Rutas compartidas con main.py ─────────────────────────────
BASE_DIR   = Path(__file__).parent
STATE_PATH = BASE_DIR / "state.json"
CMD_PATH   = BASE_DIR / "sim_cmd.json"

# ── Configuración de página ───────────────────────────────────
st.set_page_config(
    page_title="Sistema Adaptativo Musical",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS mínimo ────────────────────────────────────────────────
st.markdown("""
<style>
/* Tipografía base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Métricas más compactas */
[data-testid="metric-container"] {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 10px 14px;
}
[data-testid="metric-container"] label { font-size: 0.72rem; color: #a6adc8; }
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 1.4rem; font-weight: 700;
}

/* Tarjetas de estado cognitivo */
.state-card {
    padding: 12px 16px; border-radius: 10px;
    margin: 4px 0; font-weight: 600; font-size: 1rem;
    display: flex; justify-content: space-between; align-items: center;
}
.state-alta  { background:#1a3a2a; border:2px solid #40bf80; color:#40bf80; }
.state-media { background:#3a3520; border:2px solid #e5c347; color:#e5c347; }
.state-baja  { background:#1a2a3a; border:2px solid #4d9de0; color:#4d9de0; }
.state-estres{ background:#3a1a1a; border:2px solid #e05252; color:#e05252; }
.state-none  { background:#2a2a2a; border:2px solid #555;    color:#aaa;    }

/* Barra de progreso de buffer */
.buf-bar {
    height: 6px; border-radius: 3px;
    background: linear-gradient(90deg, #7c3aed, #06b6d4);
}

/* Recuadro de eventos */
.event-box {
    background: #1e1e2e; border: 1px solid #313244;
    border-radius: 8px; padding: 10px 14px;
    font-size: 0.8rem; color: #cdd6f4;
    max-height: 180px; overflow-y: auto;
    font-family: monospace;
}

/* Insight LLM */
.llm-box {
    background: #1e1e2e; border: 1px solid #7c3aed;
    border-radius: 8px; padding: 14px; color: #cba6f7;
    font-size: 0.85rem; line-height: 1.5;
}

/* Divider */
hr { border-color: #313244; margin: 10px 0; }

/* Ocultar menú hamburguesa */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=0.4)   # refresca cada 400 ms
def _load_state() -> dict:
    """Lee state.json. Retorna dict vacío si no existe."""
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _send_cmd(cmd: dict) -> None:
    """Escribe sim_cmd.json para que main.py lo procese."""
    cmd["ts"] = time.time()
    try:
        with open(CMD_PATH, "w", encoding="utf-8") as f:
            json.dump(cmd, f)
    except Exception:
        pass


def _state_class(state: str) -> str:
    return {
        "alta_conc":  "state-alta",
        "media_conc": "state-media",
        "baja_conc":  "state-baja",
        "estres":     "state-estres",
    }.get(state, "state-none")


def _state_emoji(state: str) -> str:
    return {
        "alta_conc":  "🟢 Alta concentración",
        "media_conc": "🟡 Concentración media",
        "baja_conc":  "🔵 Baja concentración",
        "estres":     "🔴 Estrés",
    }.get(state, f"⚪ {state}")


def _fmt_time(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m}:{s:02d}"


def _quality_color(q: float) -> str:
    if q >= 0.8: return "#40bf80"
    if q >= 0.5: return "#e5c347"
    return "#e05252"


# ─────────────────────────────────────────────────────────────
# RENDER PRINCIPAL
# ─────────────────────────────────────────────────────────────

def render() -> None:
    s = _load_state()
    demo = s.get("demo_mode", False)
    connected = bool(s)

    # ── Cabecera ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    with c1:
        st.markdown("## 🧠 Sistema Adaptativo de Recomendación Musical")
    with c2:
        mode_txt = "🟡 SIMULACIÓN" if demo else "🟢 ARDUINO"
        st.markdown(f"**{mode_txt}**")
    with c3:
        llm_txt = "🟣 LLM activo" if s.get("llm_enabled") else "⚫ LLM off"
        st.markdown(f"**{llm_txt}**")
    with c4:
        rf = s.get("rf_samples", 0)
        rf_txt = f"🤖 RF ({rf})" if rf >= 50 else f"📐 Heurística ({rf}/50)"
        st.markdown(f"**{rf_txt}**")

    if not connected:
        st.warning("⏳ Esperando datos de `main.py`... "
                   "Ejecuta `python main.py --demo` en otra terminal.")
        st.info("El dashboard se actualizará automáticamente cada vez "
                "que `main.py` exporte `state.json`.")
        st.stop()

    # Buffer progress
    buf = s.get("buffer_pct", 0)
    if buf < 100:
        st.markdown(
            f"**Llenando buffer** — {buf:.0f}%  "
            f"({s.get('windows_processed', 0)} ventanas procesadas)")
        st.progress(buf / 100)
    else:
        st.markdown(
            f"✅ Sistema activo — "
            f"**{s.get('windows_processed', 0)}** ventanas procesadas")

    st.markdown("---")

    # ── FILA 1: Fisiología + Estado cognitivo ─────────────────
    col_fisio, col_cog = st.columns([3, 2])

    with col_fisio:
        _render_physio(s)

    with col_cog:
        _render_cognitive(s)

    st.markdown("---")

    # ── FILA 2: Motor musical + LLM ───────────────────────────
    col_music, col_llm = st.columns([3, 2])

    with col_music:
        _render_music(s)

    with col_llm:
        _render_llm(s)

    st.markdown("---")

    # ── FILA 3: Simulación + Histórico ────────────────────────
    if demo:
        col_sim, col_hist = st.columns([2, 3])
        with col_sim:
            _render_simulation(s)
        with col_hist:
            _render_history(s)
    else:
        _render_history(s)

    # ── Auto-refresco ─────────────────────────────────────────
    ts = s.get("ts", 0)
    age = time.time() - ts if ts else 999
    age_color = "#40bf80" if age < 15 else "#e5c347" if age < 30 else "#e05252"
    st.markdown(
        f"<p style='font-size:0.72rem;color:{age_color};text-align:right'>"
        f"Último dato: {age:.1f}s atrás</p>",
        unsafe_allow_html=True)

    # Refresco automático cada 1 s
    time.sleep(1)
    st.rerun()


# ─────────────────────────────────────────────────────────────
# PANELES
# ─────────────────────────────────────────────────────────────

def _render_physio(s: dict) -> None:
    st.markdown("### 📡 Estado Fisiológico")

    bpm  = s.get("bpm", 0)
    rmssd= s.get("rmssd", 0)
    sdnn = s.get("sdnn", 0)
    nn50 = s.get("nn50", 0)
    pnn50= s.get("pnn50", 0)
    resp = s.get("resp_rate", 0)
    rreg = s.get("resp_reg", 0)
    scl  = s.get("scl", 0)
    scr  = s.get("scr_count", 0)
    q    = s.get("signal_quality", 0)

    # Fila 1: ECG
    c1, c2, c3, c4, c5 = st.columns(5)
    bpm_delta  = "✅" if 58 <= bpm <= 80 else ("⚠️" if bpm > 0 else "—")
    rmss_delta = "✅" if rmssd >= 30 else ("⚠️" if rmssd > 0 else "—")
    c1.metric("❤️ BPM",        f"{bpm:.1f}",   delta=bpm_delta,
              delta_color="off")
    c2.metric("📊 RMSSD †",    f"{rmssd:.1f} ms", delta=rmss_delta,
              delta_color="off")
    c3.metric("📈 SDNN †",     f"{sdnn:.1f} ms")
    c4.metric("🔢 NN50",       f"{nn50}")
    c5.metric("📉 pNN50",      f"{pnn50:.1f}%")

    # Fila 2: Resp + GSR
    c1, c2, c3, c4 = st.columns(4)
    resp_ok = "✅" if 10 <= resp <= 16 else ("⚠️" if resp > 0 else "—")
    scl_ok  = "✅" if 2 <= scl <= 8  else ("⚠️" if scl > 0  else "—")
    c1.metric("🌬️ Resp (rpm)", f"{resp:.1f}", delta=resp_ok, delta_color="off")
    c2.metric("📐 Regularidad",f"{rreg:.2f}",
              delta="✅" if rreg >= 0.7 else "⚠️", delta_color="off")
    c3.metric("⚡ GSR/SCL",    f"{scl:.2f} µS", delta=scl_ok, delta_color="off")
    c4.metric("🌊 SCR",        f"{scr}")

    # Calidad de señal
    qpct = q * 100
    qcolor = _quality_color(q)
    st.markdown(
        f"**Calidad de señal:** "
        f"<span style='color:{qcolor};font-weight:700'>{qpct:.0f}%</span>",
        unsafe_allow_html=True)
    st.progress(q)

    st.caption("† RMSSD/SDNN a 50 Hz son indicadores de tendencia, "
               "no valores clínicos (resolución ±20 ms)")


def _render_cognitive(s: dict) -> None:
    st.markdown("### 🧩 Estado Cognitivo")

    state = s.get("cog_state", "—")
    conf  = s.get("confidence", 0.0)
    method= s.get("method", "—")
    scores= s.get("class_scores", {})

    css_class = _state_class(state)
    label     = _state_emoji(state)

    st.markdown(
        f'<div class="state-card {css_class}">'
        f'<span>{label}</span>'
        f'<span>{conf*100:.1f}%</span>'
        f'</div>',
        unsafe_allow_html=True)

    st.markdown(f"**Método:** `{method}`")

    if scores:
        st.markdown("**Probabilidad por estado:**")
        STATE_ORDER = ["alta_conc", "media_conc", "baja_conc", "estres"]
        LABELS      = {
            "alta_conc":  "🟢 Alta conc.",
            "media_conc": "🟡 Media conc.",
            "baja_conc":  "🔵 Baja conc.",
            "estres":     "🔴 Estrés",
        }
        for st_key in STATE_ORDER:
            p = scores.get(st_key, 0.0)
            active = "**" if st_key == state else ""
            st.markdown(
                f"{active}{LABELS.get(st_key, st_key)}{active} — {p*100:.1f}%")
            st.progress(p)


def _render_music(s: dict) -> None:
    st.markdown("### 🎵 Motor Musical")

    track   = s.get("track_name", "—")
    cat     = s.get("track_state", "—")
    secs    = s.get("seconds_played", 0.0)
    score   = s.get("track_score", 0.0)
    rsum    = s.get("track_reward_sum", 0.0)
    plays   = s.get("track_plays", 0)
    last_rw = s.get("last_reward")
    reason  = s.get("change_reason", "—")
    epsilon = s.get("epsilon", 0.15)

    c1, c2, c3 = st.columns(3)
    c1.metric("🎶 Pista actual", track[:28] if track else "—")
    c2.metric("📁 Categoría",    cat)
    c3.metric("⏱️ Tiempo",       _fmt_time(secs))

    c1, c2, c3 = st.columns(3)
    score_color = "#40bf80" if score > 0 else "#e05252"
    c1.metric("📊 Score RL",     f"{score:+.3f}")
    c2.metric("💰 Reward acum.", f"{rsum:+.3f}")
    c3.metric("🔁 Reproducc.",   f"{plays}")

    if last_rw is not None:
        rw_delta = f"+{last_rw:.3f}" if last_rw >= 0 else f"{last_rw:.3f}"
        rw_color = "#40bf80" if last_rw >= 0 else "#e05252"
        st.markdown(
            f"**Último reward:** "
            f"<span style='color:{rw_color};font-weight:700'>{rw_delta}</span>",
            unsafe_allow_html=True)

    st.markdown(f"**ε (exploración):** {epsilon:.3f}")

    with st.expander("🔍 Motivo del último cambio", expanded=True):
        st.markdown(f"> {reason}")

    # Historial de eventos
    events = s.get("music_events", [])
    if events:
        st.markdown("**Historial de eventos:**")
        html_events = "".join(
            f"<div>{e}</div>" for e in reversed(events[-8:]))
        st.markdown(
            f'<div class="event-box">{html_events}</div>',
            unsafe_allow_html=True)


def _render_llm(s: dict) -> None:
    st.markdown("### 🤖 Análisis LLM")

    llm_ok  = s.get("llm_enabled", False)
    insight = s.get("last_insight", "")

    if not llm_ok:
        st.info("LLM desactivado. Usa `--api-key gsk_xxx` para activarlo.")
        return

    if not insight:
        st.markdown(
            '<div class="llm-box">Esperando primer análisis LLM...</div>',
            unsafe_allow_html=True)
        return

    st.markdown(
        f'<div class="llm-box">{insight}</div>',
        unsafe_allow_html=True)


def _render_simulation(s: dict) -> None:
    st.markdown("### 🎛️ Panel de Simulación")

    # ── Presets ───────────────────────────────────────────────
    st.markdown("**Presets rápidos:**")
    presets = {
        "🟢 Alta conc.":  "alta_conc",
        "🟡 Media conc.": "media_conc",
        "🔵 Baja conc.":  "baja_conc",
        "🔴 Estrés":      "estres",
        "😴 Fatiga":      "fatiga",
        "😌 Relajación":  "relajacion",
    }

    # 3 columnas de botones
    btn_items = list(presets.items())
    for row_start in range(0, len(btn_items), 3):
        cols = st.columns(3)
        for i, (label, key) in enumerate(btn_items[row_start:row_start+3]):
            if cols[i].button(label, key=f"preset_{key}", use_container_width=True):
                _send_cmd({"action": "preset", "preset": key})
                st.toast(f"Preset aplicado: {label}", icon="✅")

    st.markdown("---")

    # ── Sliders individuales ───────────────────────────────────
    st.markdown("**Ajuste fino:**")

    PARAMS = [
        ("bpm_target",   "❤️ BPM objetivo",          35.0, 180.0, 1.0),
        ("resp_rate",    "🌬️ Resp. (rpm)",             6.0,  30.0,  0.5),
        ("resp_reg",     "📐 Regularidad resp.",       0.0,   1.0,  0.05),
        ("gsr_scl",      "⚡ GSR / SCL (µS)",          0.1,  20.0,  0.5),
        ("stress_level", "😰 Nivel de estrés",         0.0,   1.0,  0.05),
        ("conc_level",   "🎯 Concentración objetivo",  0.0,   1.0,  0.05),
        ("noise_level",  "📡 Ruido de señal",          0.0,   1.0,  0.02),
    ]

    # Valores actuales (del state.json si main.py los exporta,
    # o defaults del SimulationController)
    sim_params = s.get("sim_params", {})

    # ── Sincronizar session_state con valores del JSON ─────────
    # Cuando el feedback loop mueve un parámetro en main.py, el JSON
    # lo exporta y aquí actualizamos el session_state para que el
    # slider refleje el valor real. Solo actualizamos si el usuario
    # NO está arrastrando (flag _user_dragging_<param>).
    for param, _, lo, hi, step in PARAMS:
        json_val = float(sim_params.get(param, (lo + hi) / 2))
        sk       = f"slider_{param}"
        drag_sk  = f"_drag_{param}"
        # Si no hay valor en session_state o el feedback lo cambió
        # y el usuario no está interactuando → sincronizar
        if sk not in st.session_state:
            st.session_state[sk] = json_val
        elif not st.session_state.get(drag_sk, False):
            # Actualizar solo si difiere en más de 1 step (evita jitter)
            if abs(st.session_state[sk] - json_val) > step:
                st.session_state[sk] = json_val

    # ── Renderizar sliders ────────────────────────────────────
    for param, label, lo, hi, step in PARAMS:
        sk      = f"slider_{param}"
        drag_sk = f"_drag_{param}"
        prev_val = st.session_state.get(sk, float((lo + hi) / 2))

        val = st.slider(
            label,
            min_value=float(lo),
            max_value=float(hi),
            value=float(prev_val),
            step=float(step),
            key=sk,
        )

        # Detectar movimiento manual del slider
        user_moved = abs(val - prev_val) > step * 0.05
        st.session_state[drag_sk] = user_moved

        if user_moved:
            _send_cmd({"action": "set_param", "param": param, "value": val})

    # Indicador visual del feedback loop
    track_folder = s.get("track_state", "—")
    if sim_params:
        preset = sim_params.get("preset", "custom")
        st.caption(
            f"🎵 Música activa: **{track_folder}** | "
            f"Preset: `{preset}` | "
            f"Los sliders se mueven solos a medida que la música "
            f"ajusta los parámetros fisiológicos simulados."
        )
    else:
        st.caption("Los cambios se aplican en el próximo ciclo de main.py (~1 s)")


def _render_history(s: dict) -> None:
    st.markdown("### 📈 Comportamiento y Comparativa")

    hist = s.get("session_history", [])
    if not hist:
        st.info("Datos históricos disponibles tras las primeras ventanas procesadas.")
        return

    df = pd.DataFrame(hist)
    if df.empty:
        return

    # ── Distribución de estados ───────────────────────────────
    st.markdown("**Distribución de estados (sesión actual):**")
    state_counts = df["state"].value_counts()
    total = len(df)

    STATE_COLORS_CSS = {
        "alta_conc":  "#40bf80",
        "media_conc": "#e5c347",
        "baja_conc":  "#4d9de0",
        "estres":     "#e05252",
    }
    STATE_LABELS = {
        "alta_conc":  "🟢 Alta conc.",
        "media_conc": "🟡 Media conc.",
        "baja_conc":  "🔵 Baja conc.",
        "estres":     "🔴 Estrés",
    }

    cols = st.columns(len(state_counts))
    for i, (st_key, count) in enumerate(state_counts.items()):
        pct = count / total * 100
        color = STATE_COLORS_CSS.get(st_key, "#aaa")
        cols[i].markdown(
            f"<div style='text-align:center'>"
            f"<span style='color:{color};font-size:1.5rem;font-weight:700'>"
            f"{pct:.0f}%</span><br>"
            f"<span style='font-size:0.75rem;color:#aaa'>"
            f"{STATE_LABELS.get(st_key, st_key)}</span>"
            f"</div>",
            unsafe_allow_html=True)

    st.markdown("---")

    # ── Tendencias fisiológicas de la sesión ──────────────────
    st.markdown("**Tendencias de la sesión:**")

    # Estadísticas descriptivas
    cols = st.columns(4)
    for col, (metric, label, unit) in zip(cols, [
        ("bpm",   "BPM promedio",  "bpm"),
        ("rmssd", "RMSSD medio",   "ms"),
        ("scl",   "SCL medio",     "µS"),
        ("resp",  "Resp. media",   "rpm"),
    ]):
        if metric in df.columns:
            val = df[metric].mean()
            std = df[metric].std()
            col.metric(label, f"{val:.1f} {unit}",
                       delta=f"±{std:.1f}" if std > 0 else None,
                       delta_color="off")

    # ── Comparativa con umbrales de la literatura ─────────────
    st.markdown("**Comparativa con umbrales teóricos (literatura):**")

    THRESHOLDS = [
        ("bpm",   "BPM",         58,  80,  "bpm",
         "Nourbakhsh 2012 + Task Force 1996"),
        ("rmssd", "RMSSD",       30,  None, "ms",
         "Task Force ESC/NASPE 1996 (>30 ms = tono vagal saludable)"),
        ("scl",   "GSR/SCL",     2.0, 8.0, "µS",
         "Nourbakhsh et al. 2012 (2–8 µS = zona óptima)"),
        ("resp",  "Respiración", 10,  16,  "rpm",
         "Critchley & Garfinkel 2017 (10–16 rpm = óptimo cognitivo)"),
    ]

    for metric, name, lo, hi, unit, ref in THRESHOLDS:
        if metric not in df.columns:
            continue
        val  = df[metric].mean()
        rows = []
        if lo is not None and val < lo:
            status = f"⬇️ Por debajo del óptimo ({lo} {unit})"
            color  = "#4d9de0"
        elif hi is not None and val > hi:
            status = f"⬆️ Por encima del óptimo ({hi} {unit})"
            color  = "#e05252"
        else:
            status = "✅ Dentro del rango óptimo"
            color  = "#40bf80"

        range_str = (f"{lo}–{hi}" if lo and hi
                     else f">{lo}" if lo else f"<{hi}")
        st.markdown(
            f"**{name}** — promedio `{val:.1f} {unit}` | "
            f"rango óptimo: `{range_str} {unit}` | "
            f"<span style='color:{color}'>{status}</span>  \n"
            f"<span style='color:#666;font-size:0.72rem'>{ref}</span>",
            unsafe_allow_html=True)

    # ── Tiempo en cada estado ─────────────────────────────────
    st.markdown("---")
    st.markdown("**Tiempo aproximado en cada estado:**")

    # Asumir 1 registro ≈ 5 s (ventana de 5 s step)
    WINDOW_STEP_SEC = 5
    time_per_state = {
        st_key: count * WINDOW_STEP_SEC
        for st_key, count in state_counts.items()
    }
    total_sec = sum(time_per_state.values())

    for st_key, secs in sorted(time_per_state.items(),
                                key=lambda x: x[1], reverse=True):
        mins = secs // 60
        sec_r= secs % 60
        color = STATE_COLORS_CSS.get(st_key, "#aaa")
        label = STATE_LABELS.get(st_key, st_key)
        pct   = secs / total_sec * 100 if total_sec > 0 else 0
        st.markdown(
            f"<span style='color:{color}'>{label}</span> — "
            f"**{mins}:{sec_r:02d}** ({pct:.0f}%)",
            unsafe_allow_html=True)
        st.progress(pct / 100)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    render()
