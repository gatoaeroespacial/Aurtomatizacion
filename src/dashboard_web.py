"""
================================================================
DASHBOARD_WEB.PY — Interfaz web del Sistema Adaptativo
================================================================
Ejecutar:
    streamlit run dashboard_web.py

Requiere que main.py esté corriendo en otra terminal.
Lee state.json (escrito por main.py cada ~10 s).
Escribe sim_cmd.json para enviar comandos al simulador y al
reproductor musical integrado.

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

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

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

.player-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
.player-track-category {
    font-size: 0.72rem; color: #a6adc8;
    letter-spacing: 0.06em; margin-bottom: 5px;
}
.player-track-name {
    font-size: 1.05rem; font-weight: 600; color: #cdd6f4;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.player-status-playing { color: #40bf80; font-size: 0.75rem; margin-top: 8px; }
.player-status-paused  { color: #e5c347; font-size: 0.75rem; margin-top: 8px; }
.player-status-stopped { color: #888;    font-size: 0.75rem; margin-top: 8px; }

.event-box {
    background: #1e1e2e; border: 1px solid #313244;
    border-radius: 8px; padding: 10px 14px;
    font-size: 0.8rem; color: #cdd6f4;
    max-height: 180px; overflow-y: auto;
    font-family: monospace;
}

.llm-box {
    background: #1e1e2e; border: 1px solid #7c3aed;
    border-radius: 8px; padding: 14px; color: #cba6f7;
    font-size: 0.85rem; line-height: 1.5;
}

hr { border-color: #313244; margin: 10px 0; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=0.4)
def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _send_cmd(cmd: dict) -> None:
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
    m, s = divmod(int(max(0, secs)), 60)
    return f"{m}:{s:02d}"


def _quality_color(q: float) -> str:
    if q >= 0.8: return "#40bf80"
    if q >= 0.5: return "#e5c347"
    return "#e05252"


# ─────────────────────────────────────────────────────────────
# REPRODUCTOR INTEGRADO
# ─────────────────────────────────────────────────────────────

def _render_player(s: dict) -> None:
    """
    Reproductor musical integrado con el MusicEngine real.
    Lee estado desde state.json y envía comandos via sim_cmd.json.
    No crea reproductores paralelos — controla el AudioPlayer existente.

    Comportamiento de cada control:
      Play    → resume si estaba pausado, o inicia en la categoría actual
      Pause   → pausa sin perder posición
      Stop    → detiene completamente
      Siguiente → cambia dentro de la MISMA categoría activa
      Volumen → ajuste en tiempo real sobre el mixer de pygame
    """
    st.markdown("### 🎵 Reproductor Musical")

    track_name   = s.get("track_name", "—")
    track_state  = s.get("track_state", "—")
    secs_played  = float(s.get("seconds_played", 0.0))
    duration     = s.get("current_track_duration")    # None si librosa no disponible
    is_playing   = bool(s.get("player_is_playing", False))
    is_paused    = bool(s.get("player_is_paused", False))
    user_stopped = bool(s.get("player_user_stopped", False))
    volume       = float(s.get("player_volume", 0.8))
    cog_state    = s.get("cog_state", "—")

    FOLDER_LABELS = {
        "focus":        "🎯 Focus",
        "calm":         "🌊 Calm",
        "energize":     "⚡ Energize",
        "stress_relief":"🍃 Stress Relief",
        # aliases por si track_state viene como estado cognitivo
        "alta_conc":    "🎯 Focus",
        "media_conc":   "🌊 Calm",
        "baja_conc":    "⚡ Energize",
        "estres":       "🍃 Stress Relief",
    }
    COG_LABELS = {
        "alta_conc":  "🟢 Alta conc.",
        "media_conc": "🟡 Media conc.",
        "baja_conc":  "🔵 Baja conc.",
        "estres":     "🔴 Estrés",
    }

    folder_label = FOLDER_LABELS.get(track_state, track_state)
    cog_label    = COG_LABELS.get(cog_state, cog_state)
    no_track     = (track_name in ("—", "", None))

    # ── Tarjeta de pista actual ───────────────────────────────
    track_display = f"🎵 {track_name}" if not no_track else "⏹ Sin reproducción activa"
    st.markdown(
        f"""
        <div class="player-card">
            <div class="player-track-category">
                {folder_label} &nbsp;·&nbsp; Estado cognitivo: {cog_label}
            </div>
            <div class="player-track-name">{track_display}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Barra de progreso + tiempos ───────────────────────────
    if duration and duration > 0 and not no_track:
        progress_pct = min(1.0, secs_played / duration)
        st.progress(progress_pct)
        t1, t2 = st.columns([1, 1])
        t1.markdown(
            f"<span style='font-size:0.8rem;color:#a6adc8'>"
            f"{_fmt_time(secs_played)}</span>",
            unsafe_allow_html=True)
        t2.markdown(
            f"<span style='font-size:0.8rem;color:#a6adc8;float:right'>"
            f"{_fmt_time(duration)}</span>",
            unsafe_allow_html=True)
    elif not no_track:
        # Sin duración conocida: mostrar solo tiempo transcurrido
        st.progress(0.0)
        st.markdown(
            f"<span style='font-size:0.8rem;color:#a6adc8'>"
            f"{_fmt_time(secs_played)} &nbsp;(duración desconocida — "
            f"instala librosa para detectarla)</span>",
            unsafe_allow_html=True)
    else:
        st.progress(0.0)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Botones de control ────────────────────────────────────
    c_play, c_pause, c_stop, c_next = st.columns(4)

    can_pause = is_playing and not is_paused
    can_resume = is_paused and not no_track
    can_play = user_stopped or (no_track and not is_playing) or can_resume

    if is_playing and not is_paused:
        play_label, play_disabled = "▶ Reproduciendo", True
    elif can_resume:
        play_label, play_disabled = "▶ Reanudar", False
    else:
        play_label, play_disabled = "▶ Play", not can_play

    if c_play.button(play_label, key="btn_player_play",
                     use_container_width=True, disabled=play_disabled):
        _send_cmd({"action": "player_play"})
        st.toast("Iniciando reproducción...", icon="▶️")

    if c_pause.button("⏸ Pause", key="btn_player_pause",
                      use_container_width=True,
                      disabled=no_track or not can_pause):
        _send_cmd({"action": "player_pause"})
        st.toast("Pausado", icon="⏸️")

    if c_stop.button("⏹ Stop", key="btn_player_stop",
                     use_container_width=True,
                     disabled=no_track and not is_playing and not is_paused):
        _send_cmd({"action": "player_stop"})
        st.toast("Reproducción detenida", icon="⏹️")

    next_help = (
        f"Cambia a otra canción dentro de [{folder_label}] "
        f"(misma carpeta MP3)."
    )
    if c_next.button("⏭ Siguiente", key="btn_player_next",
                     use_container_width=True,
                     disabled=no_track,
                     help=next_help):
        _send_cmd({"action": "player_next"})
        st.toast(f"Siguiente en {folder_label}...", icon="⏭️")

    # ── Control de volumen ─────────────────────────────────────
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    v1, v2 = st.columns([1, 5])
    v1.markdown(
        "<span style='font-size:0.85rem;color:#a6adc8;"
        "line-height:38px;display:inline-block'>🔊 Vol</span>",
        unsafe_allow_html=True)

    vol_key  = "player_volume_slider"
    vol_lock = "_volume_user_lock"

    if vol_key not in st.session_state:
        st.session_state[vol_key] = volume
    if vol_lock not in st.session_state:
        st.session_state[vol_lock] = False
    if not st.session_state[vol_lock]:
        if abs(st.session_state[vol_key] - volume) > 0.04:
            st.session_state[vol_key] = volume

    def _on_volume_change() -> None:
        st.session_state[vol_lock] = True
        _send_cmd({
            "action": "player_volume",
            "value": round(float(st.session_state[vol_key]), 2),
        })

    v2.slider(
        "Volumen",
        min_value=0.0, max_value=1.0,
        step=0.05,
        key=vol_key,
        label_visibility="collapsed",
        on_change=_on_volume_change,
    )

    # ── Indicador de estado ───────────────────────────────────
    if is_playing:
        status_css   = "player-status-playing"
        status_label = "● Reproduciendo"
    elif is_paused:
        status_css   = "player-status-paused"
        status_label = "● Pausado"
    elif user_stopped:
        status_css   = "player-status-stopped"
        status_label = "● Detenido (usuario)"
    else:
        status_css   = "player-status-stopped"
        status_label = "● Detenido"

    st.markdown(
        f"<p class='{status_css}'>{status_label}</p>",
        unsafe_allow_html=True)


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

    # ── FILA 2: Reproductor + LLM ─────────────────────────────
    col_music, col_llm = st.columns([3, 2])

    with col_music:
        _render_player(s)          # Reproductor integrado (nuevo)
        st.markdown("---")
        _render_music(s)           # Métricas RL existentes (sin cambios)

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

    ts = s.get("ts", 0)
    age = time.time() - ts if ts else 999
    age_color = "#40bf80" if age < 15 else "#e5c347" if age < 30 else "#e05252"
    st.markdown(
        f"<p style='font-size:0.72rem;color:{age_color};text-align:right'>"
        f"Último dato: {age:.1f}s atrás</p>",
        unsafe_allow_html=True)

    time.sleep(1)
    st.rerun()


# ─────────────────────────────────────────────────────────────
# PANELES (sin cambios respecto al original)
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

    c1, c2, c3, c4, c5 = st.columns(5)
    bpm_delta  = "✅" if 58 <= bpm <= 80 else ("⚠️" if bpm > 0 else "—")
    rmss_delta = "✅" if rmssd >= 30 else ("⚠️" if rmssd > 0 else "—")
    c1.metric("❤️ BPM",        f"{bpm:.1f}",   delta=bpm_delta,  delta_color="off")
    c2.metric("📊 RMSSD †",    f"{rmssd:.1f} ms", delta=rmss_delta, delta_color="off")
    c3.metric("📈 SDNN †",     f"{sdnn:.1f} ms")
    c4.metric("🔢 NN50",       f"{nn50}")
    c5.metric("📉 pNN50",      f"{pnn50:.1f}%")

    c1, c2, c3, c4 = st.columns(4)
    resp_ok = "✅" if 10 <= resp <= 16 else ("⚠️" if resp > 0 else "—")
    scl_ok  = "✅" if 2 <= scl <= 8  else ("⚠️" if scl > 0  else "—")
    c1.metric("🌬️ Resp (rpm)", f"{resp:.1f}", delta=resp_ok, delta_color="off")
    c2.metric("📐 Regularidad",f"{rreg:.2f}",
              delta="✅" if rreg >= 0.7 else "⚠️", delta_color="off")
    c3.metric("⚡ GSR/SCL",    f"{scl:.2f} µS", delta=scl_ok, delta_color="off")
    c4.metric("🌊 SCR",        f"{scr}")

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
    st.markdown("### 📊 Métricas del Motor RL")

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
    c1.metric("📊 Score RL",     f"{score:+.3f}")
    c2.metric("💰 Reward acum.", f"{rsum:+.3f}")
    c3.metric("🔁 Reproducc.",   f"{plays}")

    if last_rw is not None:
        rw_color = "#40bf80" if last_rw >= 0 else "#e05252"
        st.markdown(
            f"**Último reward:** "
            f"<span style='color:{rw_color};font-weight:700'>{last_rw:+.3f}</span>",
            unsafe_allow_html=True)

    st.markdown(f"**ε (exploración):** {epsilon:.3f}")

    with st.expander("🔍 Motivo del último cambio", expanded=False):
        st.markdown(f"> {reason}")

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


_SIM_LOCK_KEY = "_sim_sliders_user_lock"


def _on_sim_param_change(param: str) -> None:
    st.session_state[_SIM_LOCK_KEY] = True
    sk = f"slider_{param}"
    _send_cmd({
        "action": "set_param",
        "param": param,
        "value": float(st.session_state[sk]),
    })


def _render_simulation(s: dict) -> None:
    st.markdown("### 🎛️ Panel de Simulación")

    if _SIM_LOCK_KEY not in st.session_state:
        st.session_state[_SIM_LOCK_KEY] = False

    st.markdown("**Presets rápidos:**")
    presets = {
        "🟢 Alta conc.":  "alta_conc",
        "🟡 Media conc.": "media_conc",
        "🔵 Baja conc.":  "baja_conc",
        "🔴 Estrés":      "estres",
        "😴 Fatiga":      "fatiga",
        "😌 Relajación":  "relajacion",
    }

    btn_items = list(presets.items())
    for row_start in range(0, len(btn_items), 3):
        cols = st.columns(3)
        for i, (label, key) in enumerate(btn_items[row_start:row_start+3]):
            if cols[i].button(label, key=f"preset_{key}", use_container_width=True):
                _send_cmd({"action": "preset", "preset": key})
                st.session_state[_SIM_LOCK_KEY] = False
                st.toast(f"Preset aplicado: {label}", icon="✅")

    st.markdown("---")

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

    sim_params = s.get("sim_params", {})

    for param, _, lo, hi, step in PARAMS:
        sk = f"slider_{param}"
        json_val = float(sim_params.get(param, (lo + hi) / 2))
        if sk not in st.session_state:
            st.session_state[sk] = json_val
        elif not st.session_state[_SIM_LOCK_KEY]:
            if abs(st.session_state[sk] - json_val) > step * 0.5:
                st.session_state[sk] = json_val

    for param, label, lo, hi, step in PARAMS:
        sk = f"slider_{param}"
        st.slider(
            label,
            min_value=float(lo), max_value=float(hi),
            step=float(step),
            key=sk,
            on_change=_on_sim_param_change,
            args=(param,),
        )

    track_folder = s.get("track_state", "—")
    if sim_params:
        preset = sim_params.get("preset", "custom")
        lock_note = " (ajuste manual activo)" if st.session_state[_SIM_LOCK_KEY] else ""
        st.caption(
            f"🎵 Música: **{track_folder}** | Preset: `{preset}`{lock_note}. "
            f"Objetivos convergen en ~12 s; métricas mostradas son calculadas."
        )


def _render_history(s: dict) -> None:
    st.markdown("### 📈 Comportamiento y Comparativa")

    hist = s.get("session_history", [])
    if not hist:
        st.info("Datos históricos disponibles tras las primeras ventanas procesadas.")
        return

    df = pd.DataFrame(hist)
    if df.empty:
        return

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
    st.markdown("**Tendencias de la sesión:**")

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

    st.markdown("**Comparativa con umbrales teóricos (literatura):**")

    THRESHOLDS = [
        ("bpm",   "BPM",         58,  80,  "bpm",  "Nourbakhsh 2012 + Task Force 1996"),
        ("rmssd", "RMSSD",       30,  None,"ms",   "Task Force ESC/NASPE 1996 (>30 ms)"),
        ("scl",   "GSR/SCL",     2.0, 8.0, "µS",   "Nourbakhsh et al. 2012 (2–8 µS)"),
        ("resp",  "Respiración", 10,  16,  "rpm",  "Critchley & Garfinkel 2017 (10–16 rpm)"),
    ]

    for metric, name, lo, hi, unit, ref in THRESHOLDS:
        if metric not in df.columns:
            continue
        val = df[metric].mean()
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

    st.markdown("---")
    st.markdown("**Tiempo aproximado en cada estado:**")

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
