"""
================================================================
DASHBOARD.PY — Monitor ligero de texto en tiempo real
================================================================
Renderiza en terminal usando ANSI. Sin matplotlib, sin gráficas.
Refresco cada 500 ms. CPU mínimo.

Incluye Panel de Simulación interactivo (solo en modo demo):
  sliders de texto para controlar BPM, resp, GSR, estrés, etc.
  Cambios se reflejan en el SimulationController compartido.

Arquitectura:
  Dashboard corre en su propio hilo.
  Lee datos del SystemState (dataclass compartida con main.py).
  No modifica el pipeline — solo lectura + control de simulación.
================================================================
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Deque
from collections import deque

# ── Detección de plataforma para input no-bloqueante ─────────
IS_WINDOWS = sys.platform == "win32"
if IS_WINDOWS:
    import msvcrt
else:
    import select
    import tty
    import termios

# ─────────────────────────────────────────────────────────────
# ESTADO COMPARTIDO — main.py escribe, dashboard lee
# ─────────────────────────────────────────────────────────────

@dataclass
class SystemState:
    """
    Datos del sistema que el dashboard visualiza.
    main.py actualiza este objeto; dashboard lo lee.
    Thread-safe via lock.
    """
    # Señales
    bpm:          float = 0.0
    rmssd:        float = 0.0
    sdnn:         float = 0.0
    nn50:         int   = 0
    pnn50:        float = 0.0
    resp_rate:    float = 0.0
    resp_reg:     float = 0.0
    scl:          float = 0.0
    scr_count:    int   = 0
    gsr_std:      float = 0.0
    signal_quality: float = 0.0

    # Estado cognitivo
    cog_state:    str   = "—"
    confidence:   float = 0.0
    method:       str   = "—"
    class_scores: Dict[str, float] = field(default_factory=dict)

    # Música
    track_name:   str   = "—"
    track_state:  str   = "—"
    track_score:  float = 0.0
    track_reward_sum: float = 0.0
    track_plays:  int   = 0
    seconds_played: float = 0.0
    change_reason: str  = "—"
    last_reward:  Optional[float] = None

    # Sistema
    buffer_pct:   float = 0.0
    windows_processed: int = 0
    demo_mode:    bool  = False
    serial_ok:    bool  = False
    llm_enabled:  bool  = False
    rf_samples:   int   = 0
    epsilon:      float = 0.15

    # LLM
    last_insight: str   = ""

    # Eventos musicales (log circular)
    music_events: Deque[str] = field(
        default_factory=lambda: deque(maxlen=8))

    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False)

    def update(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def add_music_event(self, msg: str) -> None:
        with self._lock:
            ts = time.strftime("%H:%M:%S")
            self.music_events.append(f"[{ts}] {msg}")

    def snapshot(self) -> dict:
        with self._lock:
            return {k: v for k, v in self.__dict__.items()
                    if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────
# CONTROLADOR DE SIMULACIÓN — sliders en memoria
# ─────────────────────────────────────────────────────────────

@dataclass
class SimulationController:
    """
    Parámetros que el usuario controla desde el panel.
    SerialReader._simulate_loop los lee para generar señales.
    """
    # Parámetros directos
    bpm_target:   float = 70.0    # 35 – 180
    resp_rate:    float = 14.0    # 6  – 30
    resp_reg:     float = 0.8     # 0  – 1 (regularidad)
    gsr_scl:      float = 4.0     # 0.1 – 20 µS
    stress_level: float = 0.0     # 0  – 1 (0=ninguno, 1=máximo)
    conc_level:   float = 0.7     # 0  – 1 (nivel de concentración deseado)
    noise_level:  float = 0.1     # 0  – 1 (ruido en señales)

    # Presets activos
    _preset: str = "custom"
    _lock:   threading.Lock = field(
        default_factory=threading.Lock, repr=False)

    PRESETS = {
        "alta_conc":  dict(bpm_target=68, resp_rate=13, resp_reg=0.9,
                           gsr_scl=4.0, stress_level=0.0, conc_level=0.9,
                           noise_level=0.05),
        "media_conc": dict(bpm_target=75, resp_rate=15, resp_reg=0.75,
                           gsr_scl=5.5, stress_level=0.15, conc_level=0.6,
                           noise_level=0.1),
        "baja_conc":  dict(bpm_target=72, resp_rate=11, resp_reg=0.5,
                           gsr_scl=2.0, stress_level=0.1, conc_level=0.25,
                           noise_level=0.15),
        "estres":     dict(bpm_target=98, resp_rate=22, resp_reg=0.35,
                           gsr_scl=13.0, stress_level=0.9, conc_level=0.2,
                           noise_level=0.25),
        "fatiga":     dict(bpm_target=64, resp_rate=9, resp_reg=0.55,
                           gsr_scl=1.5, stress_level=0.2, conc_level=0.15,
                           noise_level=0.2),
        "relajacion": dict(bpm_target=60, resp_rate=8, resp_reg=0.95,
                           gsr_scl=1.0, stress_level=0.0, conc_level=0.3,
                           noise_level=0.03),
    }

    def apply_preset(self, name: str) -> bool:
        if name not in self.PRESETS:
            return False
        with self._lock:
            for k, v in self.PRESETS[name].items():
                setattr(self, k, v)
            self._preset = name
        return True

    def get(self) -> dict:
        with self._lock:
            return {
                "bpm_target":   self.bpm_target,
                "resp_rate":    self.resp_rate,
                "resp_reg":     self.resp_reg,
                "gsr_scl":      self.gsr_scl,
                "stress_level": self.stress_level,
                "conc_level":   self.conc_level,
                "noise_level":  self.noise_level,
                "preset":       self._preset,
            }

    def set_param(self, param: str, value: float) -> bool:
        limits = {
            "bpm_target":   (35, 180),
            "resp_rate":    (6, 30),
            "resp_reg":     (0, 1),
            "gsr_scl":      (0.1, 20),
            "stress_level": (0, 1),
            "conc_level":   (0, 1),
            "noise_level":  (0, 1),
        }
        if param not in limits:
            return False
        lo, hi = limits[param]
        with self._lock:
            setattr(self, param, float(max(lo, min(hi, value))))
            self._preset = "custom"
        return True


# ─────────────────────────────────────────────────────────────
# COLORES ANSI
# ─────────────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    BG_DARK= "\033[40m"

    STATE_COLORS = {
        "alta_conc":  "\033[92m",   # verde
        "media_conc": "\033[93m",   # amarillo
        "baja_conc":  "\033[94m",   # azul
        "estres":     "\033[91m",   # rojo
        "—":          "\033[2m",
    }

    @staticmethod
    def state(s: str) -> str:
        return C.STATE_COLORS.get(s, C.WHITE) + C.BOLD + s + C.RESET

    @staticmethod
    def bar(val: float, lo: float, hi: float,
            width: int = 15, good_range: tuple = None) -> str:
        """Barra ASCII de progreso coloreada."""
        pct = (val - lo) / max(hi - lo, 1e-9)
        pct = max(0.0, min(1.0, pct))
        filled = int(pct * width)
        bar    = "█" * filled + "░" * (width - filled)
        color  = C.GREEN
        if good_range:
            if lo + (hi - lo) * 0.2 <= val <= lo + (hi - lo) * 0.8:
                color = C.GREEN
            else:
                color = C.YELLOW
        return f"{color}{bar}{C.RESET}"

    @staticmethod
    def slider(val: float, lo: float, hi: float,
               width: int = 20) -> str:
        pct    = (val - lo) / max(hi - lo, 1e-9)
        pos    = int(pct * width)
        track  = ["-"] * width
        if 0 <= pos < width:
            track[pos] = "●"
        return C.CYAN + "[" + "".join(track) + "]" + C.RESET


# ─────────────────────────────────────────────────────────────
# DASHBOARD PRINCIPAL
# ─────────────────────────────────────────────────────────────

class Dashboard:
    """
    Renderiza el dashboard completo en terminal cada 500 ms.
    Corre en hilo daemon — no bloquea main.py.

    Modo interactivo (demo):
      Teclas numéricas para presets, +/- para ajustar parámetros,
      q para salir del panel de control.
    """

    def __init__(self, state: SystemState,
                 sim_ctrl: Optional[SimulationController] = None,
                 refresh_ms: int = 500):
        self._state    = state
        self._sim      = sim_ctrl
        self._refresh  = refresh_ms / 1000.0
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        self._selected_param = 0   # índice del parámetro seleccionado
        self._param_names = [
            "bpm_target", "resp_rate", "resp_reg",
            "gsr_scl", "stress_level", "conc_level", "noise_level"
        ]
        self._param_labels = [
            "BPM objetivo  ", "Resp (rpm)    ", "Regular. resp ",
            "GSR/SCL (µS)  ", "Nivel estrés  ", "Concentración ",
            "Ruido señal   "
        ]
        self._param_ranges = [
            (35, 180), (6, 30), (0, 1),
            (0.1, 20), (0, 1), (0, 1), (0, 1)
        ]
        self._param_steps = [
            1.0, 0.5, 0.05, 0.5, 0.05, 0.05, 0.02
        ]

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="dashboard")
        self._thread.start()

        if self._sim is not None:
            # Hilo de input para el panel interactivo
            t = threading.Thread(
                target=self._input_loop, daemon=True, name="dash-input")
            t.start()

    def stop(self) -> None:
        self._running = False

    # ── Loop de renderizado ───────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            try:
                self._render()
            except Exception:
                pass
            time.sleep(self._refresh)

    def _render(self) -> None:
        s   = self._state.snapshot()
        sim = self._sim.get() if self._sim else None

        lines = []
        W = 72  # ancho total

        def sep(char="─", label=""):
            if label:
                pad = (W - len(label) - 2) // 2
                lines.append(C.DIM + "─" * pad + C.RESET
                              + C.BOLD + f" {label} " + C.RESET
                              + C.DIM + "─" * pad + C.RESET)
            else:
                lines.append(C.DIM + char * W + C.RESET)

        def row(label, value, bar=None, unit=""):
            lbl = f"{C.DIM}{label:<20}{C.RESET}"
            val = f"{C.WHITE}{C.BOLD}{value}{C.RESET}{C.DIM} {unit}{C.RESET}"
            b   = f"  {bar}" if bar else ""
            lines.append(f"  {lbl}{val}{b}")

        # ── Cabecera ──────────────────────────────────────────
        lines.append("")
        mode = (f"{C.YELLOW}● SIMULACIÓN{C.RESET}"
                if s["demo_mode"] else f"{C.GREEN}● ARDUINO{C.RESET}")
        llm  = (f"{C.GREEN}LLM OK{C.RESET}"
                if s["llm_enabled"] else f"{C.DIM}LLM off{C.RESET}")
        rf   = (f"RF({s['rf_samples']})" if s["rf_samples"] > 0
                else f"{C.DIM}heurística{C.RESET}")
        lines.append(
            f"  {C.BOLD}Sistema Adaptativo de Recomendación Musical{C.RESET}"
            f"  {mode}  {llm}  {rf}")

        # Buffer de llenado (primeros 30s)
        if s["buffer_pct"] < 100:
            bw  = 30
            fil = int(s["buffer_pct"] / 100 * bw)
            bar = C.CYAN + "█" * fil + C.DIM + "░" * (bw - fil) + C.RESET
            lines.append(
                f"  {C.YELLOW}Llenando buffer:{C.RESET} [{bar}] "
                f"{s['buffer_pct']:.0f}%  ({s['windows_processed']} ventanas)")

        sep()

        # ── Estado fisiológico ────────────────────────────────
        sep(label="SEÑALES FISIOLÓGICAS")

        bpm_bar   = C.bar(s["bpm"], 40, 120, good_range=(58, 80))
        rmssd_bar = C.bar(s["rmssd"], 0, 80)
        resp_bar  = C.bar(s["resp_rate"], 5, 30, good_range=(10, 16))
        gsr_bar   = C.bar(s["scl"], 0, 15, good_range=(2, 8))
        q_bar     = C.bar(s["signal_quality"], 0, 1)

        row("BPM cardíaco",   f"{s['bpm']:5.1f}", bpm_bar,   "bpm")
        row("RMSSD †",        f"{s['rmssd']:5.1f}", rmssd_bar, "ms")
        row("SDNN †",         f"{s['sdnn']:5.1f}", bar=None,  unit="ms")
        row("NN50 / pNN50",
            f"{s['nn50']}  /  {s['pnn50']:4.1f}%")
        row("Respiración",    f"{s['resp_rate']:4.1f}", resp_bar, "rpm")
        row("Regular. resp",  f"{s['resp_reg']:.2f}",
            C.bar(s["resp_reg"], 0, 1))
        row("GSR / SCL",      f"{s['scl']:5.2f}", gsr_bar,   "µS")
        row("SCR detectados", f"{s['scr_count']}")
        row("Var. GSR",       f"{s['gsr_std']:.3f}",  unit="µS")
        row("Calidad señal",  f"{s['signal_quality']*100:.0f}%", q_bar)

        lines.append(
            f"  {C.DIM}† RMSSD/SDNN a 50 Hz = indicadores de tendencia, "
            f"no valores clínicos{C.RESET}")

        sep()

        # ── Estado cognitivo ──────────────────────────────────
        sep(label="ESTADO COGNITIVO")

        state_str = C.state(s["cog_state"])
        conf_bar  = C.bar(s["confidence"], 0, 1)
        lines.append(f"  Estado:     {state_str}")
        lines.append(
            f"  Confianza:  {C.WHITE}{s['confidence']*100:.1f}%{C.RESET}"
            f"  {conf_bar}")
        lines.append(
            f"  Método:     {C.CYAN}{s['method']}{C.RESET}")

        scores = s.get("class_scores", {})
        if scores:
            lines.append(f"  {C.DIM}Probabilidades:{C.RESET}")
            for st, prob in sorted(scores.items(),
                                   key=lambda x: x[1], reverse=True):
                pb = C.bar(prob, 0, 1, width=12)
                lines.append(
                    f"    {C.STATE_COLORS.get(st, '')}{st:<12}{C.RESET}"
                    f"  {pb}  {prob*100:.1f}%")

        sep()

        # ── Monitor musical ───────────────────────────────────
        sep(label="MOTOR MUSICAL")

        mins = int(s["seconds_played"]) // 60
        secs = int(s["seconds_played"]) % 60
        row("Pista actual",   s["track_name"])
        row("Categoría",      s["track_state"])
        row("Tiempo repr.",   f"{mins}:{secs:02d}")
        row("Score RL",       f"{s['track_score']:+.3f}",
            C.bar(s["track_score"] + 2, 0, 4, width=12))
        row("Reward acum.",   f"{s['track_reward_sum']:+.3f}")
        row("Reproducciones", f"{s['track_plays']}")
        if s["last_reward"] is not None:
            color = C.GREEN if s["last_reward"] >= 0 else C.RED
            row("Último reward",
                f"{color}{s['last_reward']:+.3f}{C.RESET}")

        lines.append(f"  {C.DIM}Motivo cambio:{C.RESET}")
        lines.append(f"    {C.YELLOW}{s['change_reason'][:65]}{C.RESET}")

        # Log de eventos musicales
        events = list(s.get("music_events", []))
        if events:
            lines.append(f"  {C.DIM}Historial de eventos:{C.RESET}")
            for ev in events[-5:]:
                lines.append(f"    {C.DIM}{ev}{C.RESET}")

        sep()

        # ── LLM insight ───────────────────────────────────────
        if s["last_insight"]:
            sep(label="ÚLTIMO INSIGHT LLM")
            insight = s["last_insight"][:200]
            # Wrap manual a 65 chars
            while insight:
                lines.append(f"  {C.MAGENTA}{insight[:65]}{C.RESET}")
                insight = insight[65:]
            sep()

        # ── Panel de simulación ───────────────────────────────
        if sim is not None:
            sep(label="PANEL DE SIMULACIÓN  [↑↓ selector | +/- valor | 1-6 preset]")

            params = self._param_names
            ctrl   = self._sim
            p      = ctrl.get()

            for i, (name, label, (lo, hi)) in enumerate(
                    zip(params, self._param_labels, self._param_ranges)):
                val     = p[name]
                sl      = C.slider(val, lo, hi, width=18)
                sel_ind = f"{C.CYAN}▶{C.RESET}" if i == self._selected_param \
                          else " "
                lines.append(
                    f"  {sel_ind} {C.WHITE}{label}{C.RESET}"
                    f"{sl}  {C.BOLD}{val:.2f}{C.RESET}  "
                    f"{C.DIM}[{lo} – {hi}]{C.RESET}")

            preset_line = "  Presets: "
            for k in SimulationController.PRESETS:
                active = p["preset"] == k
                color  = C.GREEN if active else C.DIM
                preset_line += f"{color}[{k}]{C.RESET} "
            lines.append(preset_line)

            lines.append(
                f"  {C.DIM}Teclas: ↑/↓ seleccionar | "
                f"+/= incrementar | -/_ reducir{C.RESET}")
            lines.append(
                f"  {C.DIM}1=alta_conc  2=media_conc  3=baja_conc  "
                f"4=estres  5=fatiga  6=relajacion{C.RESET}")
            sep()

        lines.append(
            f"  {C.DIM}Ctrl+C para detener el sistema{C.RESET}\n")

        # ── Render ────────────────────────────────────────────
        output = "\n".join(lines)
        # Mover cursor al inicio (sin limpiar — evita parpadeo)
        sys.stdout.write("\033[H" + output)
        sys.stdout.flush()

    # ── Input no-bloqueante ───────────────────────────────────

    def _input_loop(self) -> None:
        """Lee teclas sin bloquear. Solo activo en modo simulación."""
        if IS_WINDOWS:
            self._input_loop_windows()
        else:
            self._input_loop_unix()

    def _input_loop_windows(self) -> None:
        while self._running and self._sim:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                self._handle_key(ch)
            time.sleep(0.05)

    def _input_loop_unix(self) -> None:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self._running and self._sim:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    ch = sys.stdin.read(1)
                    self._handle_key(ch)
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _handle_key(self, ch: str) -> None:
        if not self._sim:
            return
        n = len(self._param_names)

        if ch in ("\x1b[A", "w", "W"):   # arriba
            self._selected_param = (self._selected_param - 1) % n
        elif ch in ("\x1b[B", "s", "S"): # abajo
            self._selected_param = (self._selected_param + 1) % n
        elif ch in ("+", "="):            # incrementar
            p    = self._param_names[self._selected_param]
            step = self._param_steps[self._selected_param]
            cur  = getattr(self._sim, p)
            self._sim.set_param(p, cur + step)
        elif ch in ("-", "_"):            # reducir
            p    = self._param_names[self._selected_param]
            step = self._param_steps[self._selected_param]
            cur  = getattr(self._sim, p)
            self._sim.set_param(p, cur - step)
        elif ch == "1":
            self._sim.apply_preset("alta_conc")
        elif ch == "2":
            self._sim.apply_preset("media_conc")
        elif ch == "3":
            self._sim.apply_preset("baja_conc")
        elif ch == "4":
            self._sim.apply_preset("estres")
        elif ch == "5":
            self._sim.apply_preset("fatiga")
        elif ch == "6":
            self._sim.apply_preset("relajacion")
