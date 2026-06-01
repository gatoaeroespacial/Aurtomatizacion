"""
================================================================
MAIN.PY — Orquestador del Sistema Adaptativo de Recomendación Musical
================================================================
Integra:
  SerialReader (hardware o simulación controlada)
  SignalProcessor → Classifier → MusicEngine
  LLMEngine (hilo asíncrono)
  Dashboard (hilo de visualización)
  SessionLogger (CSV persistente)
  SimulationController (panel interactivo en modo demo)

Uso:
  python main.py --demo                         # simulación
  python main.py --port COM4 --user yo          # hardware
  python main.py --demo --api-key gsk_xxx       # con LLM Groq
  python main.py --demo --calibrate 60          # calibración previa
================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import numpy as np

from config import (
    SERIAL_PORT, SERIAL_BAUD, SERIAL_TIMEOUT, SAMPLE_HZ,
    SESSION_DIR, LOG_DIR, LOG_LEVEL, LOG_FORMAT,
    LLM_INSIGHT_EVERY, WINDOW_SECONDS, CLASSIFIER_HEURISTIC_UNTIL,
    DASH_UPDATE_MS, BASE_DIR,
    MUSIC_FEEDBACK_RATE, MUSIC_FEEDBACK_EVERY,
)

# Rutas de archivos de estado compartido con dashboard_web.py
_STATE_PATH = BASE_DIR / "state.json"
_SIM_PATH   = BASE_DIR / "sim_ctrl.json"
_CMD_PATH   = BASE_DIR / "sim_cmd.json"   # comandos web → main

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(
            LOG_DIR / f"sesion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8"),
    ]
)
logger = logging.getLogger("main")

from signal_processor import SignalProcessor, RawSample
from classifier import CognitiveClassifier, ClassificationResult
from music_engine import MusicEngine
from llm_integration import LLMEngine
from dashboard import Dashboard, SystemState, SimulationController


# ─────────────────────────────────────────────────────────────
# LECTOR SERIAL
# ─────────────────────────────────────────────────────────────

class SerialReader:
    def __init__(self, port: str, baud: int,
                 sim_ctrl: Optional[SimulationController] = None):
        self._port    = port
        self._baud    = baud
        self._sim     = sim_ctrl
        self._ser     = None
        self._ok      = False
        self._running = False
        self._lock    = threading.Lock()
        self._queue:  list = []

    def start(self) -> bool:
        if self._sim is None:
            try:
                import serial
                self._ser = serial.Serial(
                    self._port, self._baud, timeout=SERIAL_TIMEOUT)
                time.sleep(2.0)
                self._ok = True
                logger.info("Serial OK: %s @ %d baud", self._port, self._baud)
            except Exception as e:
                logger.warning("Serial no disponible (%s) — SIMULACIÓN", e)
                self._ok  = False
                self._sim = SimulationController()

        self._running = True
        self._ecg_phase = 0.0
        target = self._read_loop if self._ok else self._simulate_loop
        threading.Thread(target=target, daemon=True,
                         name="serial-reader").start()
        return self._ok

    def stop(self) -> None:
        self._running = False
        if self._ser:
            try: self._ser.close()
            except Exception: pass

    def drain(self) -> list:
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
        return items

    @property
    def connected(self) -> bool:
        return self._ok

    def _read_loop(self) -> None:
        while self._running:
            try:
                raw = self._ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw or raw.startswith("#"):
                    continue
                s = self._parse(raw)
                if s:
                    with self._lock:
                        self._queue.append(s)
            except Exception as e:
                logger.debug("Serial read error: %s", e)
                time.sleep(0.1)

    def _parse(self, line: str) -> Optional[RawSample]:
        parts = line.split(",")
        if len(parts) != 6:
            return None
        try:
            return RawSample(
                ts_ms    = int(parts[0]),
                gsr_cond = float(parts[1]),
                gsr_res  = float(parts[2]),
                gsr_volt = float(parts[3]),
                air_volt = float(parts[4]),
                ecg_volt = float(parts[5]),
            )
        except ValueError:
            return None

    def _simulate_loop(self) -> None:
        t      = 0.0
        ts_ms  = 0
        ivl    = 1.0 / SAMPLE_HZ

        while self._running:
            t     += ivl
            ts_ms  = int(t * 1000)
            self._sim.tick(ivl)
            p      = self._sim.get_active()

            bpm_t      = p["bpm_target"]
            resp_t     = p["resp_rate"]
            resp_reg   = p["resp_reg"]
            gsr_base   = p["gsr_scl"]
            stress     = p["stress_level"]
            noise      = p["noise_level"]

            gsr_cond = (gsr_base
                        + stress * 4.0
                        + 0.35 * np.sin(t * 0.025)
                        + np.random.normal(0, noise * 0.25))
            gsr_cond  = max(0.05, gsr_cond)
            gsr_res   = 1.0 / (gsr_cond * 1e-6)
            gsr_volt  = min(5.0, gsr_cond * 0.05)

            irreg  = (1 - resp_reg) * 0.08 * np.random.normal(0, 1)
            air_volt = (2.5
                        + 0.35 * np.sin(2 * np.pi * (resp_t / 60) * t + irreg)
                        + np.random.normal(0, noise * 0.03))
            air_volt = float(np.clip(air_volt, 0.0, 5.0))

            # BPM efectivo con modulación HRV coherente (no ruido entre latidos)
            bpm_eff = float(np.clip(
                bpm_t + stress * 12.0 + 2.0 * np.sin(t * 0.05), 40.0, 120.0))
            hrv_frac = max(0.0, min(0.12, 0.04 * (1.0 - stress * 0.65)))
            self._ecg_phase += (bpm_eff / 60.0) * ivl * (
                1.0 + hrv_frac * np.sin(2 * np.pi * self._ecg_phase))
            phase = self._ecg_phase % 1.0
            if phase < 0.018:
                ecg_volt = 2.5 + 1.0 * np.exp(
                    -((phase - 0.009) ** 2) / 0.00008)
            elif phase < 0.12:
                ecg_volt = 2.5 - 0.06 * np.sin(
                    (phase - 0.018) * np.pi / 0.102)
            else:
                ecg_volt = 2.5 + np.random.normal(0, noise * 0.008)
            ecg_volt = float(np.clip(ecg_volt, 0.0, 5.0))

            sample = RawSample(
                ts_ms    = ts_ms,
                gsr_cond = float(gsr_cond),
                gsr_res  = float(gsr_res),
                gsr_volt = float(gsr_volt),
                air_volt = air_volt,
                ecg_volt = ecg_volt,
            )
            with self._lock:
                self._queue.append(sample)

            time.sleep(ivl)


# ─────────────────────────────────────────────────────────────
# SESSION LOGGER
# ─────────────────────────────────────────────────────────────

class SessionLogger:
    def __init__(self, user_id: str):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SESSION_DIR / f"sesion_{user_id}_{ts}.csv"
        self._f   = open(path, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._f)
        self._csv.writerow([
            "timestamp", "state", "confidence", "method",
            "bpm", "rmssd", "sdnn", "nn50", "pnn50",
            "resp_rate", "resp_reg", "scl", "scr_count", "gsr_std",
            "signal_quality", "track_name", "track_score",
        ])
        logger.info("CSV sesión: %s", path)

    def log(self, result: ClassificationResult,
            track_name: Optional[str] = None,
            track_score: Optional[float] = None) -> None:
        f = result.features
        self._csv.writerow([
            datetime.now().isoformat(),
            result.state, round(result.confidence, 4), result.method,
            round(f.bpm, 2)        if f else "",
            round(f.rmssd, 2)      if f else "",
            round(f.sdnn, 2)       if f else "",
            f.nn50                 if f else "",
            round(f.pnn50, 2)      if f else "",
            round(f.resp_rate, 2)  if f else "",
            round(f.resp_reg, 3)   if f else "",
            round(f.scl, 4)        if f else "",
            f.scr_count            if f else "",
            round(f.gsr_std, 4)    if f else "",
            round(f.signal_quality, 3) if f else "",
            track_name or "",
            round(track_score, 4)  if track_score is not None else "",
        ])
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ─────────────────────────────────────────────────────────────
# SISTEMA PRINCIPAL
# ─────────────────────────────────────────────────────────────

class AdaptiveMusicSystem:

    def __init__(self, user_id: str, serial_port: str,
                 api_key: Optional[str] = None, demo: bool = False):
        self.user_id  = user_id
        self._demo    = demo
        self._running = False

        self._sim_ctrl = SimulationController() if demo else None

        self._serial = SerialReader(
            serial_port if not demo else "DEMO",
            SERIAL_BAUD,
            sim_ctrl=self._sim_ctrl)

        self._processor  = SignalProcessor()
        self._classifier = CognitiveClassifier(user_id)
        self._music      = MusicEngine(user_id)
        self._llm        = LLMEngine(user_id)
        self._session_log = SessionLogger(user_id)

        if api_key:
            self._llm.configure(api_key)

        self._sys_state = SystemState(
            demo_mode=demo,
            llm_enabled=self._llm.enabled,
        )

        self._dashboard = Dashboard(
            state=self._sys_state,
            sim_ctrl=self._sim_ctrl,
            refresh_ms=DASH_UPDATE_MS,
        )

        self._current_state:  Optional[str] = None
        self._prev_state:     Optional[str] = None
        self._last_result:    Optional[ClassificationResult] = None
        self._windows_processed = 0
        self._session_history: list = []

        self._load_baseline()

        signal.signal(signal.SIGINT,  self._on_shutdown)
        signal.signal(signal.SIGTERM, self._on_shutdown)

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        self._dashboard.start()

        logger.info("Sistema iniciado | user=%s demo=%s llm=%s",
                    self.user_id, self._demo, self._llm.enabled)

        self._serial.start()
        self._sys_state.update(serial_ok=self._serial.connected)
        self._run_loop()

    def stop(self) -> None:
        self._running = False
        self._dashboard.stop()
        self._serial.stop()
        self._music.stop()
        self._save_baseline()
        self._classifier.save_model()

        report = self._llm.generate_session_report()
        if report:
            logger.info("Reporte sesión:\n%s", report.content[:600])

        self._session_log.close()
        logger.info("Sistema detenido correctamente.")

    # ── Loop principal ────────────────────────────────────────

    def _run_loop(self) -> None:
        ivl = 1.0 / SAMPLE_HZ

        while self._running:
            t0 = time.time()

            samples = self._serial.drain()
            for s in samples:
                self._processor.add_sample(s)

            buf_n = len(self._processor._ecg_buf)
            self._sys_state.update(
                buffer_pct=min(100.0, buf_n / 1500 * 100))

            features = self._processor.get_features()
            if features is None:
                time.sleep(max(0, ivl - (time.time() - t0)))
                continue

            result = self._classifier.classify(features)
            self._windows_processed += 1
            self._last_result = result

            state_changed = (self._current_state is not None
                             and result.state != self._current_state)
            self._prev_state    = self._current_state
            self._current_state = result.state

            self._music.update(result.state, result.confidence)
            track      = self._music.get_current_track()
            track_ts   = self._music.get_current_ts()
            track_name = track.name if track else "—"
            track_score = self._music.get_current_score() or 0.0

            if state_changed and self._prev_state:
                reward = self._music.compute_and_apply_reward(
                    self._prev_state, result.state)
                msg = (f"Cambio '{self._prev_state}' → '{result.state}' "
                       f"| reward {reward:+.3f}")
                self._sys_state.add_music_event(msg)
                logger.info("[RL] %s", msg)
                self._async_llm(
                    lambda ps=self._prev_state, ns=result.state:
                        self._llm.on_state_change(ps, ns))

            if (self._music.last_event and
                    self._music.last_event.started_at !=
                    getattr(self, "_last_event_ts", 0)):
                ev = self._music.last_event
                self._last_event_ts = ev.started_at
                self._sys_state.add_music_event(
                    f"🎵 {ev.track_name} → {ev.state_at_end or '?'} "
                    f"| {ev.change_reason[:50]}")

            self._llm.update_context(result, track_name)
            if self._windows_processed % LLM_INSIGHT_EVERY == 0:
                self._async_llm(self._llm.maybe_generate_insight)
            if result.confidence < 0.35:
                self._async_llm(
                    lambda r=result:
                        self._llm.request_label_validation(r))

            f = features
            self._sys_state.update(
                bpm=f.bpm, rmssd=f.rmssd, sdnn=f.sdnn,
                nn50=f.nn50, pnn50=f.pnn50,
                resp_rate=f.resp_rate, resp_reg=f.resp_reg,
                scl=f.scl, scr_count=f.scr_count, gsr_std=f.gsr_std,
                signal_quality=f.signal_quality,
                cog_state=result.state,
                confidence=result.confidence,
                method=result.method,
                class_scores=result.scores,
                track_name=track_name,
                track_state=self._music._current_state or "—",
                track_score=track_score,
                track_reward_sum=track_ts.reward_sum if track_ts else 0.0,
                track_plays=track_ts.play_count if track_ts else 0,
                seconds_played=self._music.seconds_played(),
                change_reason=self._music.last_change_reason,
                last_reward=track_ts.last_reward if track_ts else None,
                windows_processed=self._windows_processed,
                rf_samples=self._classifier.get_training_size(),
                epsilon=self._music._rl.epsilon,
                llm_enabled=self._llm.enabled,
            )

            self._session_log.log(result, track_name, track_score)
            f = features
            self._session_history.append({
                "ts": time.time(), "state": result.state,
                "bpm": round(f.bpm, 1), "rmssd": round(f.rmssd, 1),
                "scl": round(f.scl, 2), "resp": round(f.resp_rate, 1),
                "conf": round(result.confidence, 2),
            })
            if len(self._session_history) > 200:
                self._session_history = self._session_history[-200:]

            logger.info(
                "W%d | %s (%.2f) | BPM:%.0f RMSSD:%.0f SCL:%.2f "
                "resp:%.1f | 🎵 %s",
                self._windows_processed, result.state, result.confidence,
                f.bpm, f.rmssd, f.scl, f.resp_rate, track_name)

            if self._windows_processed % 2 == 0:
                self._export_state_json()

            if (self._sim_ctrl is not None
                    and self._windows_processed % MUSIC_FEEDBACK_EVERY == 0):
                self._apply_music_feedback()

            self._process_web_commands()

            time.sleep(max(0, ivl - (time.time() - t0)))

    # ── LLM async ─────────────────────────────────────────────

    def _async_llm(self, fn) -> None:
        def _run():
            try:
                r = fn()
                if r:
                    logger.info("[LLM] %s: %s", r.type, r.content[:100])
                    self._sys_state.update(last_insight=r.content[:300])
            except Exception as e:
                logger.debug("[LLM] Error async: %s", e)
        threading.Thread(target=_run, daemon=True).start()

    # ── Calibración ───────────────────────────────────────────

    def run_calibration(self, duration_sec: int = 120) -> None:
        logger.info("Calibración: %d s de reposo", duration_sec)
        t0 = time.time()
        self._serial.start()
        while time.time() - t0 < duration_sec:
            for s in self._serial.drain():
                self._processor.add_sample(s)
            elapsed = int(time.time() - t0)
            print(f"\r  Calibrando [{elapsed}/{duration_sec} s]", end="")
            time.sleep(0.1)
        print()
        bl = self._processor.compute_baseline(duration_sec)
        if bl:
            self._baseline = bl
            self._processor.set_baseline(bl)
            self._classifier._heuristic.baseline = bl
            self._save_baseline()
            logger.info("Baseline: %s", bl)

    def _save_baseline(self) -> None:
        bl = getattr(self, "_baseline", None)
        if not bl:
            return
        p = Path("perfiles") / f"baseline_{self.user_id}.json"
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(bl, f, indent=2)
        except Exception as e:
            logger.error("Error guardando baseline: %s", e)

    def _load_baseline(self) -> None:
        p = Path("perfiles") / f"baseline_{self.user_id}.json"
        if not p.exists():
            return
        try:
            with open(p, encoding="utf-8") as f:
                self._baseline = json.load(f)
            self._processor.set_baseline(self._baseline)
            logger.info("Baseline cargado: %s", self._baseline)
        except Exception as e:
            logger.warning("No se pudo cargar baseline: %s", e)

    def _on_shutdown(self, sig, frame) -> None:
        logger.info("Shutdown (%s)", sig)
        self._running = False

    # ── Exportación JSON para dashboard web ───────────────────

    def _export_state_json(self) -> None:
        """
        Escribe state.json (leído por dashboard_web.py).
        Escritura atómica: escribe en .tmp y luego renombra.
        Incluye estado completo del reproductor para el widget web.
        """
        try:
            snap = self._sys_state.snapshot()
            snap["music_events"] = list(snap.get("music_events", []))
            snap["ts"] = time.time()
            snap["session_history"] = self._session_history[-50:]

            # Parámetros del simulador
            if self._sim_ctrl:
                snap["sim_params"] = self._sim_ctrl.get()

            # ── Estado del reproductor para el widget web ─────
            snap["player_is_playing"]      = self._music._player.is_audible()
            snap["player_is_paused"]       = self._music._player.is_paused()
            snap["player_user_stopped"]    = self._music.user_stopped
            snap["player_volume"]          = self._music._player.current_volume
            track = self._music.get_current_track()
            snap["current_track_duration"] = (
                float(track.duration) if track and track.duration else None)
            # ─────────────────────────────────────────────────

            tmp = _STATE_PATH.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snap, f, default=str)
            tmp.replace(_STATE_PATH)
        except Exception as e:
            logger.debug("Error exportando state.json: %s", e)

    def _apply_music_feedback(self) -> None:
        if not self._sim_ctrl or self._sim_ctrl.is_manual_locked():
            return
        track = self._music.get_current_track()
        if not track:
            return

        targets = {
            "focus": {
                "bpm_target": 68.0, "resp_rate": 13.0, "resp_reg": 0.88,
                "gsr_scl": 4.0, "stress_level": 0.05,
                "conc_level": 0.88, "noise_level": 0.05,
            },
            "calm": {
                "bpm_target": 72.0, "resp_rate": 14.0, "resp_reg": 0.80,
                "gsr_scl": 5.0, "stress_level": 0.10,
                "conc_level": 0.60, "noise_level": 0.08,
            },
            "energize": {
                "bpm_target": 82.0, "resp_rate": 16.0, "resp_reg": 0.70,
                "gsr_scl": 6.5, "stress_level": 0.15,
                "conc_level": 0.55, "noise_level": 0.10,
            },
            "stress_relief": {
                "bpm_target": 63.0, "resp_rate": 9.0, "resp_reg": 0.92,
                "gsr_scl": 2.5, "stress_level": 0.05,
                "conc_level": 0.35, "noise_level": 0.04,
            },
        }

        folder = track.folder
        target = targets.get(folder)
        if not target:
            return

        rate    = MUSIC_FEEDBACK_RATE
        current = self._sim_ctrl.get()
        moved   = []
        for param, tgt_val in target.items():
            cur_val = current.get(param, tgt_val)
            if abs(cur_val - tgt_val) < 0.001:
                continue
            new_val = cur_val + rate * (tgt_val - cur_val)
            if self._sim_ctrl.set_param(param, new_val):
                moved.append(param)
        if moved:
            logger.debug("[Feedback] %s ajusta %s", folder, moved[:3])

    def _export_sim_json(self) -> None:
        if not self._sim_ctrl:
            return
        try:
            tmp = _SIM_PATH.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._sim_ctrl.get(), f)
            tmp.replace(_SIM_PATH)
        except Exception as e:
            logger.debug("Error exportando sim_ctrl.json: %s", e)

    def _process_web_commands(self) -> None:
        """
        Lee sim_cmd.json escrito por dashboard_web.py.

        Comandos de simulación:
          {"action": "preset",    "preset": "estres"}
          {"action": "set_param", "param": "bpm_target", "value": 95.0}

        Comandos del reproductor musical (nuevos):
          {"action": "player_play"}
          {"action": "player_pause"}
          {"action": "player_resume"}
          {"action": "player_stop"}
          {"action": "player_next"}              — siguiente en MISMA categoría
          {"action": "player_volume", "value": 0.8}

        Borra el archivo tras procesar para evitar re-ejecución.
        """
        if not _CMD_PATH.exists():
            return
        try:
            with open(_CMD_PATH, encoding="utf-8") as f:
                cmd = json.load(f)
            _CMD_PATH.unlink(missing_ok=True)

            action = cmd.get("action")

            # ── Comandos de simulación (existentes) ───────────
            if action == "preset" and self._sim_ctrl:
                name = cmd.get("preset", "")
                if self._sim_ctrl.apply_preset(name):
                    logger.info("[WebCmd] Preset aplicado: %s", name)
                    self._sys_state.add_music_event(
                        f"[Web] Preset → {name}")

            elif action == "set_param" and self._sim_ctrl:
                param = cmd.get("param", "")
                value = float(cmd.get("value", 0))
                if self._sim_ctrl.set_param(param, value):
                    logger.info("[WebCmd] %s = %.2f", param, value)

            # ── Comandos del reproductor (nuevos) ─────────────
            elif action == "player_play":
                self._music.clear_user_stop()
                if self._music._player.is_paused():
                    self._music.resume()
                    logger.info("[WebCmd] Player: resume")
                    self._sys_state.add_music_event("[Web] ▶ Resume")
                elif (self._music._player.current_path is not None
                      and not self._music._player.is_audible()):
                    self._music.resume()
                    logger.info("[WebCmd] Player: resume (path)")
                    self._sys_state.add_music_event("[Web] ▶ Resume")
                else:
                    state = self._current_state or "media_conc"
                    self._music.force_change(state)
                    logger.info("[WebCmd] Player: play → %s", state)
                    self._sys_state.add_music_event(
                        f"[Web] ▶ Play [{state}]")

            elif action == "player_pause":
                if self._music._player.is_audible():
                    self._music.pause()
                    logger.info("[WebCmd] Player: pause")
                    self._sys_state.add_music_event("[Web] ⏸ Pause")

            elif action == "player_resume":
                self._music.clear_user_stop()
                self._music.resume()
                logger.info("[WebCmd] Player: resume")
                self._sys_state.add_music_event("[Web] ▶ Resume")

            elif action == "player_stop":
                self._music.stop(user_initiated=True)
                logger.info("[WebCmd] Player: stop")
                self._sys_state.add_music_event("[Web] ⏹ Stop")

            elif action == "player_next":
                self._music.clear_user_stop()
                self._music.next_track_same_folder()
                track = self._music.get_current_track()
                name = track.name if track else "?"
                folder = track.folder if track else "?"
                logger.info("[WebCmd] Player: next → %s [%s]", name, folder)
                self._sys_state.add_music_event(
                    f"[Web] ⏭ Next [{folder}] → {name}")

            elif action == "player_volume":
                vol = float(cmd.get("value", 0.8))
                vol = max(0.0, min(1.0, vol))
                self._music.set_volume(vol)
                logger.info("[WebCmd] Player: volume = %.2f", vol)

            elif action == "experiment_condition":
                cond = cmd.get("condition", "musica_adaptativa")
                fixed = cmd.get("fixed_state", "media_conc")
                self._music.set_experiment_mode(cond, fixed)
                logger.info("[WebCmd] Experimento: %s (fijo=%s)", cond, fixed)
                self._sys_state.add_music_event(
                    f"[Web] Protocolo → {cond}")

        except Exception as e:
            logger.debug("Error procesando web cmd: %s", e)
            try: _CMD_PATH.unlink(missing_ok=True)
            except Exception: pass


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Sistema Adaptativo de Recomendación Musical")
    p.add_argument("--user",      default="estudiante")
    p.add_argument("--port",      default=SERIAL_PORT)
    p.add_argument("--api-key",   default=os.getenv("GROQ_API_KEY", ""),
                   help="Groq API key — gratis en console.groq.com")
    p.add_argument("--demo",      action="store_true",
                   help="Modo simulación con panel interactivo")
    p.add_argument("--calibrate", type=int, default=0, metavar="SEG")
    args = p.parse_args()

    system = AdaptiveMusicSystem(
        user_id    =args.user,
        serial_port=args.port,
        api_key    =args.api_key or None,
        demo       =args.demo,
    )

    if args.calibrate > 0:
        system.run_calibration(args.calibrate)

    try:
        system.start()
    except KeyboardInterrupt:
        pass
    finally:
        system.stop()


if __name__ == "__main__":
    main()
