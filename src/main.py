"""
================================================================
MAIN.PY — Orquestador del Sistema Adaptativo de Recomendación Musical
================================================================
Pipeline completo:
  Serial (Arduino) → SignalProcessor → Classifier → MusicEngine
                   ↘ LLMEngine (análisis asíncrono)
                   ↘ Dashboard (visualización)
                   ↘ SessionLogger (persistencia)

Uso:
  python main.py [--user usuario] [--api-key sk-ant-xxx] [--port COM4]
  python main.py --demo                (modo simulación sin hardware)

El sistema arranca en modo simulación si el serial falla.
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
from typing import Optional, Dict, Any

import numpy as np

# ── Configurar logging antes de importar módulos propios ─────
from config import (
    SERIAL_PORT, SERIAL_BAUD, SERIAL_TIMEOUT, SAMPLE_HZ,
    SESSION_DIR, LOG_DIR, LOG_LEVEL, LOG_FORMAT,
    LLM_INSIGHT_EVERY, RL_EVAL_DELAY_SEC, WINDOW_SECONDS,
    CLASSIFIER_HEURISTIC_UNTIL,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / f"sesion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8"
        ),
    ]
)
logger = logging.getLogger("main")

from signal_processor import SignalProcessor, RawSample, SignalFeatures
from classifier import CognitiveClassifier, ClassificationResult
from music_engine import MusicEngine
from llm_integration import LLMEngine


# ─────────────────────────────────────────────────────────────
# LECTOR SERIAL
# ─────────────────────────────────────────────────────────────

class SerialReader:
    """
    Lee líneas CSV del Arduino en un hilo separado.
    Formato: ts_ms,gsr_cond,gsr_res,gsr_volt,air_volt,ecg_volt
    """

    def __init__(self, port: str, baud: int):
        self._port   = port
        self._baud   = baud
        self._ser    = None
        self._ok     = False
        self._running = False
        self._lock   = threading.Lock()
        self._queue: list = []   # cola de muestras sin procesar
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        try:
            import serial
            self._ser = serial.Serial(
                self._port, self._baud,
                timeout=SERIAL_TIMEOUT
            )
            time.sleep(2.0)   # Espera reset del Arduino
            self._ok = True
            logger.info("Serial OK: %s @ %d baud", self._port, self._baud)
        except Exception as e:
            logger.warning("Serial no disponible (%s) — modo SIMULACIÓN", e)
            self._ok = False

        self._running = True
        target = self._read_loop if self._ok else self._simulate_loop
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()
        return self._ok

    def stop(self) -> None:
        self._running = False
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass

    def drain(self) -> list:
        """Vacía y retorna la cola de muestras pendientes."""
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
        return items

    @property
    def connected(self) -> bool:
        return self._ok

    # ── Lectura real ──────────────────────────────────────────
    def _read_loop(self) -> None:
        while self._running:
            try:
                raw = self._ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw or raw.startswith("#"):
                    continue
                sample = self._parse_line(raw)
                if sample:
                    with self._lock:
                        self._queue.append(sample)
            except Exception as e:
                logger.debug("Serial read error: %s", e)
                time.sleep(0.1)

    def _parse_line(self, line: str) -> Optional[RawSample]:
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

    # ── Simulación ────────────────────────────────────────────
    def _simulate_loop(self) -> None:
        """
        Genera señales sintéticas realistas para pruebas sin hardware.
        Ciclo de trabajo: 90s concentración alta, 30s estrés, 60s media.
        """
        t = 0.0
        interval = 1.0 / SAMPLE_HZ
        ts_ms = 0

        while self._running:
            t    += interval
            ts_ms = int(t * 1000)

            # Ciclo de estado simulado (~3 min por ciclo)
            cycle = t % 180
            if cycle < 90:
                # Alta concentración
                gsr_cond = 3.5 + 0.5 * np.sin(t * 0.02) + np.random.normal(0, 0.05)
                bpm_base = 68
                resp_amp = 0.3
                resp_freq = 13 / 60  # 13 rpm
            elif cycle < 120:
                # Estrés
                gsr_cond = 10.0 + 1.5 * np.sin(t * 0.05) + np.random.normal(0, 0.2)
                bpm_base = 95
                resp_amp = 0.2
                resp_freq = 20 / 60  # 20 rpm
            else:
                # Concentración media
                gsr_cond = 5.0 + 0.8 * np.sin(t * 0.03) + np.random.normal(0, 0.08)
                bpm_base = 75
                resp_amp = 0.35
                resp_freq = 15 / 60

            gsr_cond = max(0.1, gsr_cond)
            gsr_res  = 1.0 / (gsr_cond * 1e-6) if gsr_cond > 0 else 1e6
            gsr_volt = gsr_cond * 0.05  # aproximación lineal

            # Airflow: onda sinusoidal respiratoria
            air_volt = 2.5 + resp_amp * np.sin(2 * np.pi * resp_freq * t)

            # ECG sintético con latidos a ~bpm_base
            bpm_var = bpm_base + 3 * np.sin(t * 0.1)
            phase   = (t * bpm_var / 60) % 1.0
            if phase < 0.03:
                ecg_volt = 2.5 + 1.2 * np.exp(-((phase - 0.015) ** 2) / 0.0001)
            elif phase < 0.12:
                ecg_volt = 2.5 - 0.1 * np.sin((phase - 0.03) * np.pi / 0.09)
            else:
                ecg_volt = 2.5 + np.random.normal(0, 0.005)

            sample = RawSample(
                ts_ms    = ts_ms,
                gsr_cond = gsr_cond,
                gsr_res  = gsr_res,
                gsr_volt = gsr_volt,
                air_volt = float(air_volt),
                ecg_volt = float(ecg_volt),
            )
            with self._lock:
                self._queue.append(sample)

            time.sleep(interval)


# ─────────────────────────────────────────────────────────────
# LOGGER DE SESIÓN
# ─────────────────────────────────────────────────────────────

class SessionLogger:
    """Persiste todas las clasificaciones y features en CSV."""

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
        logger.info("Sesión guardada en: %s", path)

    def log(self, result: ClassificationResult,
            track_name: Optional[str] = None,
            track_score: Optional[float] = None) -> None:
        f = result.features
        self._csv.writerow([
            datetime.now().isoformat(),
            result.state, round(result.confidence, 4), result.method,
            round(f.bpm, 2)       if f else "",
            round(f.rmssd, 2)     if f else "",
            round(f.sdnn, 2)      if f else "",
            f.nn50                if f else "",
            round(f.pnn50, 2)     if f else "",
            round(f.resp_rate, 2) if f else "",
            round(f.resp_reg, 3)  if f else "",
            round(f.scl, 4)       if f else "",
            f.scr_count           if f else "",
            round(f.gsr_std, 4)   if f else "",
            round(f.signal_quality, 3) if f else "",
            track_name or "",
            round(track_score, 4) if track_score is not None else "",
        ])
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ─────────────────────────────────────────────────────────────
# SISTEMA PRINCIPAL
# ─────────────────────────────────────────────────────────────

class AdaptiveMusicSystem:
    """
    Orquestador de todo el pipeline.
    """

    def __init__(self, user_id: str, serial_port: str,
                 api_key: Optional[str] = None, demo: bool = False):
        self.user_id = user_id
        self._demo   = demo
        self._running = False

        # Módulos
        self._serial      = SerialReader(
            "DEMO" if demo else serial_port, SERIAL_BAUD)
        self._processor   = SignalProcessor()
        self._classifier  = CognitiveClassifier(user_id)
        self._music       = MusicEngine(user_id)
        self._llm         = LLMEngine(user_id)
        self._session_log = SessionLogger(user_id)

        if api_key:
            self._llm.configure(api_key)

        # Estado del sistema
        self._current_state:  Optional[str] = None
        self._prev_state:     Optional[str] = None
        self._last_result:    Optional[ClassificationResult] = None
        self._windows_processed = 0
        self._state_changed_at  = 0.0

        # Calibración
        self._calibrated = False
        self._baseline:  Optional[Dict] = None

        self._load_baseline()

        # Shutdown handler
        signal.signal(signal.SIGINT,  self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        logger.info("=" * 60)
        logger.info("Sistema Adaptativo de Recomendación Musical")
        logger.info("Usuario: %s | Demo: %s | LLM: %s",
                    self.user_id, self._demo, self._llm.enabled)
        logger.info("Biblioteca musical: %s", self._music.get_library_summary())
        logger.info("=" * 60)

        if self._demo:
            logger.info("Modo DEMO activo — señales simuladas")

        self._serial.start()
        self._run_loop()

    def stop(self) -> None:
        self._running = False
        self._serial.stop()
        self._music.stop()
        self._save_baseline()
        self._classifier.save_model()

        # Reporte final de sesión
        logger.info("Generando reporte de sesión...")
        report = self._llm.generate_session_report()
        if report:
            logger.info("\n%s\n", report.content[:500])

        self._session_log.close()
        logger.info("Sistema detenido.")

    # ── Loop principal ────────────────────────────────────────

    def _run_loop(self) -> None:
        """
        Loop principal a ~50 Hz.
        1. Drain de la cola serial
        2. Agregar muestras al procesador
        3. Extraer features si la ventana está lista
        4. Clasificar estado cognitivo
        5. Actualizar motor musical
        6. LLM (asíncrono, cada N ventanas)
        7. Logging
        """
        loop_interval = 1.0 / SAMPLE_HZ

        while self._running:
            t0 = time.time()

            # 1. Procesar todas las muestras en cola
            samples = self._serial.drain()
            for sample in samples:
                self._processor.add_sample(sample)

            # 2. Extraer features (solo si ventana completa)
            features = self._processor.get_features()
            if features is None:
                elapsed = time.time() - t0
                time.sleep(max(0, loop_interval - elapsed))
                self._print_progress()
                continue

            # 3. Clasificar
            result = self._classifier.classify(features)
            self._windows_processed += 1
            self._last_result = result

            # 4. Detectar cambio de estado
            state_changed = (self._current_state is not None
                             and result.state != self._current_state)
            self._prev_state    = self._current_state
            self._current_state = result.state

            # 5. Motor musical
            track_tid = self._music.update(result.state, result.confidence)
            track     = self._music.get_current_track()
            track_name  = track.name  if track else None
            track_score = self._music.get_current_score()

            # 5b. Calcular reward si cambió de estado (evaluación RL)
            if state_changed and self._prev_state:
                reward = self._music.compute_and_apply_reward(
                    self._prev_state, result.state)
                logger.info("RL Reward: %.3f (%s → %s)",
                            reward, self._prev_state, result.state)

            # 6. LLM — asíncrono para no bloquear el loop
            if state_changed and self._prev_state:
                self._async_llm(
                    lambda: self._llm.on_state_change(
                        self._prev_state, result.state))

            self._llm.update_context(result, track_name)
            if self._windows_processed % LLM_INSIGHT_EVERY == 0:
                self._async_llm(self._llm.maybe_generate_insight)

            # Validación de baja confianza
            if result.confidence < 0.35:
                self._async_llm(
                    lambda: self._llm.request_label_validation(result))

            # 7. Logging
            self._session_log.log(result, track_name, track_score)
            self._log_state(result, track_name)

            elapsed = time.time() - t0
            time.sleep(max(0, loop_interval - elapsed))

    def _print_progress(self) -> None:
        """Muestra progreso de llenado del buffer (primeros 30 s)."""
        n = len(self._processor._ecg_buf)
        total = 1500  # WINDOW_SAMPLES
        if n < total and n % 50 == 0:
            pct = int(n / total * 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  Buffer: [{bar}] {pct}% ({n}/{total} muestras)  ",
                  end="", flush=True)
        elif n >= total:
            print("\r" + " " * 60 + "\r", end="", flush=True)

    def _log_state(self, result: ClassificationResult,
                   track_name: Optional[str]) -> None:
        sensor = self._processor.get_sensor_status()
        rf_n   = self._classifier.get_training_size()
        method = result.method
        if rf_n < CLASSIFIER_HEURISTIC_UNTIL:
            method += f" (RF en {CLASSIFIER_HEURISTIC_UNTIL - rf_n} muestras)"

        logger.info(
            "Estado: %-12s | Conf: %.2f | %s | "
            "BPM: %4.0f | RMSSD: %4.0f | SCL: %.2f µS | "
            "Resp: %.1f rpm | 🎵 %s",
            result.state, result.confidence, method,
            result.features.bpm    if result.features else 0,
            result.features.rmssd  if result.features else 0,
            result.features.scl    if result.features else 0,
            result.features.resp_rate if result.features else 0,
            track_name or "—",
        )

        # Warnings de sensores
        s = sensor
        if not s.ecg_ok:     logger.warning("⚠ ECG sin señal — verifica electrodos")
        if not s.gsr_ok:     logger.warning("⚠ GSR sin contacto dérmico")
        if not s.airflow_ok: logger.warning("⚠ Airflow sin variación — verifica sensor")

    # ── Calibración ───────────────────────────────────────────

    def run_calibration(self, duration_sec: int = 120) -> None:
        """
        Sesión de calibración: usuario en reposo durante `duration_sec`.
        Llama a este método antes del loop principal para mejores resultados.
        """
        logger.info("Iniciando calibración — reposo %d s", duration_sec)
        t0 = time.time()
        self._serial.start()

        while time.time() - t0 < duration_sec and self._running:
            samples = self._serial.drain()
            for s in samples:
                self._processor.add_sample(s)
            elapsed = int(time.time() - t0)
            print(f"\r  Calibrando: {elapsed}/{duration_sec} s", end="")
            time.sleep(0.1)

        print()
        baseline = self._processor.compute_baseline(duration_sec)
        if baseline:
            self._baseline = baseline
            self._processor.set_baseline(baseline)
            self._classifier._heuristic.baseline = baseline
            self._save_baseline()
            logger.info("Calibración completada: %s", baseline)
        else:
            logger.warning("Calibración insuficiente — se usarán umbrales estándar")

    def _save_baseline(self) -> None:
        if not self._baseline:
            return
        path = Path("perfiles") / f"baseline_{self.user_id}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._baseline, f, indent=2)
        except Exception as e:
            logger.error("Error guardando baseline: %s", e)

    def _load_baseline(self) -> None:
        path = Path("perfiles") / f"baseline_{self.user_id}.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    self._baseline = json.load(f)
                self._processor.set_baseline(self._baseline)
                logger.info("Baseline cargado: %s", self._baseline)
            except Exception as e:
                logger.warning("No se pudo cargar baseline: %s", e)

    # ── Async LLM ─────────────────────────────────────────────

    def _async_llm(self, fn) -> None:
        """Ejecuta una función LLM en un hilo separado."""
        t = threading.Thread(target=self._safe_call, args=(fn,), daemon=True)
        t.start()

    @staticmethod
    def _safe_call(fn) -> None:
        try:
            result = fn()
            if result:
                logger.info("[LLM] %s: %s",
                            result.type, result.content[:120])
        except Exception as e:
            logger.debug("LLM async error: %s", e)

    # ── Shutdown ──────────────────────────────────────────────

    def _shutdown_handler(self, sig, frame) -> None:
        logger.info("Señal de cierre recibida (%s)", sig)
        self._running = False


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sistema Adaptativo de Recomendación Musical")
    p.add_argument("--user",    default="estudiante", help="ID de usuario")
    p.add_argument("--port",    default=SERIAL_PORT,  help="Puerto serial")
    p.add_argument("--api-key", default=os.getenv("GROQ_API_KEY", ""),
                   help="Groq API key (o env GROQ_API_KEY) — gratis en console.groq.com")
    p.add_argument("--demo",    action="store_true",
                   help="Modo simulación sin hardware")
    p.add_argument("--calibrate", type=int, default=0, metavar="SEG",
                   help="Segundos de calibración antes de iniciar")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    system = AdaptiveMusicSystem(
        user_id    = args.user,
        serial_port= args.port,
        api_key    = args.api_key or None,
        demo       = args.demo,
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
