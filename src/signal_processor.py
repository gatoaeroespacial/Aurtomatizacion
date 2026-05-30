"""
================================================================
SIGNAL_PROCESSOR.PY — Procesamiento de bioseñales en tiempo real
================================================================
Responsabilidades:
  1. Buffer deslizante de 30 s a 50 Hz
  2. Limpieza y filtrado de señales (ECG, GSR, Airflow)
  3. Detección robusta de picos R (ECG)
  4. Extracción de 10 features por ventana
  5. Validación fisiológica y rechazo de artefactos
  6. Calibración de baseline por usuario

IMPORTANTE: No se inventan fórmulas. Todo sigue la literatura
citada en el informe del proyecto (Nourbakhsh 2012, Task Force
1996 para HRV, Gonzalez & Aiello 2019).

Limitación conocida: RMSSD/SDNN a 50 Hz tienen resolución de
20 ms (no clínica). Se usan como indicadores de tendencia.
================================================================
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

from config import (
    SAMPLE_HZ, WINDOW_SAMPLES, WINDOW_SECONDS,
    ECG_BANDPASS_LOW, ECG_BANDPASS_HIGH,
    GSR_LOWPASS_FREQ, AIRFLOW_LOWPASS,
    RPEAK_MIN_HEIGHT_STD, RPEAK_REFRACTORY_MS, RPEAK_MIN_BPM, RPEAK_MAX_BPM,
    ARTIFACT_GSR_MIN, ARTIFACT_GSR_MAX, ARTIFACT_ECG_CLAMP, ARTIFACT_AIR_CLAMP,
    RESP_OPTIMAL_LOW, RESP_OPTIMAL_HIGH,
    GSR_OPTIMAL_LOW, GSR_OPTIMAL_HIGH, GSR_STRESS_HIGH,
    BPM_CALM_LOW, BPM_CALM_HIGH, BPM_STRESS_MIN,
    RMSSD_HEALTHY, RMSSD_STRESS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# ESTRUCTURAS DE DATOS
# ─────────────────────────────────────────────────────────────

@dataclass
class RawSample:
    """Una muestra cruda del Arduino."""
    ts_ms:       int
    gsr_cond:    float   # µS
    gsr_res:     float   # Ω
    gsr_volt:    float   # V
    air_volt:    float   # V
    ecg_volt:    float   # V
    ts_real:     float = field(default_factory=time.time)


@dataclass
class SignalFeatures:
    """
    10 features extraídas de una ventana de 30 s.
    Son el input del clasificador.
    """
    # ECG / HRV
    bpm:       float   # latidos por minuto (media de la ventana)
    rmssd:     float   # ms — raíz media cuadrática de diferencias NN (tendencia)
    sdnn:      float   # ms — desviación estándar de intervalos NN
    nn50:      int     # # de pares NN con diferencia > 50 ms
    pnn50:     float   # % de nn50 sobre total de pares

    # Respiración
    resp_rate: float   # respiraciones por minuto
    resp_reg:  float   # regularidad (0=irregular, 1=perfecta)

    # GSR
    scl:       float   # µS — nivel de conductancia de la piel (media ventana)
    scr_count: int     # n° de respuestas de conductancia de la piel detectadas
    gsr_std:   float   # µS — variabilidad de GSR en la ventana

    # Metadatos de calidad
    signal_quality: float  # 0.0–1.0 — cuánta señal válida hubo
    window_start_ts: float = 0.0


@dataclass
class SensorStatus:
    """Estado de conectividad de cada sensor."""
    ecg_ok:     bool = False
    gsr_ok:     bool = False
    airflow_ok: bool = False
    last_check: float = 0.0

    @property
    def all_ok(self) -> bool:
        return self.ecg_ok and self.gsr_ok and self.airflow_ok


# ─────────────────────────────────────────────────────────────
# FILTROS
# ─────────────────────────────────────────────────────────────

class SignalFilter:
    """
    Filtros IIR (Butterworth) inicializados una sola vez y
    aplicados muestra a muestra mediante lfilter con zi (estado).
    Esto evita latencia de bloque y permite filtrado en tiempo real.
    """

    def __init__(self, fs: float = SAMPLE_HZ):
        self.fs = fs
        nyq = fs / 2.0

        # ECG: pasa-banda 0.5–40 Hz
        lo = ECG_BANDPASS_LOW / nyq
        hi = min(ECG_BANDPASS_HIGH / nyq, 0.99)
        self._b_ecg, self._a_ecg = sp_signal.butter(4, [lo, hi], btype="band")
        self._zi_ecg = sp_signal.lfilter_zi(self._b_ecg, self._a_ecg) * 0.0

        # GSR: pasa-bajos 1 Hz
        fc_gsr = GSR_LOWPASS_FREQ / nyq
        self._b_gsr, self._a_gsr = sp_signal.butter(4, fc_gsr, btype="low")
        self._zi_gsr = sp_signal.lfilter_zi(self._b_gsr, self._a_gsr) * 0.0

        # Airflow: pasa-bajos 2 Hz
        fc_air = AIRFLOW_LOWPASS / nyq
        self._b_air, self._a_air = sp_signal.butter(4, fc_air, btype="low")
        self._zi_air = sp_signal.lfilter_zi(self._b_air, self._a_air) * 0.0

    def filter_ecg(self, x: float) -> float:
        y, self._zi_ecg = sp_signal.lfilter(
            self._b_ecg, self._a_ecg, [x], zi=self._zi_ecg)
        return float(y[0])

    def filter_gsr(self, x: float) -> float:
        y, self._zi_gsr = sp_signal.lfilter(
            self._b_gsr, self._a_gsr, [x], zi=self._zi_gsr)
        return float(y[0])

    def filter_air(self, x: float) -> float:
        y, self._zi_air = sp_signal.lfilter(
            self._b_air, self._a_air, [x], zi=self._zi_air)
        return float(y[0])

    def reset(self) -> None:
        self._zi_ecg = sp_signal.lfilter_zi(self._b_ecg, self._a_ecg) * 0.0
        self._zi_gsr = sp_signal.lfilter_zi(self._b_gsr, self._a_gsr) * 0.0
        self._zi_air = sp_signal.lfilter_zi(self._b_air, self._a_air) * 0.0


# ─────────────────────────────────────────────────────────────
# DETECTOR DE PICOS R (ECG)
# ─────────────────────────────────────────────────────────────

class RPeakDetector:
    """
    Detector Pan-Tompkins simplificado adaptado a 50 Hz.
    Mantiene estado entre llamadas para detección en tiempo real.

    Limitación: con 50 Hz la precisión temporal es ±20 ms.
    Para investigación clínica se requiere ≥250 Hz.
    """

    def __init__(self, fs: float = SAMPLE_HZ):
        self.fs = fs
        self._refractory_samples = int(RPEAK_REFRACTORY_MS / 1000.0 * fs)
        self._samples_since_peak = self._refractory_samples  # ya listo al inicio
        self._recent_ecg: Deque[float] = deque(maxlen=int(fs * 2))  # 2s de contexto
        self._dynamic_threshold = 0.5   # se adapta online
        self._threshold_alpha   = 0.05  # EMA para actualizar umbral

        self.peak_indices: List[int] = []   # índices globales
        self._global_idx = 0

    def process_sample(self, ecg_filtered: float) -> bool:
        """
        Procesa una muestra filtrada.
        Retorna True si se detectó un pico R en esta muestra.
        """
        self._recent_ecg.append(ecg_filtered)
        self._global_idx += 1
        self._samples_since_peak += 1

        if len(self._recent_ecg) < 10:
            return False

        arr = np.array(self._recent_ecg)
        arr_centered = arr - arr.mean()
        std = arr_centered.std()

        if std < 0.01:  # señal plana → sin latidos detectables
            return False

        threshold = arr_centered.mean() + RPEAK_MIN_HEIGHT_STD * std

        # Actualiza umbral dinámico con EMA
        self._dynamic_threshold = (
            (1 - self._threshold_alpha) * self._dynamic_threshold
            + self._threshold_alpha * threshold
        )

        # El último valor supera el umbral y es mayor que el penúltimo
        # (flanco ascendente) y hay suficiente período refractario
        if (self._samples_since_peak >= self._refractory_samples
                and ecg_filtered > self._dynamic_threshold
                and len(self._recent_ecg) >= 2
                and self._recent_ecg[-1] > self._recent_ecg[-2]):
            self._samples_since_peak = 0
            self.peak_indices.append(self._global_idx)
            return True

        return False

    def get_rr_intervals_ms(self, last_n: Optional[int] = None) -> np.ndarray:
        """
        Devuelve intervalos RR en ms a partir de los índices de picos.
        """
        peaks = self.peak_indices[-last_n:] if last_n else self.peak_indices
        if len(peaks) < 2:
            return np.array([])
        rr_samples = np.diff(peaks)
        rr_ms = rr_samples * (1000.0 / self.fs)
        # Filtrar intervalos fisiológicamente imposibles
        min_rr = 60000.0 / RPEAK_MAX_BPM
        max_rr = 60000.0 / RPEAK_MIN_BPM
        rr_ms = rr_ms[(rr_ms >= min_rr) & (rr_ms <= max_rr)]
        return rr_ms

    def reset(self) -> None:
        self.peak_indices.clear()
        self._global_idx = 0
        self._samples_since_peak = self._refractory_samples
        self._recent_ecg.clear()


# ─────────────────────────────────────────────────────────────
# PROCESADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────

class SignalProcessor:
    """
    Procesador central. Recibe muestras crudas, filtra, detecta
    picos y extrae features cada WINDOW_STEP_SEC segundos.

    Uso típico:
        proc = SignalProcessor()
        while True:
            sample = read_from_serial()
            proc.add_sample(sample)
            features = proc.get_features()   # None si ventana incompleta
    """

    def __init__(self, fs: float = SAMPLE_HZ,
                 baseline: Optional[Dict] = None):
        self.fs = fs
        self._filter   = SignalFilter(fs)
        self._detector = RPeakDetector(fs)
        self._status   = SensorStatus()

        # Buffers deslizantes (WINDOW_SAMPLES = 1500 muestras = 30 s)
        self._ecg_buf: Deque[float] = deque(maxlen=WINDOW_SAMPLES)
        self._gsr_buf: Deque[float] = deque(maxlen=WINDOW_SAMPLES)
        self._air_buf: Deque[float] = deque(maxlen=WINDOW_SAMPLES)
        self._ts_buf:  Deque[float] = deque(maxlen=WINDOW_SAMPLES)

        # Baseline de calibración del usuario
        self.baseline = baseline or {}

        # Contador de muestras procesadas
        self._total_samples    = 0
        self._invalid_samples  = 0
        self._last_feature_ts  = 0.0
        self._step_samples     = int(WINDOW_SECONDS / 1 * fs)  # re-evalúa c/5s

        self._features_ready   = False
        self._last_features: Optional[SignalFeatures] = None

        logger.info("SignalProcessor inicializado (fs=%d Hz, ventana=%d s)",
                    fs, WINDOW_SECONDS)

    # ── API pública ───────────────────────────────────────────

    def add_sample(self, s: RawSample) -> None:
        """Agrega una muestra al pipeline. Llámalo a 50 Hz."""
        self._total_samples += 1

        # 1. Validación de artefactos
        ecg_valid = ARTIFACT_ECG_CLAMP[0] <= s.ecg_volt <= ARTIFACT_ECG_CLAMP[1]
        gsr_valid = ARTIFACT_GSR_MIN <= s.gsr_cond <= ARTIFACT_GSR_MAX
        air_valid = ARTIFACT_AIR_CLAMP[0] <= s.air_volt <= ARTIFACT_AIR_CLAMP[1]

        if not ecg_valid:
            s.ecg_volt = float(np.clip(s.ecg_volt, *ARTIFACT_ECG_CLAMP))
        if not gsr_valid:
            s.gsr_cond = 0.0  # marca sin contacto
        if not air_valid:
            s.air_volt = float(np.clip(s.air_volt, *ARTIFACT_AIR_CLAMP))

        if not (ecg_valid and gsr_valid and air_valid):
            self._invalid_samples += 1

        # 2. Filtrado
        ecg_f = self._filter.filter_ecg(s.ecg_volt)
        gsr_f = self._filter.filter_gsr(s.gsr_cond)
        air_f = self._filter.filter_air(s.air_volt)

        # 3. Detección de pico R (en tiempo real, muestra a muestra)
        self._detector.process_sample(ecg_f)

        # 4. Buffers
        self._ecg_buf.append(ecg_f)
        self._gsr_buf.append(gsr_f)
        self._air_buf.append(air_f)
        self._ts_buf.append(s.ts_ms / 1000.0)

        # 5. Actualizar estado de sensores
        now = time.time()
        if now - self._status.last_check > 2.0:
            self._update_sensor_status()
            self._status.last_check = now

    def get_features(self) -> Optional[SignalFeatures]:
        """
        Retorna features si hay ventana completa (≥30 s de datos).
        Se extrae una vez por WINDOW_STEP_SEC (cada 5 s por defecto).
        """
        if len(self._ecg_buf) < WINDOW_SAMPLES:
            return None

        now = time.time()
        if now - self._last_feature_ts < 5.0 and self._last_features is not None:
            return self._last_features   # no re-calcular hasta el próximo step

        feats = self._extract_features()
        self._last_feature_ts = now
        self._last_features = feats
        return feats

    def get_sensor_status(self) -> SensorStatus:
        return self._status

    def get_signal_quality(self) -> float:
        """Ratio de muestras válidas en la sesión actual."""
        if self._total_samples == 0:
            return 0.0
        return 1.0 - self._invalid_samples / self._total_samples

    def set_baseline(self, baseline: Dict) -> None:
        """Inyecta baseline de calibración del usuario."""
        self.baseline = baseline
        logger.info("Baseline actualizado: %s", baseline)

    def reset(self) -> None:
        self._filter.reset()
        self._detector.reset()
        self._ecg_buf.clear()
        self._gsr_buf.clear()
        self._air_buf.clear()
        self._ts_buf.clear()
        self._total_samples   = 0
        self._invalid_samples = 0
        self._last_features   = None
        logger.info("SignalProcessor reseteado")

    # ── Extracción de features ────────────────────────────────

    def _extract_features(self) -> SignalFeatures:
        ecg_arr = np.array(self._ecg_buf)
        gsr_arr = np.array(self._gsr_buf)
        air_arr = np.array(self._air_buf)

        # ── ECG / HRV ─────────────────────────────────────────
        # Usamos los últimos WINDOW_SAMPLES picos detectados
        n_peaks_window = int(WINDOW_SAMPLES)
        rr_ms = self._detector.get_rr_intervals_ms(
            last_n=int(WINDOW_SECONDS * 3))  # margen: 3x latidos posibles

        if len(rr_ms) >= 2:
            bpm   = float(60000.0 / np.mean(rr_ms))
            sdnn  = float(np.std(rr_ms))
            diff_rr = np.diff(rr_ms)
            rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))
            nn50  = int(np.sum(np.abs(diff_rr) > 50))
            pnn50 = float(nn50 / len(diff_rr) * 100) if len(diff_rr) > 0 else 0.0
        else:
            # Sin suficientes picos detectados — señal ECG probablemente
            # no está en contacto; valores neutros que no engañen al clf.
            bpm = rmssd = sdnn = pnn50 = 0.0
            nn50 = 0

        # Clampeo fisiológico
        bpm = float(np.clip(bpm, 0, 220))

        # ── Respiración ───────────────────────────────────────
        resp_rate, resp_reg = self._calc_respiration(air_arr)

        # ── GSR ───────────────────────────────────────────────
        gsr_valid = gsr_arr[gsr_arr > ARTIFACT_GSR_MIN]
        if len(gsr_valid) > 10:
            scl      = float(np.mean(gsr_valid))
            gsr_std  = float(np.std(gsr_valid))
            scr_count = self._count_scr(gsr_valid)
        else:
            scl = gsr_std = 0.0
            scr_count = 0

        # ── Calidad de señal en esta ventana ──────────────────
        ecg_q = float(np.sum(
            (ecg_arr > ARTIFACT_ECG_CLAMP[0]) &
            (ecg_arr < ARTIFACT_ECG_CLAMP[1])) / len(ecg_arr))
        gsr_q = float(len(gsr_valid) / max(len(gsr_arr), 1))
        signal_quality = float(np.mean([ecg_q, gsr_q]))

        feats = SignalFeatures(
            bpm=bpm,
            rmssd=rmssd,
            sdnn=sdnn,
            nn50=nn50,
            pnn50=pnn50,
            resp_rate=resp_rate,
            resp_reg=resp_reg,
            scl=scl,
            scr_count=scr_count,
            gsr_std=gsr_std,
            signal_quality=signal_quality,
            window_start_ts=self._ts_buf[0] if self._ts_buf else 0.0,
        )

        logger.debug("Features: BPM=%.1f RMSSD=%.1f SCL=%.2f resp=%.1f q=%.2f",
                     bpm, rmssd, scl, resp_rate, signal_quality)
        return feats

    def _calc_respiration(self, air_arr: np.ndarray) -> Tuple[float, float]:
        """
        Frecuencia respiratoria via FFT + regularidad via entropía espectral.
        Rango fisiológico: 8–30 rpm.
        """
        if len(air_arr) < self.fs * 6:
            return 0.0, 0.0

        a = air_arr - air_arr.mean()
        fft_amp = np.abs(np.fft.rfft(a))
        freqs   = np.fft.rfftfreq(len(a), d=1.0 / self.fs)

        # Rango de respiración normal: 0.13–0.5 Hz (8–30 rpm)
        mask = (freqs >= 0.10) & (freqs <= 0.55)
        if not mask.any():
            return 0.0, 0.0

        fft_resp = fft_amp[mask]
        freqs_resp = freqs[mask]
        peak_freq = freqs_resp[np.argmax(fft_resp)]
        resp_rate = round(float(peak_freq * 60), 1)

        # Regularidad: energía del pico dominante vs total (0–1)
        total_energy = float(np.sum(fft_resp ** 2))
        if total_energy > 0:
            peak_energy = float(fft_resp.max() ** 2)
            regularity = float(np.clip(peak_energy / total_energy, 0, 1))
        else:
            regularity = 0.0

        return resp_rate, regularity

    def _count_scr(self, gsr_arr: np.ndarray) -> int:
        """
        Cuenta Respuestas de Conductancia de la Piel (SCR):
        incrementos de ≥0.05 µS con pendiente positiva.
        """
        if len(gsr_arr) < 5:
            return 0
        diff = np.diff(gsr_arr)
        # SCR: pendiente positiva > umbral mínimo
        threshold = max(0.05, gsr_arr.std() * 0.3)
        peaks_idx, _ = sp_signal.find_peaks(diff, height=threshold,
                                             distance=int(self.fs * 2))
        return len(peaks_idx)

    def _update_sensor_status(self) -> None:
        """Detecta sensores desconectados o con artefactos persistentes."""
        if len(self._ecg_buf) < 10:
            return

        recent_ecg = list(self._ecg_buf)[-50:]
        recent_gsr = list(self._gsr_buf)[-50:]
        recent_air = list(self._air_buf)[-50:]

        # ECG: si std es casi 0, no hay señal (plana = desconectado)
        self._status.ecg_ok = float(np.std(recent_ecg)) > 0.005

        # GSR: si todos son 0, sin contacto dérmico
        self._status.gsr_ok = float(np.mean(recent_gsr)) > ARTIFACT_GSR_MIN

        # Airflow: variación mínima (respirar genera ~0.1–0.5V pk-pk)
        air_range = float(np.max(recent_air) - np.min(recent_air))
        self._status.airflow_ok = air_range > 0.05

    # ── Calibración de baseline ───────────────────────────────

    def compute_baseline(self, duration_sec: int = 120) -> Optional[Dict]:
        """
        Calcula baseline fisiológico del usuario con los datos
        actuales en buffer (llamar después de ≥2 min de reposo).
        Retorna el dict de baseline o None si no hay suficientes datos.
        """
        n_needed = int(duration_sec * self.fs)
        if len(self._ecg_buf) < n_needed:
            logger.warning("Insuficientes datos para baseline (%d/%d muestras)",
                           len(self._ecg_buf), n_needed)
            return None

        feats = self._extract_features()
        baseline = {
            "bpm_rest":    feats.bpm,
            "rmssd_rest":  feats.rmssd,
            "scl_rest":    feats.scl,
            "resp_rest":   feats.resp_rate,
            "computed_at": time.time(),
        }
        logger.info("Baseline calculado: %s", baseline)
        return baseline

    # ── Conversión a vector para el clasificador ──────────────

    @staticmethod
    def features_to_vector(f: SignalFeatures) -> np.ndarray:
        """Devuelve los 10 features como array 1-D para sklearn."""
        return np.array([
            f.bpm, f.rmssd, f.sdnn, f.nn50, f.pnn50,
            f.resp_rate, f.resp_reg,
            f.scl, f.scr_count, f.gsr_std
        ], dtype=float)

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "bpm", "rmssd", "sdnn", "nn50", "pnn50",
            "resp_rate", "resp_reg",
            "scl", "scr_count", "gsr_std"
        ]
