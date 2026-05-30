"""
================================================================
CLASSIFIER.PY — Clasificación del estado cognitivo
================================================================
Arquitectura de dos fases:
  Fase 1 (< 50 muestras): heurística basada en literatura
  Fase 2 (≥ 50 muestras): Random Forest adaptativo + heurística
                           como fallback si RF tiene baja confianza

Estados posibles:
  alta_conc  — concentración alta
  media_conc — concentración media
  baja_conc  — concentración baja / fatiga
  estres     — estrés fisiológico

El RF se re-entrena automáticamente cada RF_RETRAIN_EVERY
ventanas nuevas usando el historial de etiquetas confirmadas.

Perfiles personales: el baseline fisiológico del usuario
ajusta los umbrales de la heurística proporcionalmente.
================================================================
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    CLASSIFIER_HEURISTIC_UNTIL, CLASSIFIER_STATES,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RETRAIN_EVERY,
    MODEL_DIR,
    BPM_CALM_LOW, BPM_CALM_HIGH, BPM_STRESS_MIN,
    RMSSD_HEALTHY, RMSSD_STRESS,
    GSR_OPTIMAL_LOW, GSR_OPTIMAL_HIGH, GSR_STRESS_HIGH, GSR_RELAX_LOW,
    RESP_OPTIMAL_LOW, RESP_OPTIMAL_HIGH, RESP_STRESS_HIGH,
)
from signal_processor import SignalFeatures, SignalProcessor

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# ESTRUCTURAS DE DATOS
# ─────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    state:       str       # uno de CLASSIFIER_STATES
    confidence:  float     # 0.0–1.0
    method:      str       # "heuristic" | "random_forest" | "hybrid"
    scores:      Dict[str, float] = field(default_factory=dict)
    # scores para cada estado (probabilidades o puntuaciones)
    timestamp:   float = field(default_factory=time.time)
    features:    Optional[SignalFeatures] = None


@dataclass
class TrainingPoint:
    """Un punto de entrenamiento etiquetado."""
    vector:    np.ndarray   # 10 features
    label:     str          # estado cognitivo
    timestamp: float
    source:    str          # "heuristic" | "user_confirmed" | "llm_inferred"


# ─────────────────────────────────────────────────────────────
# HEURÍSTICA
# ─────────────────────────────────────────────────────────────

class HeuristicClassifier:
    """
    Clasificación basada en umbrales de la literatura.
    Funciona sin datos de entrenamiento.

    Sistema de puntuación: cada señal contribuye al score de
    cada estado. El estado con mayor score gana.

    Referencia de umbrales:
      GSR:  Nourbakhsh et al. 2012
      HRV:  Task Force ESC/NASPE 1996
      Resp: Critchley & Garfinkel 2017
      Música: Gonzalez & Aiello 2019; Hasegawa 2004
    """

    def __init__(self, baseline: Optional[Dict] = None):
        self.baseline = baseline or {}

    def _adjust(self, value: float, key: str, factor: float = 0.2) -> float:
        """
        Ajusta un umbral según el baseline personal.
        Si el usuario tiene un valor de reposo diferente al estándar,
        los umbrales se desplazan proporcionalmente.
        """
        if key not in self.baseline:
            return value
        standard = {"bpm_rest": 70, "rmssd_rest": 35,
                    "scl_rest": 3.0, "resp_rest": 14}.get(key, value)
        delta = self.baseline[key] - standard
        return value + delta * factor

    def classify(self, f: SignalFeatures) -> ClassificationResult:
        scores: Dict[str, float] = {s: 0.0 for s in CLASSIFIER_STATES}

        # ── Penalización por calidad de señal ─────────────────
        q = max(f.signal_quality, 0.1)

        # ── GSR ───────────────────────────────────────────────
        scl = f.scl
        scl_opt_lo = self._adjust(GSR_OPTIMAL_LOW,  "scl_rest")
        scl_opt_hi = self._adjust(GSR_OPTIMAL_HIGH, "scl_rest")

        if scl <= 0:
            pass  # sin contacto: no suma ni resta
        elif scl_opt_lo <= scl <= scl_opt_hi:
            scores["alta_conc"]  += 3.0 * q
            scores["media_conc"] += 1.5 * q
        elif GSR_RELAX_LOW < scl < scl_opt_lo:
            scores["media_conc"] += 2.0 * q
            scores["baja_conc"]  += 1.5 * q
        elif scl > GSR_STRESS_HIGH:
            scores["estres"]     += 4.0 * q
            scores["baja_conc"]  += 1.0 * q
        elif scl_opt_hi < scl <= GSR_STRESS_HIGH:
            scores["estres"]     += 2.0 * q
            scores["media_conc"] += 1.0 * q
        else:
            scores["baja_conc"]  += 2.0 * q

        # GSR variabilidad alta = estrés o arousal
        if f.gsr_std > 1.0:
            scores["estres"]    += 1.5 * q
        if f.scr_count >= 3:
            scores["estres"]    += 1.5 * q

        # ── ECG / BPM ─────────────────────────────────────────
        bpm = f.bpm
        bpm_calm_lo = self._adjust(BPM_CALM_LOW,  "bpm_rest")
        bpm_calm_hi = self._adjust(BPM_CALM_HIGH, "bpm_rest")

        if bpm <= 0:
            pass  # ECG sin contacto
        elif bpm_calm_lo <= bpm <= bpm_calm_hi:
            scores["alta_conc"]  += 3.0 * q
            scores["media_conc"] += 1.0 * q
        elif bpm_calm_hi < bpm <= BPM_STRESS_MIN:
            scores["media_conc"] += 2.5 * q
        elif bpm > BPM_STRESS_MIN:
            scores["estres"]     += 3.5 * q

        # ── HRV ───────────────────────────────────────────────
        # Recordatorio: a 50 Hz, RMSSD es indicador de tendencia, no clínico
        rmssd_h = self._adjust(RMSSD_HEALTHY, "rmssd_rest")
        if f.rmssd > 0:
            if f.rmssd >= rmssd_h:
                scores["alta_conc"]  += 2.0 * q
                scores["media_conc"] += 1.0 * q
            elif f.rmssd >= RMSSD_STRESS:
                scores["media_conc"] += 1.5 * q
            else:
                scores["estres"]     += 2.0 * q
                scores["baja_conc"]  += 1.0 * q

        # NN50 alto → buena variabilidad → calma/concentración
        if f.nn50 >= 5:
            scores["alta_conc"]  += 1.0 * q
        elif f.nn50 == 0 and f.bpm > 0:
            scores["estres"]     += 1.0 * q

        # ── Respiración ───────────────────────────────────────
        resp = f.resp_rate
        if resp > 0:
            if RESP_OPTIMAL_LOW <= resp <= RESP_OPTIMAL_HIGH:
                scores["alta_conc"]  += 2.0 * q
                scores["media_conc"] += 1.0 * q
            elif RESP_OPTIMAL_HIGH < resp <= RESP_STRESS_HIGH:
                scores["estres"]     += 2.0 * q
                scores["media_conc"] += 0.5 * q
            elif resp > RESP_STRESS_HIGH:
                scores["estres"]     += 3.5 * q
            else:
                scores["baja_conc"]  += 1.5 * q

        if f.resp_reg < 0.3 and resp > 0:
            scores["estres"]     += 1.0 * q  # respiración irregular

        # ── Ganador ───────────────────────────────────────────
        total = sum(scores.values())
        if total == 0:
            state = "media_conc"
            confidence = 0.1
        else:
            state = max(scores, key=scores.__getitem__)
            probs = {k: v / total for k, v in scores.items()}
            confidence = float(probs[state])

        return ClassificationResult(
            state=state,
            confidence=confidence,
            method="heuristic",
            scores=scores,
            features=f,
        )


# ─────────────────────────────────────────────────────────────
# RANDOM FOREST
# ─────────────────────────────────────────────────────────────

class RFClassifier:
    """
    Random Forest adaptativo. Se activa tras CLASSIFIER_HEURISTIC_UNTIL
    puntos de entrenamiento y se re-entrena periódicamente.
    """

    def __init__(self):
        if not SKLEARN_OK:
            raise ImportError("scikit-learn no disponible. "
                              "pip install scikit-learn")
        self.clf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._trained = False
        self._cv_score = 0.0

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Entrena el modelo. Retorna el accuracy de validación cruzada.
        """
        if len(np.unique(y)) < 2:
            logger.warning("RF: necesita ≥2 clases para entrenar (%d disponible)",
                           len(np.unique(y)))
            return 0.0

        X_scaled = self.scaler.fit_transform(X)
        scores = cross_val_score(self.clf, X_scaled, y,
                                 cv=min(5, len(y) // 5),
                                 scoring="balanced_accuracy")
        self.clf.fit(X_scaled, y)
        self._trained = True
        self._cv_score = float(np.mean(scores))
        logger.info("RF entrenado: CV balanced_accuracy=%.3f (n=%d)",
                    self._cv_score, len(y))
        return self._cv_score

    def predict(self, x: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Predice estado. Retorna (estado, confianza, probabilidades por clase).
        """
        if not self._trained:
            raise RuntimeError("RF no entrenado todavía")
        x_s = self.scaler.transform(x.reshape(1, -1))
        proba = self.clf.predict_proba(x_s)[0]
        classes = self.clf.classes_
        state = classes[np.argmax(proba)]
        confidence = float(np.max(proba))
        scores = {c: float(p) for c, p in zip(classes, proba)}
        return str(state), confidence, scores

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def cv_score(self) -> float:
        return self._cv_score

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"clf": self.clf, "scaler": self.scaler,
                         "cv_score": self._cv_score}, f)
        logger.info("RF guardado: %s", path)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
            self.clf = d["clf"]
            self.scaler = d["scaler"]
            self._cv_score = d.get("cv_score", 0.0)
            self._trained = True
            logger.info("RF cargado: %s (CV=%.3f)", path, self._cv_score)
            return True
        except Exception as e:
            logger.error("Error cargando RF: %s", e)
            return False


# ─────────────────────────────────────────────────────────────
# CLASIFICADOR PRINCIPAL
# ─────────────────────────────────────────────────────────────

class CognitiveClassifier:
    """
    Orquestador. Combina heurística y RF.
    """

    def __init__(self, user_id: str = "default",
                 baseline: Optional[Dict] = None):
        self.user_id   = user_id
        self.baseline  = baseline or {}
        self._heuristic = HeuristicClassifier(baseline)
        self._rf        = RFClassifier() if SKLEARN_OK else None

        self._training_data: List[TrainingPoint] = []
        self._windows_since_retrain = 0
        self._total_windows = 0

        # Historial de clasificaciones (para el LLM y el motor musical)
        self.history: List[ClassificationResult] = []

        # Cargar modelo previo si existe
        self._model_path = MODEL_DIR / f"rf_{user_id}.pkl"
        if self._rf and self._rf.load(self._model_path):
            logger.info("Modelo RF previo cargado para usuario '%s'", user_id)

    # ── API pública ───────────────────────────────────────────

    def classify(self, f: SignalFeatures) -> ClassificationResult:
        """
        Clasifica el estado cognitivo dado un objeto SignalFeatures.
        """
        self._total_windows += 1
        vec = SignalProcessor.features_to_vector(f)

        # Heurística siempre disponible
        heuristic_result = self._heuristic.classify(f)

        # RF si ya está entrenado con suficientes datos
        result = heuristic_result
        if (self._rf is not None
                and self._rf.is_trained
                and len(self._training_data) >= CLASSIFIER_HEURISTIC_UNTIL):

            try:
                rf_state, rf_conf, rf_scores = self._rf.predict(vec)

                if rf_conf >= 0.55:
                    # RF tiene suficiente confianza → lo usamos
                    result = ClassificationResult(
                        state=rf_state,
                        confidence=rf_conf,
                        method="random_forest",
                        scores=rf_scores,
                        features=f,
                    )
                elif rf_conf >= 0.40:
                    # Confianza media → voto ponderado RF+heurística
                    merged = self._merge_scores(
                        rf_scores, heuristic_result.scores,
                        w_rf=rf_conf, w_h=1 - rf_conf)
                    state = max(merged, key=merged.__getitem__)
                    result = ClassificationResult(
                        state=state,
                        confidence=float(merged[state]),
                        method="hybrid",
                        scores=merged,
                        features=f,
                    )
                # else: confianza RF muy baja → usar heurística (ya asignado)

            except Exception as e:
                logger.warning("RF falló, usando heurística: %s", e)

        # Guardar como punto de entrenamiento (etiqueta = heurística)
        self._add_training_point(
            vec, result.state, source="heuristic"
        )

        # Re-entrenar si toca
        self._maybe_retrain()

        self.history.append(result)
        if len(self.history) > 200:
            self.history = self.history[-200:]

        return result

    def confirm_label(self, state: str, timestamp: Optional[float] = None) -> None:
        """
        Permite al usuario (o al LLM) confirmar/corregir la etiqueta
        de la última clasificación. Mejora el entrenamiento futuro.
        """
        if state not in CLASSIFIER_STATES:
            logger.warning("Estado inválido: %s", state)
            return
        if not self._training_data:
            return
        # Actualizar el último punto con la etiqueta confirmada
        self._training_data[-1].label  = state
        self._training_data[-1].source = "user_confirmed"
        logger.info("Etiqueta confirmada: %s", state)

    def get_state_distribution(self, last_n: int = 20) -> Dict[str, float]:
        """Distribución de estados en las últimas N clasificaciones."""
        recent = self.history[-last_n:]
        if not recent:
            return {s: 0.0 for s in CLASSIFIER_STATES}
        counts: Dict[str, int] = {s: 0 for s in CLASSIFIER_STATES}
        for r in recent:
            counts[r.state] = counts.get(r.state, 0) + 1
        total = len(recent)
        return {s: c / total for s, c in counts.items()}

    def get_training_size(self) -> int:
        return len(self._training_data)

    def save_model(self) -> None:
        if self._rf and self._rf.is_trained:
            self._rf.save(self._model_path)

    # ── Entrenamiento ─────────────────────────────────────────

    def _add_training_point(self, vec: np.ndarray, label: str,
                            source: str) -> None:
        self._training_data.append(TrainingPoint(
            vector=vec.copy(),
            label=label,
            timestamp=time.time(),
            source=source,
        ))
        # Mantener historial razonable (últimas 500 ventanas)
        if len(self._training_data) > 500:
            self._training_data = self._training_data[-500:]

        self._windows_since_retrain += 1

    def _maybe_retrain(self) -> None:
        if self._rf is None:
            return
        n = len(self._training_data)
        if (n >= CLASSIFIER_HEURISTIC_UNTIL
                and self._windows_since_retrain >= RF_RETRAIN_EVERY):
            self._retrain()
            self._windows_since_retrain = 0

    def _retrain(self) -> None:
        X = np.array([p.vector for p in self._training_data])
        y = np.array([p.label  for p in self._training_data])
        try:
            cv = self._rf.train(X, y)
            self._rf.save(self._model_path)
            logger.info("RF re-entrenado (n=%d, CV=%.3f)", len(y), cv)
        except Exception as e:
            logger.error("Error re-entrenando RF: %s", e)

    @staticmethod
    def _merge_scores(s1: Dict[str, float], s2: Dict[str, float],
                      w_rf: float, w_h: float) -> Dict[str, float]:
        """Suma ponderada de scores de dos clasificadores."""
        all_states = set(s1) | set(s2)
        merged = {}
        total_w = w_rf + w_h
        for s in all_states:
            merged[s] = (s1.get(s, 0) * w_rf + s2.get(s, 0) * w_h) / total_w
        # Normalizar
        t = sum(merged.values())
        if t > 0:
            merged = {k: v / t for k, v in merged.items()}
        return merged
