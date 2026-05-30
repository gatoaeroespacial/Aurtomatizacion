"""
================================================================
LLM_INTEGRATION.PY — Integración con Claude (Anthropic API)
================================================================
El LLM actúa como capa de razonamiento contextual sobre el
pipeline fisiológico. No reemplaza al clasificador: lo enriquece.

Funciones:
  1. Interpretar transiciones de estado fisiológico
  2. Generar insights durante la sesión
  3. Justificar recomendaciones musicales
  4. Detectar patrones anómalos
  5. Generar reportes de sesión al finalizar
  6. Confirmar/corregir etiquetas del clasificador (feedback loop)
  7. Adaptar parámetros del sistema si detecta inconsistencias

Política de invocación:
  - Cada LLM_INSIGHT_EVERY ventanas procesadas
  - Al cambiar de estado cognitivo
  - Al finalizar sesión (reporte completo)
  - Bajo demanda desde main.py

No se invoca en cada muestra ni en cada ventana para evitar
latencia y costos innecesarios.
================================================================
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import urllib.request
import urllib.error

from config import (
    LLM_MODEL, LLM_MAX_TOKENS, LLM_ENDPOINT, LLM_INSIGHT_EVERY,
    CLASSIFIER_STATES, SESSION_DIR,
)
from signal_processor import SignalFeatures
from classifier import ClassificationResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# ESTRUCTURAS DE DATOS
# ─────────────────────────────────────────────────────────────

@dataclass
class LLMInsight:
    """Resultado de una consulta al LLM."""
    type:        str        # "insight" | "report" | "label_correction" | "warning"
    content:     str        # texto generado
    state:       Optional[str] = None    # estado sugerido (si aplica)
    confidence:  float = 0.0
    raw_json:    Optional[Dict] = None
    timestamp:   float = field(default_factory=time.time)
    latency_ms:  float = 0.0


@dataclass
class SessionContext:
    """
    Contexto acumulado de la sesión para dar al LLM.
    Se comprime para no exceder el context window.
    """
    user_id:          str
    start_time:       float
    state_history:    List[str]         # últimos N estados
    feature_snapshots: List[Dict]       # últimas N features (resumidas)
    music_history:    List[str]         # canciones reproducidas
    current_track:    Optional[str]
    current_state:    Optional[str]
    transitions:      List[Dict]        # cambios de estado {from, to, ts}
    total_windows:    int
    avg_confidence:   float


# ─────────────────────────────────────────────────────────────
# CLIENTE LLMM
# ─────────────────────────────────────────────────────────────

class LLMClient:
    """
    Cliente HTTP para la API de Anthropic.
    No usa el SDK para minimizar dependencias.
    """

    def __init__(self):
        self._api_key: Optional[str] = None
        self._enabled = False

    def configure(self, api_key: str) -> None:
        if not api_key or len(api_key) < 10:
            logger.warning("API key inválida — LLM desactivado")
            return
        self._api_key = api_key
        self._enabled = True
        logger.info("LLM configurado con modelo %s", LLM_MODEL)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def call(self, system: str, user: str,
             max_tokens: int = LLM_MAX_TOKENS) -> Optional[str]:
        """
        Llamada directa a /v1/messages.
        Retorna el texto de respuesta o None si falla.
        """
        if not self._enabled:
            return None

        payload = json.dumps({
            "model": LLM_MODEL,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }).encode("utf-8")

        req = urllib.request.Request(
            LLM_ENDPOINT,
            data=payload,
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         self._api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            latency = (time.time() - t0) * 1000

            text = raw.get("content", [{}])[0].get("text", "")
            logger.debug("LLM respondió en %.0f ms (%d chars)",
                         latency, len(text))
            return text

        except urllib.error.HTTPError as e:
            logger.error("LLM HTTP %d: %s", e.code, e.read().decode()[:200])
        except urllib.error.URLError as e:
            logger.error("LLM URL error: %s", e.reason)
        except Exception as e:
            logger.error("LLM error inesperado: %s", e)

        return None


# ─────────────────────────────────────────────────────────────
# PROMPTS DEL SISTEMA
# ─────────────────────────────────────────────────────────────

SYSTEM_PHYSIOLOGIST = """Eres un asistente experto en fisiología cognitiva y neurociencia afectiva, integrado en un sistema de recomendación musical adaptativa.

Tu rol es analizar señales fisiológicas (GSR, ECG/HRV, respiración) e inferir el estado cognitivo del usuario para:
1. Validar o corregir las clasificaciones automáticas del sistema
2. Detectar patrones fisiológicos relevantes
3. Justificar cambios en la recomendación musical
4. Generar insights accionables sobre la sesión

Marco científico de referencia:
- GSR/SCL: Nourbakhsh et al. 2012 (2–8 µS = zona óptima de concentración)
- HRV/RMSSD: Task Force ESC/NASPE 1996 (>30 ms = tono vagal saludable; indicador de tendencia a 50 Hz)
- Respiración: Critchley & Garfinkel 2017 (10–16 rpm = óptimo cognitivo)
- Música: Gonzalez & Aiello 2019; Hasegawa 2004; Nadon et al. 2021

IMPORTANTE:
- Las métricas HRV (RMSSD, SDNN) a 50 Hz tienen resolución de ±20 ms. Son indicadores de tendencia, no valores clínicos.
- GSR = 0 indica sensor sin contacto dérmico.
- BPM = 0 indica ECG sin contacto.
- Responde siempre en español.
- Sé conciso y técnico. No incluyas disclaimers médicos innecesarios.

Cuando detectes inconsistencias entre señales o clasificaciones improbables, señálalo explícitamente."""


SYSTEM_SESSION_REPORTER = """Eres un analista científico especializado en biofeedback cognitivo y sistemas de recomendación adaptativa.

Tu tarea es generar un reporte técnico de sesión de una sesión de trabajo con el Sistema Adaptativo de Recomendación Musical.

El reporte debe incluir:
1. Resumen ejecutivo del estado cognitivo durante la sesión
2. Análisis de tendencias fisiológicas
3. Evaluación de la efectividad musical (qué canciones correlacionaron con mejor estado)
4. Recomendaciones para próximas sesiones
5. Flags o alertas si detectas patrones preocupantes

Formato: Markdown estructurado, conciso y técnico.
Idioma: Español.
Longitud: 400–600 palabras."""


# ─────────────────────────────────────────────────────────────
# MOTOR LLM
# ─────────────────────────────────────────────────────────────

class LLMEngine:
    """
    Orquesta todas las interacciones con el LLM.
    Mantiene contexto de sesión y gestiona la frecuencia de invocación.
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self._client = LLMClient()
        self._context = SessionContext(
            user_id=user_id,
            start_time=time.time(),
            state_history=[],
            feature_snapshots=[],
            music_history=[],
            current_track=None,
            current_state=None,
            transitions=[],
            total_windows=0,
            avg_confidence=0.0,
        )
        self._windows_since_insight = 0
        self._insight_history: List[LLMInsight] = []
        self._last_state: Optional[str] = None

    def configure(self, api_key: str) -> None:
        self._client.configure(api_key)

    @property
    def enabled(self) -> bool:
        return self._client.enabled

    # ── Actualización de contexto ─────────────────────────────

    def update_context(self, result: ClassificationResult,
                       track_name: Optional[str] = None) -> None:
        """Llamar cada vez que el clasificador produce un resultado."""
        ctx = self._context
        ctx.total_windows      += 1
        ctx.current_state       = result.state
        ctx.current_track       = track_name

        # Historial de estados (últimos 50)
        ctx.state_history.append(result.state)
        if len(ctx.state_history) > 50:
            ctx.state_history = ctx.state_history[-50:]

        # Detectar transición de estado
        if self._last_state and self._last_state != result.state:
            ctx.transitions.append({
                "from": self._last_state,
                "to":   result.state,
                "ts":   time.time(),
                "conf": result.confidence,
            })
            if len(ctx.transitions) > 30:
                ctx.transitions = ctx.transitions[-30:]

        self._last_state = result.state

        # Snapshot de features (resumido para no saturar el contexto LLM)
        if result.features:
            f = result.features
            snap = {
                "bpm":   round(f.bpm, 1),
                "rmssd": round(f.rmssd, 1),
                "scl":   round(f.scl, 2),
                "resp":  round(f.resp_rate, 1),
                "q":     round(f.signal_quality, 2),
            }
            ctx.feature_snapshots.append(snap)
            if len(ctx.feature_snapshots) > 20:
                ctx.feature_snapshots = ctx.feature_snapshots[-20:]

        # Historial de música
        if track_name and (not ctx.music_history
                           or ctx.music_history[-1] != track_name):
            ctx.music_history.append(track_name)
            if len(ctx.music_history) > 20:
                ctx.music_history = ctx.music_history[-20:]

        # Confidence promedio (EMA)
        alpha = 0.1
        ctx.avg_confidence = (
            alpha * result.confidence + (1 - alpha) * ctx.avg_confidence
            if ctx.avg_confidence > 0
            else result.confidence
        )

        self._windows_since_insight += 1

    # ── Generación de insights ────────────────────────────────

    def maybe_generate_insight(self) -> Optional[LLMInsight]:
        """
        Genera un insight si han pasado LLM_INSIGHT_EVERY ventanas.
        """
        if (not self._client.enabled
                or self._windows_since_insight < LLM_INSIGHT_EVERY):
            return None

        self._windows_since_insight = 0
        return self._generate_insight()

    def on_state_change(self, from_state: str,
                        to_state: str) -> Optional[LLMInsight]:
        """
        Genera insight específico cuando cambia el estado cognitivo.
        """
        if not self._client.enabled:
            return None
        return self._generate_state_change_insight(from_state, to_state)

    def generate_session_report(self) -> Optional[LLMInsight]:
        """Reporte completo al finalizar la sesión."""
        if not self._client.enabled:
            return self._generate_local_report()
        return self._generate_llm_report()

    def request_label_validation(self,
                                 result: ClassificationResult) -> Optional[LLMInsight]:
        """
        Pide al LLM que valide la etiqueta del clasificador.
        Útil cuando la confianza es baja (<0.4).
        """
        if not self._client.enabled or result.confidence >= 0.4:
            return None
        return self._validate_label(result)

    # ── Implementaciones internas ─────────────────────────────

    def _generate_insight(self) -> Optional[LLMInsight]:
        ctx = self._context
        prompt = self._build_insight_prompt(ctx)

        t0  = time.time()
        txt = self._client.call(SYSTEM_PHYSIOLOGIST, prompt, max_tokens=300)
        if not txt:
            return None

        insight = LLMInsight(
            type="insight",
            content=txt.strip(),
            state=ctx.current_state,
            latency_ms=(time.time() - t0) * 1000,
        )
        self._insight_history.append(insight)
        logger.info("LLM insight generado: %s...", txt[:80])
        return insight

    def _generate_state_change_insight(self, from_s: str,
                                        to_s: str) -> Optional[LLMInsight]:
        ctx = self._context
        feat_str = self._features_to_str(
            ctx.feature_snapshots[-1] if ctx.feature_snapshots else {})

        prompt = f"""El sistema detectó una transición de estado cognitivo:
{from_s} → {to_s}

Señales fisiológicas actuales:
{feat_str}

Pista musical actual: {ctx.current_track or 'ninguna'}

¿Esta transición es fisiológicamente coherente? ¿Qué puede estar causando el cambio?
Responde en máximo 3 oraciones técnicas."""

        t0  = time.time()
        txt = self._client.call(SYSTEM_PHYSIOLOGIST, prompt, max_tokens=200)
        if not txt:
            return None

        return LLMInsight(
            type="insight",
            content=txt.strip(),
            state=to_s,
            latency_ms=(time.time() - t0) * 1000,
        )

    def _validate_label(self, result: ClassificationResult) -> Optional[LLMInsight]:
        if not result.features:
            return None
        f = result.features
        feat_str = self._features_to_str({
            "bpm": f.bpm, "rmssd": f.rmssd, "scl": f.scl,
            "resp": f.resp_rate, "nn50": f.nn50,
        })

        scores_str = ", ".join(
            f"{k}: {v:.2f}" for k, v in result.scores.items())

        prompt = f"""El clasificador asignó el estado '{result.state}' con confianza {result.confidence:.2f}.
Puntuaciones por estado: {scores_str}

Señales fisiológicas:
{feat_str}

¿Confirmas este estado? Si no, ¿cuál sería más apropiado?
Responde SOLO con JSON: {{"state": "...", "confidence": 0.X, "reason": "..."}}
Estado debe ser uno de: alta_conc, media_conc, baja_conc, estres"""

        t0  = time.time()
        txt = self._client.call(SYSTEM_PHYSIOLOGIST, prompt, max_tokens=150)
        if not txt:
            return None

        try:
            # Limpiar JSON
            clean = txt.strip().strip("```json").strip("```").strip()
            data  = json.loads(clean)
            return LLMInsight(
                type="label_correction",
                content=data.get("reason", txt),
                state=data.get("state", result.state),
                confidence=data.get("confidence", result.confidence),
                raw_json=data,
                latency_ms=(time.time() - t0) * 1000,
            )
        except json.JSONDecodeError:
            return LLMInsight(
                type="label_correction",
                content=txt.strip(),
                state=result.state,
                latency_ms=(time.time() - t0) * 1000,
            )

    def _generate_llm_report(self) -> Optional[LLMInsight]:
        ctx = self._context
        duration_min = (time.time() - ctx.start_time) / 60

        state_dist = {}
        for s in ctx.state_history:
            state_dist[s] = state_dist.get(s, 0) + 1

        transitions_str = "\n".join(
            f"  {t['from']} → {t['to']} (confianza: {t['conf']:.2f})"
            for t in ctx.transitions[-10:])

        features_avg = self._avg_features(ctx.feature_snapshots)
        feat_str     = self._features_to_str(features_avg)

        music_str = ", ".join(ctx.music_history[-10:]) or "ninguna"

        prompt = f"""Sesión de trabajo cognitivo con biofeedback musical:

DURACIÓN: {duration_min:.1f} minutos
USUARIO: {ctx.user_id}
VENTANAS PROCESADAS: {ctx.total_windows}
CONFIANZA PROMEDIO: {ctx.avg_confidence:.2f}

DISTRIBUCIÓN DE ESTADOS:
{json.dumps(state_dist, ensure_ascii=False, indent=2)}

ÚLTIMAS TRANSICIONES:
{transitions_str or 'ninguna'}

PROMEDIOS FISIOLÓGICOS DE LA SESIÓN:
{feat_str}

MÚSICA REPRODUCIDA (últimas 10):
{music_str}

Genera el reporte completo de la sesión."""

        t0  = time.time()
        txt = self._client.call(SYSTEM_SESSION_REPORTER, prompt,
                                max_tokens=800)
        if not txt:
            return self._generate_local_report()

        insight = LLMInsight(
            type="report",
            content=txt.strip(),
            latency_ms=(time.time() - t0) * 1000,
        )
        self._save_report(insight)
        return insight

    def _generate_local_report(self) -> LLMInsight:
        """Reporte local (sin LLM) cuando la API no está disponible."""
        ctx = self._context
        duration_min = (time.time() - ctx.start_time) / 60

        state_dist: Dict[str, int] = {}
        for s in ctx.state_history:
            state_dist[s] = state_dist.get(s, 0) + 1

        most_common = max(state_dist, key=state_dist.get) if state_dist else "N/A"

        report = f"""# Reporte de Sesión — {time.strftime('%Y-%m-%d %H:%M')}

## Resumen
- Duración: {duration_min:.1f} min
- Ventanas analizadas: {ctx.total_windows}
- Estado predominante: **{most_common}**
- Confianza promedio: {ctx.avg_confidence:.2f}

## Distribución de estados
{chr(10).join(f'- {s}: {c} ventanas ({100*c/max(len(ctx.state_history),1):.0f}%)' for s, c in state_dist.items())}

## Transiciones detectadas: {len(ctx.transitions)}

## Música
Pistas reproducidas: {len(ctx.music_history)}

*(Reporte generado localmente — configura API key para análisis LLM completo)*"""

        insight = LLMInsight(
            type="report",
            content=report,
        )
        self._save_report(insight)
        return insight

    # ── Utilidades ────────────────────────────────────────────

    def _build_insight_prompt(self, ctx: SessionContext) -> str:
        feat_str = self._features_to_str(
            ctx.feature_snapshots[-1] if ctx.feature_snapshots else {})
        recent_states = " → ".join(ctx.state_history[-8:])

        return f"""Análisis de sesión en curso (ventana {ctx.total_windows}):

Estado actual: {ctx.current_state}
Estados recientes: {recent_states}
Pista musical: {ctx.current_track or 'ninguna'}

Señales fisiológicas (última ventana de 30 s):
{feat_str}

Transiciones en esta sesión: {len(ctx.transitions)}

Genera un insight técnico conciso (máx. 2 oraciones) sobre el estado cognitivo
del usuario y si la música actual es apropiada."""

    @staticmethod
    def _features_to_str(f: Dict) -> str:
        if not f:
            return "  No disponible"
        lines = []
        labels = {
            "bpm":   "BPM cardíaco",
            "rmssd": "RMSSD (ms, indicador HRV)",
            "scl":   "SCL/GSR (µS)",
            "resp":  "Respiración (rpm)",
            "nn50":  "NN50",
            "q":     "Calidad señal",
        }
        for k, v in f.items():
            label = labels.get(k, k)
            lines.append(f"  {label}: {v}")
        return "\n".join(lines)

    @staticmethod
    def _avg_features(snapshots: List[Dict]) -> Dict:
        if not snapshots:
            return {}
        keys = snapshots[0].keys()
        return {
            k: round(sum(s.get(k, 0) for s in snapshots) / len(snapshots), 2)
            for k in keys
        }

    def _save_report(self, insight: LLMInsight) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = SESSION_DIR / f"reporte_{self.user_id}_{ts}.md"
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(insight.content)
            logger.info("Reporte guardado: %s", path)
        except Exception as e:
            logger.error("Error guardando reporte: %s", e)

    def get_recent_insights(self, n: int = 5) -> List[LLMInsight]:
        return self._insight_history[-n:]
