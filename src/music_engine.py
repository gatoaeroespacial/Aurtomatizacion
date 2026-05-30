"""
================================================================
MUSIC_ENGINE.PY — Motor de recomendación musical adaptativo
================================================================
Responsabilidades:
  1. Indexar canciones en musica/{focus,calm,energize,stress_relief}
  2. Inferir BPM desde filename (ej: track_72bpm.mp3) o librosa
  3. Seleccionar canciones via epsilon-greedy RL
  4. Reproducir con pygame.mixer (sin dependencia de Spotify)
  5. Calcular rewards basados en cambio de estado fisiológico
  6. Persistir scores por usuario entre sesiones
  7. Actualizar scores dinámicamente

Nota sobre letras: no se cargan canciones con letras en
español o inglés (Nadon et al. 2021; Du et al. 2020).
El sistema confía en que las carpetas estén curadas por el
usuario. Agrega advertencia si el archivo no tiene BPM en
el nombre.
================================================================
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    MUSIC_FOLDERS, MUSIC_BPM_TARGET,
    RL_EPSILON_START, RL_EPSILON_MIN, RL_EPSILON_DECAY,
    RL_REWARD_GOOD, RL_PENALTY_BAD, RL_REWARD_CLIP,
    RL_MIN_PLAY_SEC, RL_EVAL_DELAY_SEC,
    PROFILE_DIR, CLASSIFIER_STATES,
)

logger = logging.getLogger(__name__)

# Intentar cargar pygame y/o librosa (opcionales)
try:
    import pygame  # type: ignore
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=1024)
    pygame.mixer.init()
    PYGAME_OK = True
    logger.info("pygame.mixer inicializado")
except Exception as e:
    PYGAME_OK = False
    logger.warning("pygame no disponible (%s). Audio desactivado.", e)

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False
    logger.info("librosa no instalado — BPM se infiere solo desde filename")


# ─────────────────────────────────────────────────────────────
# MODELOS DE DATOS
# ─────────────────────────────────────────────────────────────

@dataclass
class Track:
    """Metadatos de una canción indexada."""
    path:     Path
    name:     str
    folder:   str           # "focus" | "calm" | "energize" | "stress_relief"
    state:    str           # estado cognitivo al que pertenece
    bpm:      Optional[float] = None
    duration: Optional[float] = None   # segundos
    bpm_source: str = "unknown"        # "filename" | "librosa" | "unknown"


@dataclass
class TrackScore:
    """Score RL de una canción para un usuario dado."""
    track_id:      str      # path relativa como ID único
    score:         float = 0.0
    play_count:    int   = 0
    reward_sum:    float = 0.0
    last_played:   float = 0.0
    last_reward:   Optional[float] = None


@dataclass
class PlaybackEvent:
    """Registro de una reproducción."""
    track_id:       str
    state_at_start: str
    state_at_end:   Optional[str] = None
    started_at:     float = field(default_factory=time.time)
    duration_played: float = 0.0
    reward:         Optional[float] = None


# ─────────────────────────────────────────────────────────────
# INDEXADOR DE MÚSICA
# ─────────────────────────────────────────────────────────────

class MusicLibrary:
    """
    Escanea las carpetas de música, infiere BPM y construye el índice.
    """
    AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}

    def __init__(self):
        self._tracks: Dict[str, Track] = {}   # track_id → Track
        self._by_state: Dict[str, List[str]] = {s: [] for s in CLASSIFIER_STATES}
        self._index_time = 0.0

    def index(self) -> int:
        """
        Escanea todas las carpetas de música.
        Retorna el número de pistas encontradas.
        """
        self._tracks.clear()
        self._by_state = {s: [] for s in CLASSIFIER_STATES}
        n = 0

        for state, folder in MUSIC_FOLDERS.items():
            if not folder.exists():
                logger.warning("Carpeta no existe: %s", folder)
                continue

            for path in folder.iterdir():
                if path.suffix.lower() not in self.AUDIO_EXTS:
                    continue

                track = self._build_track(path, state)
                tid = str(path.relative_to(folder.parent))
                self._tracks[tid] = track
                self._by_state[state].append(tid)
                n += 1

        self._index_time = time.time()
        logger.info("Biblioteca indexada: %d pistas en %d estados",
                    n, len([s for s in self._by_state if self._by_state[s]]))

        if n == 0:
            logger.warning(
                "No se encontraron canciones. Agrega archivos MP3 a:\n"
                "  musica/focus/       (60–80 BPM, instrumentales)\n"
                "  musica/calm/        (70–90 BPM, instrumentales)\n"
                "  musica/energize/    (90–120 BPM)\n"
                "  musica/stress_relief/ (45–70 BPM, ambient/nature)"
            )
        return n

    def _build_track(self, path: Path, state: str) -> Track:
        folder_name = path.parent.name
        bpm, bpm_source = self._infer_bpm(path)

        duration = None
        if LIBROSA_OK:
            try:
                y, sr = librosa.load(str(path), sr=None, mono=True,
                                     duration=10)  # sólo 10s para metadata
                duration = librosa.get_duration(filename=str(path))
                if bpm is None:
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    bpm = float(tempo)
                    bpm_source = "librosa"
            except Exception as e:
                logger.debug("librosa no pudo leer %s: %s", path.name, e)

        if bpm is None:
            bpm = self._default_bpm_for_state(state)
            bpm_source = "default"
            logger.debug("BPM desconocido para '%s', usando default: %.0f",
                         path.name, bpm)

        return Track(
            path=path,
            name=path.stem,
            folder=folder_name,
            state=state,
            bpm=bpm,
            duration=duration,
            bpm_source=bpm_source,
        )

    @staticmethod
    def _infer_bpm(path: Path) -> Tuple[Optional[float], str]:
        """Extrae BPM del nombre de archivo. Ej: 'focus_72bpm.mp3' → 72."""
        pattern = r"(\d{2,3})\s*bpm"
        m = re.search(pattern, path.stem.lower())
        if m:
            return float(m.group(1)), "filename"
        return None, "unknown"

    @staticmethod
    def _default_bpm_for_state(state: str) -> float:
        lo, hi = MUSIC_BPM_TARGET.get(state, (70, 90))
        return float((lo + hi) / 2)

    def get_tracks_for_state(self, state: str) -> List[str]:
        return self._by_state.get(state, [])

    def get_track(self, tid: str) -> Optional[Track]:
        return self._tracks.get(tid)

    def is_empty(self) -> bool:
        return len(self._tracks) == 0

    def summary(self) -> Dict[str, int]:
        return {s: len(ids) for s, ids in self._by_state.items()}


# ─────────────────────────────────────────────────────────────
# RL — EPSILON-GREEDY
# ─────────────────────────────────────────────────────────────

class RLScoreManager:
    """
    Gestiona los scores RL por usuario y persiste en disco.
    Implementa epsilon-greedy con decaimiento.
    """

    def __init__(self, user_id: str):
        self.user_id  = user_id
        self.epsilon  = RL_EPSILON_START
        self._scores: Dict[str, TrackScore] = {}
        self._path    = PROFILE_DIR / f"rl_{user_id}.json"
        self._load()

    def select(self, candidates: List[str],
               current_track: Optional[str] = None) -> str:
        """
        Selecciona una pista de la lista de candidatos.
        Aplica epsilon-greedy: explora con prob ε, explota con 1-ε.
        Evita repetir la pista actual.
        """
        if not candidates:
            raise ValueError("Lista de candidatos vacía")

        pool = [c for c in candidates if c != current_track]
        if not pool:
            pool = candidates

        # Exploración
        if random.random() < self.epsilon:
            chosen = random.choice(pool)
            logger.debug("RL explore: %s (ε=%.3f)", chosen, self.epsilon)
        else:
            # Explotación: mayor score → mayor probabilidad (softmax)
            scores = [self._get_score(c).score for c in pool]
            scores_arr = np.array(scores, dtype=float)
            # Softmax para convertir scores en probabilidades
            scores_arr -= scores_arr.max()
            probs = np.exp(scores_arr)
            probs /= probs.sum()
            chosen = np.random.choice(pool, p=probs)
            logger.debug("RL exploit: %s (score=%.3f, ε=%.3f)",
                         chosen, self._get_score(chosen).score, self.epsilon)

        # Decay epsilon
        self.epsilon = max(RL_EPSILON_MIN, self.epsilon * RL_EPSILON_DECAY)

        # Registrar reproducción
        ts = self._get_score(chosen)
        ts.play_count  += 1
        ts.last_played  = time.time()
        return chosen

    def update(self, track_id: str, reward: float) -> None:
        """Actualiza el score de una pista con el reward calculado."""
        reward = float(np.clip(reward, *RL_REWARD_CLIP))
        ts = self._get_score(track_id)
        ts.score      = float(np.clip(ts.score + reward, *RL_REWARD_CLIP))
        ts.reward_sum += reward
        ts.last_reward = reward
        logger.debug("RL update %s: reward=%.3f → score=%.3f",
                     track_id, reward, ts.score)
        self._save()

    def get_score(self, track_id: str) -> float:
        return self._get_score(track_id).score

    def top_tracks(self, n: int = 5) -> List[Tuple[str, float]]:
        sorted_tracks = sorted(self._scores.items(),
                               key=lambda x: x[1].score, reverse=True)
        return [(tid, ts.score) for tid, ts in sorted_tracks[:n]]

    def _get_score(self, track_id: str) -> TrackScore:
        if track_id not in self._scores:
            self._scores[track_id] = TrackScore(track_id=track_id)
        return self._scores[track_id]

    def _save(self) -> None:
        try:
            data = {
                "user_id": self.user_id,
                "epsilon": self.epsilon,
                "scores":  {
                    tid: {
                        "score":       ts.score,
                        "play_count":  ts.play_count,
                        "reward_sum":  ts.reward_sum,
                        "last_played": ts.last_played,
                        "last_reward": ts.last_reward,
                    }
                    for tid, ts in self._scores.items()
                }
            }
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Error guardando scores RL: %s", e)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            self.epsilon = data.get("epsilon", RL_EPSILON_START)
            for tid, d in data.get("scores", {}).items():
                self._scores[tid] = TrackScore(
                    track_id=tid,
                    score=d["score"],
                    play_count=d["play_count"],
                    reward_sum=d["reward_sum"],
                    last_played=d["last_played"],
                    last_reward=d.get("last_reward"),
                )
            logger.info("Scores RL cargados (%d pistas) para usuario '%s'",
                        len(self._scores), self.user_id)
        except Exception as e:
            logger.error("Error cargando scores RL: %s", e)


# ─────────────────────────────────────────────────────────────
# REPRODUCTOR
# ─────────────────────────────────────────────────────────────

class AudioPlayer:
    """
    Reproductor de audio basado en pygame.mixer.
    Thread-safe. Fallback silencioso si pygame no disponible.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._current_path: Optional[Path] = None
        self._started_at: Optional[float] = None
        self._paused = False

    def play(self, path: Path) -> bool:
        if not PYGAME_OK:
            logger.info("[SIMULACIÓN] Reproduciendo: %s", path.name)
            with self._lock:
                self._current_path = path
                self._started_at   = time.time()
            return True
        with self._lock:
            try:
                pygame.mixer.music.load(str(path))
                pygame.mixer.music.play()
                self._current_path = path
                self._started_at   = time.time()
                self._paused       = False
                logger.info("▶ Reproduciendo: %s", path.name)
                return True
            except Exception as e:
                logger.error("Error reproduciendo %s: %s", path.name, e)
                return False

    def stop(self) -> None:
        if PYGAME_OK:
            pygame.mixer.music.stop()
        with self._lock:
            self._current_path = None
            self._started_at   = None

    def pause(self) -> None:
        if PYGAME_OK and not self._paused:
            pygame.mixer.music.pause()
            self._paused = True

    def resume(self) -> None:
        if PYGAME_OK and self._paused:
            pygame.mixer.music.unpause()
            self._paused = False

    def set_volume(self, vol: float) -> None:
        """vol: 0.0–1.0"""
        if PYGAME_OK:
            pygame.mixer.music.set_volume(float(np.clip(vol, 0.0, 1.0)))

    def is_playing(self) -> bool:
        if not PYGAME_OK:
            return self._current_path is not None
        return bool(pygame.mixer.music.get_busy())

    def seconds_played(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.time() - self._started_at

    @property
    def current_path(self) -> Optional[Path]:
        return self._current_path


# ─────────────────────────────────────────────────────────────
# MOTOR PRINCIPAL
# ─────────────────────────────────────────────────────────────

class MusicEngine:
    """
    Motor musical adaptativo. Orquesta la biblioteca, el RL y
    el reproductor. Evaluación del reward cada RL_EVAL_DELAY_SEC.
    """

    def __init__(self, user_id: str = "default"):
        self.user_id  = user_id
        self._library = MusicLibrary()
        self._rl      = RLScoreManager(user_id)
        self._player  = AudioPlayer()

        self._current_tid:   Optional[str] = None
        self._current_state: Optional[str] = None
        self._event_history: List[PlaybackEvent] = []

        self._state_at_change_time: Optional[str] = None
        self._change_time: float = 0.0
        self._eval_done_for_current = False

        # Re-indexar biblioteca al inicio
        n = self._library.index()
        logger.info("MusicEngine listo: %d pistas, usuario='%s'",
                    n, user_id)

    # ── API pública ───────────────────────────────────────────

    def update(self, new_state: str, confidence: float = 1.0) -> Optional[str]:
        """
        Llamar cada vez que el clasificador produce un nuevo estado.
        Decide si cambiar canción y retorna el ID de la pista actual.
        """
        now = time.time()

        # 1. Evaluar reward de la pista anterior
        self._maybe_evaluate_reward(now)

        # 2. ¿Hay que cambiar de canción?
        should_change = self._should_change(new_state, confidence, now)

        if should_change:
            self._change_state(new_state, now)

        return self._current_tid

    def get_current_track(self) -> Optional[Track]:
        if self._current_tid is None:
            return None
        return self._library.get_track(self._current_tid)

    def get_library_summary(self) -> Dict[str, int]:
        return self._library.summary()

    def get_top_tracks(self, n: int = 5) -> List[Tuple[str, float]]:
        return self._rl.top_tracks(n)

    def get_current_score(self) -> Optional[float]:
        if self._current_tid:
            return self._rl.get_score(self._current_tid)
        return None

    def force_change(self, state: Optional[str] = None) -> Optional[str]:
        """Fuerza cambio de canción (para pruebas o control manual)."""
        state = state or self._current_state or "media_conc"
        self._change_state(state, time.time(), forced=True)
        return self._current_tid

    def pause(self) -> None:
        self._player.pause()

    def resume(self) -> None:
        self._player.resume()

    def stop(self) -> None:
        self._player.stop()
        self._current_tid = None

    def set_volume(self, vol: float) -> None:
        self._player.set_volume(vol)

    def reindex(self) -> int:
        return self._library.index()

    # ── Lógica interna ────────────────────────────────────────

    def _should_change(self, new_state: str, conf: float, now: float) -> bool:
        """Decide si corresponde cambiar de canción."""
        # Sin canción actual → siempre iniciar
        if self._current_tid is None:
            return True

        # La canción terminó naturalmente
        if not self._player.is_playing():
            return True

        # El estado cambió significativamente
        if (new_state != self._current_state
                and conf >= 0.55
                and self._player.seconds_played() >= RL_MIN_PLAY_SEC):
            return True

        return False

    def _change_state(self, state: str, now: float,
                      forced: bool = False) -> None:
        """Selecciona y reproduce una nueva canción para el estado dado."""
        candidates = self._library.get_tracks_for_state(state)

        if not candidates:
            logger.warning("Sin canciones para estado '%s'. "
                           "Agrega MP3 a musica/%s/",
                           state, MUSIC_FOLDERS[state].name)
            return

        # Registrar evento del track anterior
        if self._current_tid and self._current_state:
            ev = PlaybackEvent(
                track_id=self._current_tid,
                state_at_start=self._current_state,
                state_at_end=state,
                started_at=self._change_time,
                duration_played=self._player.seconds_played(),
            )
            self._event_history.append(ev)

        # Selección RL
        new_tid = self._rl.select(candidates, self._current_tid)
        track   = self._library.get_track(new_tid)

        if track is None:
            logger.error("Track no encontrado en biblioteca: %s", new_tid)
            return

        # Reproducir
        ok = self._player.play(track.path)
        if ok:
            self._current_tid        = new_tid
            self._current_state      = state
            self._change_time        = now
            self._state_at_change_time = state
            self._eval_done_for_current = False
            logger.info("🎵 [%s] %s (BPM: %s, score: %.2f)",
                        state, track.name,
                        f"{track.bpm:.0f}" if track.bpm else "?",
                        self._rl.get_score(new_tid))

    def _maybe_evaluate_reward(self, now: float) -> None:
        """
        Evalúa el reward de la pista actual después de RL_EVAL_DELAY_SEC.
        El reward se basa en el cambio de estado cognitivo:
          - Si el estado mejoró → reward positivo
          - Si empeoró        → penalización
          - Si no cambió      → neutro
        """
        if self._current_tid is None:
            return
        if self._eval_done_for_current:
            return
        if (now - self._change_time) < RL_EVAL_DELAY_SEC:
            return

        # Obtener el estado actual del historial (el engine lo recibe en update)
        # La evaluación compara estado en el momento del cambio vs estado actual
        # Dado que _current_state puede haberse actualizado entre llamadas,
        # usamos el estado en que se cambió vs el que viene ahora.
        # El llamador pasa new_state → se evaluará en la siguiente iteración.
        # Aquí simplemente marcamos que ya es tiempo de evaluar y lo haremos
        # en el próximo update() que traiga new_state.
        self._eval_done_for_current = True  # se ejecutará en próximo update

    def compute_and_apply_reward(self, state_before: str, state_after: str) -> float:
        """
        Calcula el reward según la transición de estado y lo aplica al
        track actual. Llamar desde main.py con los estados correctos.
        """
        if self._current_tid is None:
            return 0.0

        reward = self._state_transition_reward(state_before, state_after)
        self._rl.update(self._current_tid, reward)

        if self._event_history:
            self._event_history[-1].reward = reward

        return reward

    @staticmethod
    def _state_transition_reward(before: str, after: str) -> float:
        """
        Tabla de rewards por transición de estado.
        La música debe llevar hacia alta_conc o al menos media_conc.
        """
        # Orden de preferencia: alta_conc > media_conc > baja_conc > estres
        state_rank = {
            "alta_conc":  3,
            "media_conc": 2,
            "baja_conc":  1,
            "estres":     0,
        }
        r_before = state_rank.get(before, 1)
        r_after  = state_rank.get(after,  1)
        delta    = r_after - r_before

        if delta > 0:
            return RL_REWARD_GOOD * delta          # mejora
        elif delta == 0:
            # Sin cambio: pequeño positivo si ya estamos bien
            return RL_REWARD_GOOD * 0.3 if r_after >= 2 else 0.0
        else:
            return RL_PENALTY_BAD * abs(delta)     # empeora

    def get_session_stats(self) -> Dict:
        """Resumen de la sesión actual para logging/LLM."""
        return {
            "total_tracks_played": len(self._event_history),
            "current_state":       self._current_state,
            "current_track":       (self.get_current_track().name
                                    if self.get_current_track() else None),
            "current_score":       self.get_current_score(),
            "epsilon":             self._rl.epsilon,
            "top_tracks":          self._rl.top_tracks(3),
            "library":             self._library.summary(),
        }
