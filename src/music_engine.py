"""
================================================================
MUSIC_ENGINE.PY — Motor de recomendación musical adaptativo
================================================================
Cambios v2:
  - AudioPlayer: fade out/in en hilo separado, sin locks durante carga
  - MusicEngine: hysteresis + stability counter antes de cambiar canción
  - Política de cambio inteligente: evalúa tendencia, tiempo reproducido,
    score actual vs candidato antes de decidir
  - Logs explícitos de CADA decisión musical (motivo registrado)
  - PlaybackEvent incluye change_reason para el monitor
================================================================
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from config import (
    MUSIC_FOLDERS, MUSIC_BPM_TARGET,
    RL_EPSILON_START, RL_EPSILON_MIN, RL_EPSILON_DECAY,
    RL_REWARD_GOOD, RL_PENALTY_BAD, RL_REWARD_CLIP,
    RL_MIN_PLAY_SEC, RL_EVAL_DELAY_SEC,
    PROFILE_DIR, CLASSIFIER_STATES,
    FADE_OUT_MS, FADE_IN_MS, FADE_STEPS,
    STATE_STABILITY_COUNT, STATE_CHANGE_CONF_MIN, STATE_STABILITY_WINDOW,
)

logger = logging.getLogger(__name__)

try:
    import pygame
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=2048)
    pygame.mixer.init()
    PYGAME_OK = True
    logger.info("pygame.mixer inicializado OK")
except Exception as e:
    PYGAME_OK = False
    logger.warning("pygame no disponible (%s) — audio desactivado", e)

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False


# ─────────────────────────────────────────────────────────────
# MODELOS DE DATOS
# ─────────────────────────────────────────────────────────────

@dataclass
class Track:
    path:       Path
    name:       str
    folder:     str
    state:      str
    bpm:        Optional[float] = None
    duration:   Optional[float] = None
    bpm_source: str = "unknown"


@dataclass
class TrackScore:
    track_id:    str
    score:       float = 0.0
    play_count:  int   = 0
    reward_sum:  float = 0.0
    last_played: float = 0.0
    last_reward: Optional[float] = None


@dataclass
class PlaybackEvent:
    track_id:        str
    track_name:      str
    state_at_start:  str
    state_at_end:    Optional[str] = None
    started_at:      float = field(default_factory=time.time)
    duration_played: float = 0.0
    reward:          Optional[float] = None
    change_reason:   str = ""   # motivo explícito del cambio


# ─────────────────────────────────────────────────────────────
# BIBLIOTECA
# ─────────────────────────────────────────────────────────────

class MusicLibrary:
    AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}

    def __init__(self):
        self._tracks:   Dict[str, Track]      = {}
        self._by_state: Dict[str, List[str]]  = {s: [] for s in CLASSIFIER_STATES}

    def index(self) -> int:
        self._tracks.clear()
        self._by_state = {s: [] for s in CLASSIFIER_STATES}
        n = 0
        for state, folder in MUSIC_FOLDERS.items():
            if not folder.exists():
                logger.warning("Carpeta música no existe: %s", folder)
                continue
            for path in folder.iterdir():
                if path.suffix.lower() not in self.AUDIO_EXTS:
                    continue
                tid = str(path.relative_to(folder.parent))
                self._tracks[tid]  = self._build_track(path, state)
                self._by_state[state].append(tid)
                n += 1

        logger.info("Biblioteca: %d pistas | %s", n,
                    {s: len(v) for s, v in self._by_state.items()})
        if n == 0:
            logger.warning(
                "Sin canciones. Agrega MP3 a:\n"
                "  musica/focus/  musica/calm/  musica/energize/  musica/stress_relief/")
        return n

    def _build_track(self, path: Path, state: str) -> Track:
        import re
        m = re.search(r"(\d{2,3})\s*bpm", path.stem.lower())
        bpm, bpm_source = (float(m.group(1)), "filename") if m else (None, "unknown")
        duration = None

        if LIBROSA_OK:
            try:
                y, sr = librosa.load(str(path), sr=None, mono=True, duration=10)
                duration = librosa.get_duration(filename=str(path))
                if bpm is None:
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    bpm, bpm_source = float(tempo), "librosa"
            except Exception:
                pass

        if bpm is None:
            lo, hi = MUSIC_BPM_TARGET.get(state, (70, 90))
            bpm, bpm_source = float((lo + hi) / 2), "default"

        return Track(path=path, name=path.stem, folder=path.parent.name,
                     state=state, bpm=bpm, duration=duration, bpm_source=bpm_source)

    def get_tracks_for_state(self, state: str) -> List[str]:
        return self._by_state.get(state, [])

    def get_tracks_in_folder(self, folder: str) -> List[str]:
        return [tid for tid, tr in self._tracks.items() if tr.folder == folder]

    def get_track(self, tid: str) -> Optional[Track]:
        return self._tracks.get(tid)

    def summary(self) -> Dict[str, int]:
        return {s: len(ids) for s, ids in self._by_state.items()}

    def is_empty(self) -> bool:
        return len(self._tracks) == 0


# ─────────────────────────────────────────────────────────────
# RL SCORE MANAGER
# ─────────────────────────────────────────────────────────────

class RLScoreManager:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.epsilon = RL_EPSILON_START
        self._scores: Dict[str, TrackScore] = {}
        self._path   = PROFILE_DIR / f"rl_{user_id}.json"
        self._load()

    def select(self, candidates: List[str],
               current_track: Optional[str] = None) -> str:
        if not candidates:
            raise ValueError("Lista de candidatos vacía")
        pool = [c for c in candidates if c != current_track] or candidates

        if random.random() < self.epsilon:
            chosen = random.choice(pool)
            logger.debug("[RL] Explore → %s (ε=%.3f)", chosen, self.epsilon)
        else:
            scores = np.array([self._get(c).score for c in pool], dtype=float)
            scores -= scores.max()
            probs   = np.exp(scores)
            probs  /= probs.sum()
            chosen  = np.random.choice(pool, p=probs)
            logger.debug("[RL] Exploit → %s (score=%.3f, ε=%.3f)",
                         chosen, self._get(chosen).score, self.epsilon)

        self.epsilon = max(RL_EPSILON_MIN, self.epsilon * RL_EPSILON_DECAY)
        ts = self._get(chosen)
        ts.play_count  += 1
        ts.last_played  = time.time()
        return chosen

    def update(self, track_id: str, reward: float) -> None:
        reward = float(np.clip(reward, *RL_REWARD_CLIP))
        ts = self._get(track_id)
        ts.score       = float(np.clip(ts.score + reward, *RL_REWARD_CLIP))
        ts.reward_sum += reward
        ts.last_reward = reward
        logger.info("[RL] Reward %.3f → score %.3f | %s",
                    reward, ts.score, Path(track_id).stem)
        self._save()

    def get_score(self, tid: str) -> float:
        return self._get(tid).score

    def get_ts(self, tid: str) -> TrackScore:
        return self._get(tid)

    def top_tracks(self, n: int = 5) -> List[Tuple[str, float]]:
        s = sorted(self._scores.items(), key=lambda x: x[1].score, reverse=True)
        return [(tid, ts.score) for tid, ts in s[:n]]

    def _get(self, tid: str) -> TrackScore:
        if tid not in self._scores:
            self._scores[tid] = TrackScore(track_id=tid)
        return self._scores[tid]

    def _save(self) -> None:
        try:
            data = {"user_id": self.user_id, "epsilon": self.epsilon,
                    "scores": {tid: {"score": ts.score, "play_count": ts.play_count,
                                     "reward_sum": ts.reward_sum,
                                     "last_played": ts.last_played,
                                     "last_reward": ts.last_reward}
                               for tid, ts in self._scores.items()}}
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("[RL] Error guardando scores: %s", e)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            self.epsilon = data.get("epsilon", RL_EPSILON_START)
            for tid, d in data.get("scores", {}).items():
                self._scores[tid] = TrackScore(
                    track_id=tid, score=d["score"],
                    play_count=d["play_count"], reward_sum=d["reward_sum"],
                    last_played=d["last_played"], last_reward=d.get("last_reward"))
            logger.info("[RL] Cargados %d scores para '%s'",
                        len(self._scores), self.user_id)
        except Exception as e:
            logger.error("[RL] Error cargando scores: %s", e)


# ─────────────────────────────────────────────────────────────
# REPRODUCTOR CON FADE — hilo separado para evitar bloqueos
# ─────────────────────────────────────────────────────────────

class AudioPlayer:
    """
    Reproductor thread-safe con fade out/in.
    La carga del archivo ocurre en un hilo separado para no bloquear
    el loop principal. El lock solo protege las variables de estado,
    nunca operaciones de I/O.
    """

    def __init__(self):
        self._lock       = threading.Lock()
        self._play_lock  = threading.Lock()   # serializa operaciones pygame
        self._current_path: Optional[Path] = None
        self._started_at:   Optional[float] = None
        self._paused        = False
        self._volume        = 1.0
        self._fade_thread:  Optional[threading.Thread] = None
        self._pending_path: Optional[Path] = None

    # ── API pública ───────────────────────────────────────────

    def play(self, path: Path, fade: bool = True) -> bool:
        """
        Inicia reproducción con fade out de la pista actual (si hay)
        y fade in de la nueva. No bloquea el llamador.
        """
        if not path.exists():
            logger.error("[Audio] Archivo no existe: %s", path)
            return False

        # Cancelar fade anterior si estaba en curso
        self._pending_path = path

        t = threading.Thread(
            target=self._fade_and_play,
            args=(path, fade),
            daemon=True,
            name=f"fade-{path.stem[:20]}"
        )
        t.start()
        self._fade_thread = t
        return True

    def stop(self) -> None:
        self._pending_path = None
        if PYGAME_OK:
            with self._play_lock:
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
        with self._lock:
            self._current_path = None
            self._started_at   = None
            self._paused       = False

    def pause(self) -> None:
        if not PYGAME_OK or self._paused:
            return
        with self._play_lock:
            try:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.pause()
                self._paused = True
            except Exception:
                pass

    def resume(self) -> None:
        if not PYGAME_OK or not self._paused:
            return
        with self._play_lock:
            try:
                pygame.mixer.music.unpause()
                self._paused = False
            except Exception:
                pass

    def set_volume(self, vol: float) -> None:
        self._volume = float(np.clip(vol, 0.0, 1.0))
        if PYGAME_OK:
            with self._play_lock:
                try:
                    pygame.mixer.music.set_volume(self._volume)
                except Exception:
                    pass

    def is_paused(self) -> bool:
        return self._paused

    def is_playing(self) -> bool:
        """True si el mixer está activo (incluye pausa en algunos SO)."""
        if not PYGAME_OK:
            return self._current_path is not None and not self._paused
        try:
            return bool(pygame.mixer.music.get_busy())
        except Exception:
            return False

    def is_audible(self) -> bool:
        """True solo si hay audio sonando (no pausado)."""
        return self.is_playing() and not self._paused

    def seconds_played(self) -> float:
        with self._lock:
            return time.time() - self._started_at if self._started_at else 0.0

    @property
    def current_path(self) -> Optional[Path]:
        with self._lock:
            return self._current_path

    @property
    def current_volume(self) -> float:
        return self._volume

    # ── Implementación de fade ────────────────────────────────

    def _fade_and_play(self, target_path: Path, fade: bool) -> None:
        """
        Ejecutado en hilo separado:
        1. Fade out de la pista actual
        2. Carga la nueva pista
        3. Fade in
        Si se solicita otro cambio mientras está en curso (_pending_path
        cambia), aborta el fade y cede el paso.
        """
        # ── Fade out ──────────────────────────────────────────
        if PYGAME_OK and fade and self.is_playing():
            fade_out_step = self._volume / FADE_STEPS
            step_time     = (FADE_OUT_MS / 1000.0) / FADE_STEPS
            vol = self._volume
            for _ in range(FADE_STEPS):
                if self._pending_path != target_path:
                    return  # otro cambio llegó, ceder
                vol = max(0.0, vol - fade_out_step)
                with self._play_lock:
                    try:
                        pygame.mixer.music.set_volume(vol)
                    except Exception:
                        break
                time.sleep(step_time)

            with self._play_lock:
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass

        if self._pending_path != target_path:
            return

        # ── Cargar nueva pista ────────────────────────────────
        if PYGAME_OK:
            with self._play_lock:
                try:
                    pygame.mixer.music.set_volume(0.0)
                    pygame.mixer.music.load(str(target_path))
                    pygame.mixer.music.play()
                    logger.info("[Audio] ▶ Cargado y reproduciendo: %s",
                                target_path.name)
                except Exception as e:
                    logger.error("[Audio] Error cargando %s: %s",
                                 target_path.name, e)
                    return
        else:
            logger.info("[Audio] [SIM] Reproduciendo: %s", target_path.name)

        with self._lock:
            self._current_path = target_path
            self._started_at   = time.time()
            self._paused       = False

        if self._pending_path != target_path:
            return

        # ── Fade in ───────────────────────────────────────────
        if PYGAME_OK and fade:
            step_time     = (FADE_IN_MS / 1000.0) / FADE_STEPS
            fade_in_step  = self._volume / FADE_STEPS
            vol = 0.0
            for _ in range(FADE_STEPS):
                if self._pending_path != target_path:
                    return
                vol = min(self._volume, vol + fade_in_step)
                with self._play_lock:
                    try:
                        pygame.mixer.music.set_volume(vol)
                    except Exception:
                        break
                time.sleep(step_time)

        logger.debug("[Audio] Fade in completado para %s", target_path.name)


# ─────────────────────────────────────────────────────────────
# ESTABILIZADOR DE ESTADO (hysteresis)
# ─────────────────────────────────────────────────────────────

class StateStabilizer:
    """
    Requiere que un nuevo estado aparezca STATE_STABILITY_COUNT
    veces consecutivas antes de reportarlo como "estable".
    Evita oscilaciones alta_conc → estres → alta_conc por ruido.
    """

    def __init__(self):
        self._history:    Deque[str] = deque(maxlen=STATE_STABILITY_WINDOW)
        self._candidate:  Optional[str] = None
        self._count:      int = 0
        self.stable_state: Optional[str] = None

    def update(self, state: str, confidence: float) -> Tuple[str, bool]:
        """
        Retorna (estado_estable, cambio_confirmado).
        cambio_confirmado = True solo cuando el estado nuevo
        se confirma tras STATE_STABILITY_COUNT ciclos seguidos.
        """
        self._history.append(state)

        if confidence < STATE_CHANGE_CONF_MIN:
            # Confianza baja: no cambia el candidato
            return self.stable_state or state, False

        if state == self._candidate:
            self._count += 1
        else:
            self._candidate = state
            self._count     = 1

        if self._count >= STATE_STABILITY_COUNT:
            changed = (self.stable_state != state)
            self.stable_state = state
            self._count = 0
            return state, changed

        return self.stable_state or state, False

    def get_trend(self) -> str:
        """Estado más frecuente en las últimas N ventanas."""
        if not self._history:
            return "media_conc"
        counts: Dict[str, int] = {}
        for s in self._history:
            counts[s] = counts.get(s, 0) + 1
        return max(counts, key=counts.__getitem__)


# ─────────────────────────────────────────────────────────────
# MOTOR PRINCIPAL
# ─────────────────────────────────────────────────────────────

class MusicEngine:
    """
    Motor musical adaptativo con:
    - Hysteresis de estado (no cambia por cada oscilación)
    - Fade out/in entre pistas
    - Política de cambio inteligente (evalúa tiempo, score, tendencia)
    - Logs explícitos de cada decisión
    """

    def __init__(self, user_id: str = "default"):
        self.user_id   = user_id
        self._library  = MusicLibrary()
        self._rl       = RLScoreManager(user_id)
        self._player   = AudioPlayer()
        self._stable   = StateStabilizer()

        self._current_tid:   Optional[str] = None
        self._current_state: Optional[str] = None
        self._event_history: List[PlaybackEvent] = []

        # Registro de la última decisión de cambio
        self.last_change_reason: str = "Inicio"
        self.last_event:         Optional[PlaybackEvent] = None

        self._change_time: float = 0.0
        self._user_stopped: bool = False

        n = self._library.index()
        logger.info("[Music] Engine listo: %d pistas, usuario='%s'",
                    n, user_id)

    # ── API pública ───────────────────────────────────────────

    def update(self, raw_state: str, confidence: float = 1.0) -> Optional[str]:
        """
        Llamar cada vez que el clasificador produce un estado.
        Aplica hysteresis, evalúa si cambiar, actúa con fade.
        """
        now = time.time()

        if self._user_stopped:
            return self._current_tid

        # 1. Estabilizar estado
        stable_state, state_changed = self._stable.update(raw_state, confidence)

        # 2. Iniciar si no hay nada reproduciendo
        if self._current_tid is None:
            self._do_change(stable_state or "media_conc", now,
                            reason="Inicio de sesión")
            return self._current_tid

        # 3. Evaluar política de cambio
        if state_changed:
            decision, reason = self._change_policy(stable_state, now)
            if decision:
                self._do_change(stable_state, now, reason=reason)
            else:
                logger.info("[Music] Mantiene canción: %s", reason)
                self.last_change_reason = reason

        # 4. Canción terminó naturalmente (no pausa ni stop del usuario)
        elif (self._current_tid is not None
              and not self._player.is_paused()
              and not self._player.is_audible()):
            self._do_change(
                self._stable.get_trend(), now,
                reason="Canción terminada → siguiente por RL")

        return self._current_tid

    def get_current_track(self) -> Optional[Track]:
        if self._current_tid is None:
            return None
        return self._library.get_track(self._current_tid)

    def get_current_score(self) -> Optional[float]:
        return self._rl.get_score(self._current_tid) if self._current_tid else None

    def get_current_ts(self) -> Optional[TrackScore]:
        return self._rl.get_ts(self._current_tid) if self._current_tid else None

    def get_library_summary(self) -> Dict[str, int]:
        return self._library.summary()

    def get_top_tracks(self, n: int = 5) -> List[Tuple[str, float]]:
        return self._rl.top_tracks(n)

    def compute_and_apply_reward(self, state_before: str,
                                  state_after: str) -> float:
        if not self._current_tid:
            return 0.0
        reward = self._transition_reward(state_before, state_after)
        self._rl.update(self._current_tid, reward)
        if self._event_history:
            self._event_history[-1].reward = reward
        return reward

    def force_change(self, state: Optional[str] = None) -> Optional[str]:
        self._user_stopped = False
        state = state or self._current_state or "media_conc"
        self._do_change(state, time.time(), reason="Cambio manual forzado")
        return self._current_tid

    def next_track_same_folder(self) -> Optional[str]:
        """Siguiente pista en la misma carpeta física (focus/calm/...)."""
        self._user_stopped = False
        track = self.get_current_track()
        now = time.time()
        if not track:
            return self.force_change(self._current_state or "media_conc")

        candidates = self._library.get_tracks_in_folder(track.folder)
        if not candidates:
            return self._current_tid

        new_tid = self._rl.select(candidates, self._current_tid)
        new_track = self._library.get_track(new_tid)
        if not new_track:
            return self._current_tid

        state = self._current_state or track.state
        if self._current_tid and self._current_state:
            ev = PlaybackEvent(
                track_id=self._current_tid,
                track_name=track.name,
                state_at_start=self._current_state,
                state_at_end=state,
                started_at=self._change_time,
                duration_played=self._player.seconds_played(),
                change_reason="Siguiente manual (misma carpeta)",
            )
            self._event_history.append(ev)
            self.last_event = ev

        fade = self._current_tid is not None
        self._player.play(new_track.path, fade=fade)
        self._current_tid = new_tid
        self._change_time = now
        self.last_change_reason = f"Siguiente en {track.folder}"
        logger.info("[Music] ⏭ [%s] %s", track.folder, new_track.name)
        return self._current_tid

    @property
    def user_stopped(self) -> bool:
        return self._user_stopped

    def clear_user_stop(self) -> None:
        self._user_stopped = False

    def pause(self):
        self._player.pause()

    def resume(self):
        self._user_stopped = False
        self._player.resume()

    def stop(self, user_initiated: bool = False):
        self._player.stop()
        self._current_tid = None
        if user_initiated:
            self._user_stopped = True
    def set_volume(self, v: float): self._player.set_volume(v)
    def reindex(self) -> int: return self._library.index()
    def seconds_played(self) -> float: return self._player.seconds_played()

    def get_session_stats(self) -> Dict:
        track = self.get_current_track()
        ts    = self.get_current_ts()
        return {
            "total_tracks_played": len(self._event_history),
            "current_state":  self._current_state,
            "current_track":  track.name if track else None,
            "current_bpm":    track.bpm  if track else None,
            "current_score":  self.get_current_score(),
            "play_count":     ts.play_count if ts else 0,
            "reward_sum":     ts.reward_sum if ts else 0.0,
            "last_reward":    ts.last_reward if ts else None,
            "seconds_played": self.seconds_played(),
            "epsilon":        self._rl.epsilon,
            "last_reason":    self.last_change_reason,
            "top_tracks":     self._rl.top_tracks(3),
            "library":        self._library.summary(),
        }

    # ── Política de cambio inteligente ───────────────────────

    def _change_policy(self, new_state: str,
                       now: float) -> Tuple[bool, str]:
        """
        Evalúa si CONVIENE cambiar de canción dado el nuevo estado.
        Retorna (cambiar: bool, motivo: str).

        Jerarquía de decisión:
          1. Sin candidatos → mantener siempre
          2. Mismo estado + score bueno → mantener
          3. Tiempo mínimo no cumplido → mantener (salvo estrés urgente)
          4. Candidatos peores que actual → mantener si lleva poco tiempo
          5. Estado mejoró o empeoró con candidatos disponibles → cambiar
        """
        secs      = self._player.seconds_played()
        cur_score = self.get_current_score() or 0.0

        candidates = self._library.get_tracks_for_state(new_state)

        # ── Regla 1: sin candidatos → imposible cambiar ───────
        if not candidates:
            return False, (
                f"Mantiene — sin pistas para '{new_state}' "
                f"(agrega MP3 a musica/{new_state}/)")

        # ── Regla 2: estado estable con score alto → mantener ─
        if new_state == self._current_state and cur_score > 0.6:
            return False, (
                f"Mantiene '{new_state}' — score {cur_score:.2f}, "
                f"estado estable")

        # ── Regla 3: tiempo mínimo, con excepción urgente ─────
        # Estrés elevado salteamos el mínimo (necesita cambio rápido)
        urgent = (new_state == "estres" and
                  self._current_state not in ("estres", "stress_relief"))
        if secs < RL_MIN_PLAY_SEC and not urgent:
            return False, (
                f"Mantiene — {secs:.0f}s / {RL_MIN_PLAY_SEC}s mínimos "
                f"[{self._current_state} → {new_state}]")

        # ── Regla 4: candidatos peores en poco tiempo → mantener
        best_score = max(self._rl.get_score(c) for c in candidates)
        if best_score < cur_score - 0.5 and secs < 60:
            return False, (
                f"Mantiene — candidatos ({best_score:.2f}) vs "
                f"actual ({cur_score:.2f}), solo {secs:.0f}s")

        # ── Cambio justificado ────────────────────────────────
        urgency = " [URGENTE]" if urgent else ""
        reason  = (
            f"Cambio{urgency}: '{self._current_state}' → '{new_state}' "
            f"| {secs:.0f}s reproducidos | score actual {cur_score:.2f} "
            f"| mejor candidato {best_score:.2f}"
        )
        return True, reason

    # ── Ejecución del cambio ──────────────────────────────────

    def _do_change(self, state: str, now: float, reason: str) -> None:
        candidates = self._library.get_tracks_for_state(state)
        if not candidates:
            logger.warning("[Music] Sin canciones para '%s'", state)
            return

        # Registrar evento del track saliente
        if self._current_tid and self._current_state:
            track_out = self._library.get_track(self._current_tid)
            ev = PlaybackEvent(
                track_id=self._current_tid,
                track_name=track_out.name if track_out else self._current_tid,
                state_at_start=self._current_state,
                state_at_end=state,
                started_at=self._change_time,
                duration_played=self._player.seconds_played(),
                change_reason=reason,
            )
            self._event_history.append(ev)
            self.last_event = ev
            if len(self._event_history) > 100:
                self._event_history = self._event_history[-100:]

        new_tid = self._rl.select(candidates, self._current_tid)
        track   = self._library.get_track(new_tid)
        if not track:
            logger.error("[Music] Track no encontrado: %s", new_tid)
            return

        fade = self._current_tid is not None  # solo fade si había algo antes
        self._player.play(track.path, fade=fade)

        self._current_tid   = new_tid
        self._current_state = state
        self._change_time   = now
        self.last_change_reason = reason

        ts = self._rl.get_ts(new_tid)
        logger.info(
            "[Music] 🎵 [%s] %s | BPM: %s | Score: %.2f | "
            "Reproducciones: %d | Motivo: %s",
            state, track.name,
            f"{track.bpm:.0f}" if track.bpm else "?",
            self._rl.get_score(new_tid),
            ts.play_count,
            reason,
        )

    @staticmethod
    def _transition_reward(before: str, after: str) -> float:
        rank = {"alta_conc": 3, "media_conc": 2, "baja_conc": 1, "estres": 0}
        delta = rank.get(after, 1) - rank.get(before, 1)
        if delta > 0:
            return RL_REWARD_GOOD * delta
        elif delta == 0:
            return RL_REWARD_GOOD * 0.3 if rank.get(after, 1) >= 2 else 0.0
        else:
            return RL_PENALTY_BAD * abs(delta)
