"""
Tareas cognitivas experimentales: N-Back 2-back y Stroop.
Lógica pura + persistencia de sesiones (sin Streamlit).
"""

from __future__ import annotations

import json
import random
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    EXPERIMENTS_DIR,
    EXP_GRADE_EXCELLENT,
    EXP_GRADE_FAIR,
    EXP_GRADE_GOOD,
    EXP_MUSIC_CONDITIONS,
    EXP_NBACK_ISI_SEC,
    EXP_NBACK_OMISSION_PENALTY,
    EXP_NBACK_STIM_SEC,
    EXP_NBACK_TARGET_RATE,
    EXP_NBACK_TRIALS,
    EXP_STROOP_ACC_WEIGHT,
    EXP_STROOP_ISI_SEC,
    EXP_STROOP_RT_IDEAL_MS,
    EXP_STROOP_RT_SLOW_MS,
    EXP_STROOP_SPEED_WEIGHT,
    EXP_STROOP_STIM_SEC,
    EXP_STROOP_TRIALS,
)

STIMULI = list("BCDFGHJKLMNPQRSTVWXYZ")
STROOP_WORDS = ("ROJO", "AZUL", "VERDE", "AMARILLO")
STROOP_COLORS = {
    "ROJO": "#e05252",
    "AZUL": "#4d9de0",
    "VERDE": "#40bf80",
    "AMARILLO": "#e5c347",
}


@dataclass
class PhysioSnapshot:
    ts: float
    bpm: float = 0.0
    rmssd: float = 0.0
    sdnn: float = 0.0
    scl: float = 0.0
    resp_rate: float = 0.0
    resp_reg: float = 0.0
    cog_state: str = ""


@dataclass
class TrialRecord:
    trial_index: int
    stimulus: str
    response: str
    correct: bool
    rt_ms: Optional[float]
    error_type: str  # none | miss | false_alarm | wrong | omission | timeout | practice
    position: int = 0
    reference_2back: Optional[str] = None
    correct_answer: str = ""
    scorable: bool = True


@dataclass
class TaskMetrics:
    accuracy: float
    mean_rt_ms: float
    median_rt_ms: float
    errors: int
    hits: int = 0
    misses: int = 0
    false_alarms: int = 0
    wrong: int = 0
    timeouts: int = 0
    n_trials: int = 0


@dataclass
class ExperimentRun:
    run_id: str
    user_id: str
    task: str
    music_condition: str
    started_at: float
    finished_at: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    physio_timeseries: List[Dict[str, Any]] = field(default_factory=list)
    physio_summary: Dict[str, float] = field(default_factory=dict)
    trials: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def physio_from_state(state: dict) -> PhysioSnapshot:
    return PhysioSnapshot(
        ts=float(state.get("ts", time.time())),
        bpm=float(state.get("bpm", 0) or 0),
        rmssd=float(state.get("rmssd", 0) or 0),
        sdnn=float(state.get("sdnn", 0) or 0),
        scl=float(state.get("scl", 0) or 0),
        resp_rate=float(state.get("resp_rate", 0) or 0),
        resp_reg=float(state.get("resp_reg", 0) or 0),
        cog_state=str(state.get("cog_state", "")),
    )


def summarize_physio(samples: List[PhysioSnapshot]) -> Dict[str, float]:
    if not samples:
        return {}
    def _mean(attr: str) -> float:
        vals = [getattr(s, attr) for s in samples if getattr(s, attr, 0) > 0]
        return float(statistics.mean(vals)) if vals else 0.0

    return {
        "bpm_mean": round(_mean("bpm"), 2),
        "rmssd_mean": round(_mean("rmssd"), 2),
        "sdnn_mean": round(_mean("sdnn"), 2),
        "gsr_mean": round(_mean("scl"), 3),
        "resp_mean": round(_mean("resp_rate"), 2),
        "n_samples": len(samples),
    }


def compute_metrics(trials: List[TrialRecord]) -> TaskMetrics:
    """Métricas genéricas (Stroop)."""
    scored = [t for t in trials if t.scorable]
    if not scored:
        scored = trials
    n = len(scored)
    correct = sum(1 for t in scored if t.correct)
    rts = [
        t.rt_ms for t in scored
        if t.rt_ms is not None and t.response not in ("none", "omission", "")
    ]
    errors = sum(1 for t in scored if not t.correct)
    return TaskMetrics(
        accuracy=round(correct / n, 4) if n else 0.0,
        mean_rt_ms=round(statistics.mean(rts), 1) if rts else 0.0,
        median_rt_ms=round(statistics.median(rts), 1) if rts else 0.0,
        errors=errors,
        n_trials=n,
        timeouts=sum(1 for t in scored if t.error_type in ("timeout", "omission")),
    )


def compute_stroop_metrics_dict(trials: List[TrialRecord]) -> Dict[str, Any]:
    """Métricas Stroop + calificación compuesta (precisión + velocidad)."""
    tm = compute_metrics(trials)
    m = asdict(tm)
    m.update(compute_task_grade("stroop", m))
    return m


def nback_correct_answer(letters: List[str], position: int) -> Optional[str]:
    """
    Respuesta correcta 2-back en posición 1-based.
    Posiciones 1–2: None (sin juicio 2-back).
  Posición n≥3: match si letra[n] == letra[n-2].
    """
    if position < 3:
        return None
    i = position - 1
    return "match" if letters[i] == letters[i - 2] else "no_match"


def build_nback_trial(letters: List[str], index: int) -> Dict[str, Any]:
    """Metadatos de un ensayo a partir de la secuencia de letras."""
    pos = index + 1
    letter = letters[index]
    ref = letters[index - 2] if index >= 2 else None
    ans = nback_correct_answer(letters, pos)
    return {
        "index": index,
        "position": pos,
        "stimulus": letter,
        "reference_2back": ref,
        "correct_answer": ans,
        "scorable": index >= 2,
    }


def generate_nback_sequence(
    n_trials: int,
    target_rate: float = EXP_NBACK_TARGET_RATE,
) -> List[Dict[str, Any]]:
    """
    Secuencia 2-back con proporción controlada de MATCH (≈30%).
    Posiciones 1–2: aleatorias (calentamiento, no puntuables).
    Desde posición 3: exactamente n_match coincidencias 2-back.
    """
    if n_trials < 3:
        raise ValueError("N-Back requiere al menos 3 estímulos")

    letters: List[str] = [
        random.choice(STIMULI),
        random.choice(STIMULI),
    ]
    n_scorable = n_trials - 2
    n_match = int(round(n_scorable * target_rate))
    n_match = max(0, min(n_match, n_scorable))
    flags = [True] * n_match + [False] * (n_scorable - n_match)
    random.shuffle(flags)

    for want_match in flags:
        if want_match:
            letters.append(letters[-2])
        else:
            forbidden = {letters[-2]}
            pool = [c for c in STIMULI if c not in forbidden]
            letters.append(random.choice(pool))

    trials = [build_nback_trial(letters, i) for i in range(n_trials)]
    return trials


def score_nback_response(
    trial: Dict[str, Any],
    user_response: str,
    rt_ms: Optional[float],
) -> TrialRecord:
    """
    user_response: 'match' | 'no_match' | 'omission'
    """
    pos = trial["position"]
    letter = trial["stimulus"]
    ref = trial.get("reference_2back")
    correct_ans = trial.get("correct_answer")
    scorable = trial.get("scorable", pos >= 3)

    if not scorable:
        return TrialRecord(
            trial_index=trial["index"],
            position=pos,
            stimulus=letter,
            reference_2back=ref,
            correct_answer="n/a",
            response=user_response,
            correct=True,
            rt_ms=None,
            error_type="practice",
            scorable=False,
        )

    if user_response == "omission":
        return TrialRecord(
            trial_index=trial["index"],
            position=pos,
            stimulus=letter,
            reference_2back=ref,
            correct_answer=correct_ans or "",
            response="omission",
            correct=False,
            rt_ms=None,
            error_type="omission",
            scorable=True,
        )

    correct = user_response == correct_ans
    if correct:
        err = "none"
    elif user_response == "match":
        err = "false_alarm"
    else:
        err = "miss"

    return TrialRecord(
        trial_index=trial["index"],
        position=pos,
        stimulus=letter,
        reference_2back=ref,
        correct_answer=correct_ans or "",
        response=user_response,
        correct=correct,
        rt_ms=round(rt_ms, 1) if rt_ms is not None else None,
        error_type=err,
        scorable=True,
    )


def compute_nback_metrics(trials: List[TrialRecord]) -> Dict[str, Any]:
    """Métricas estándar 2-back (solo ensayos puntuables, posición ≥ 3)."""
    scored = [t for t in trials if t.scorable]
    n = len(scored)
    if n == 0:
        return {
            "n_trials": 0,
            "n_correct": 0,
            "n_errors": 0,
            "accuracy": 0.0,
            "mean_rt_ms": 0.0,
            "median_rt_ms": 0.0,
            "omissions": 0,
            "false_alarms": 0,
            "misses": 0,
        }
    n_correct = sum(1 for t in scored if t.correct)
    omissions = sum(1 for t in scored if t.error_type == "omission")
    rts = [
        t.rt_ms for t in scored
        if t.correct and t.rt_ms is not None
    ]
    rts_all = [
        t.rt_ms for t in scored
        if t.rt_ms is not None and t.response != "omission"
    ]
    out = {
        "n_trials": n,
        "n_correct": n_correct,
        "n_errors": n - n_correct,
        "accuracy": round(n_correct / n, 4),
        "mean_rt_ms": round(statistics.mean(rts_all), 1) if rts_all else 0.0,
        "median_rt_ms": round(statistics.median(rts_all), 1) if rts_all else 0.0,
        "mean_rt_correct_ms": round(statistics.mean(rts), 1) if rts else 0.0,
        "omissions": omissions,
        "false_alarms": sum(1 for t in scored if t.error_type == "false_alarm"),
        "misses": sum(1 for t in scored if t.error_type == "miss"),
    }
    out.update(compute_task_grade("nback_2", out))
    return out


def grade_label(score: float) -> str:
    if score >= EXP_GRADE_EXCELLENT:
        return "Excelente"
    if score >= EXP_GRADE_GOOD:
        return "Bueno"
    if score >= EXP_GRADE_FAIR:
        return "Regular"
    return "Bajo"


def compute_task_grade(task: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calificación 0–100 por tarea.
    N-Back: precisión − penalización por omisiones.
    Stroop: 65 % precisión + 35 % velocidad (RT vs referencia).
    """
    if task == "nback_2":
        acc = float(metrics.get("accuracy", 0) or 0)
        n = int(metrics.get("n_trials", 0) or 0)
        om = int(metrics.get("omissions", 0) or 0)
        base = acc * 100.0
        if n > 0:
            base -= (om / n) * EXP_NBACK_OMISSION_PENALTY
        score = float(max(0.0, min(100.0, base)))
        detail = f"Precisión {acc*100:.1f}%"
        if om:
            detail += f", -{(om/n)*EXP_NBACK_OMISSION_PENALTY:.1f} pts omisiones"
    else:
        acc = float(metrics.get("accuracy", 0) or 0)
        rt = float(metrics.get("mean_rt_ms", 0) or 0)
        acc_pts = acc * 100.0
        if rt <= 0:
            speed_pts = 50.0
        elif rt <= EXP_STROOP_RT_IDEAL_MS:
            speed_pts = 100.0
        elif rt >= EXP_STROOP_RT_SLOW_MS:
            speed_pts = 0.0
        else:
            speed_pts = 100.0 * (
                (EXP_STROOP_RT_SLOW_MS - rt)
                / (EXP_STROOP_RT_SLOW_MS - EXP_STROOP_RT_IDEAL_MS)
            )
        score = float(max(0.0, min(100.0,
            EXP_STROOP_ACC_WEIGHT * acc_pts
            + EXP_STROOP_SPEED_WEIGHT * speed_pts
        )))
        detail = (
            f"Precisión {acc*100:.1f}% ({EXP_STROOP_ACC_WEIGHT*100:.0f}%) + "
            f"velocidad {speed_pts:.0f}/100 ({EXP_STROOP_SPEED_WEIGHT*100:.0f}%), "
            f"RT {rt:.0f} ms"
        )

    label = grade_label(score)
    return {
        "score": round(score, 1),
        "grade": label,
        "display": f"{score:.1f}/100 — {label}",
        "detail": detail,
    }


def validate_nback_logic_demo() -> List[Dict[str, Any]]:
    """
    Validación paso a paso (ejemplo del informe):
    A, B, A → MATCH; F → NO MATCH; F → NO MATCH
    """
    letters = ["A", "B", "A", "F", "F"]
    steps = []
    for pos in range(1, len(letters) + 1):
        ans = nback_correct_answer(letters, pos)
        label = {
            None: "sin juicio 2-back",
            "match": "MATCH (Sí)",
            "no_match": "NO MATCH (No)",
        }.get(ans, "")
        steps.append({
            "position": pos,
            "letter": letters[pos - 1],
            "reference_2back": letters[pos - 3] if pos >= 3 else "—",
            "correct_answer": label,
        })
    return steps


# ── Stroop ────────────────────────────────────────────────────

def generate_stroop_trials(n_trials: int) -> List[Dict[str, Any]]:
    trials = []
    for i in range(n_trials):
        word = random.choice(STROOP_WORDS)
        congruent = random.random() < 0.5
        if congruent:
            ink = word
        else:
            opts = [w for w in STROOP_WORDS if w != word]
            ink = random.choice(opts)
        trials.append({
            "index": i,
            "word": word,
            "ink": ink,
            "congruent": congruent,
            "hex": STROOP_COLORS[ink],
        })
    return trials


def score_stroop_trial(trial: dict, answer: str, rt_ms: float) -> TrialRecord:
    correct = answer == trial["ink"]
    err = "none" if correct else "wrong"
    return TrialRecord(
        trial_index=trial["index"],
        stimulus=f"{trial['word']}/{trial['ink']}",
        response=answer,
        correct=correct,
        rt_ms=round(rt_ms, 1),
        error_type=err,
    )


# ── Persistencia ──────────────────────────────────────────────

class ExperimentStore:
    def __init__(self, user_id: str = "estudiante"):
        self.user_id = user_id
        self._path = EXPERIMENTS_DIR / f"exp_{user_id}.json"
        self._data = self._load()

    def _load(self) -> dict:
        if not self._path.exists():
            return {"user_id": self.user_id, "runs": []}
        try:
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"user_id": self.user_id, "runs": []}

    def save_run(self, run: ExperimentRun) -> Path:
        self._data.setdefault("runs", []).append(run.to_dict())
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        self._export_comparison_csv()
        return self._path

    def list_runs(self) -> List[dict]:
        return list(self._data.get("runs", []))

    def _export_comparison_csv(self) -> None:
        import csv
        runs = self._data.get("runs", [])
        if not runs:
            return
        csv_path = EXPERIMENTS_DIR / f"comparativa_{self.user_id}.csv"
        fields = [
            "run_id", "task", "music_condition", "started_at",
            "accuracy", "mean_rt_ms", "errors", "n_correct", "omissions",
            "score", "grade",
            "bpm_mean", "rmssd_mean", "gsr_mean", "resp_mean",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in runs:
                m = r.get("metrics", {})
                p = r.get("physio_summary", {})
                w.writerow({
                    "run_id": r.get("run_id"),
                    "task": r.get("task"),
                    "music_condition": r.get("music_condition"),
                    "started_at": r.get("started_at"),
                    "accuracy": m.get("accuracy"),
                    "mean_rt_ms": m.get("mean_rt_ms"),
                    "errors": m.get("n_errors", m.get("errors")),
                    "n_correct": m.get("n_correct"),
                    "omissions": m.get("omissions"),
                    "score": m.get("score"),
                    "grade": m.get("grade"),
                    "bpm_mean": p.get("bpm_mean"),
                    "rmssd_mean": p.get("rmssd_mean"),
                    "gsr_mean": p.get("gsr_mean"),
                    "resp_mean": p.get("resp_mean"),
                })


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def validate_condition(c: str) -> str:
    if c not in EXP_MUSIC_CONDITIONS:
        raise ValueError(f"Condición inválida: {c}")
    return c
