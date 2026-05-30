"""
================================================================
CONFIG.PY — Configuración centralizada del sistema
Sistema Adaptativo de Recomendación Musical
================================================================
REGLA: Todo valor magic number vive aquí. Nada hardcodeado
en los módulos. Cambia aquí y se propaga a todo el sistema.
================================================================
"""

from pathlib import Path

# ── Rutas base ────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
MUSIC_DIR     = BASE_DIR / "musica"
SESSION_DIR   = BASE_DIR / "sesiones"
MODEL_DIR     = BASE_DIR / "modelos"
LOG_DIR       = BASE_DIR / "logs"
PROFILE_DIR   = BASE_DIR / "perfiles"

for d in [MUSIC_DIR, SESSION_DIR, MODEL_DIR, LOG_DIR, PROFILE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Serial / Arduino ──────────────────────────────────────────
SERIAL_PORT   = "COM4"          # Cambia a /dev/ttyUSB0 en Linux
SERIAL_BAUD   = 115200
SERIAL_TIMEOUT = 2.0            # segundos
SERIAL_FIELDS  = 6              # ts_ms, gsr_c, gsr_r, gsr_v, air_v, ecg_v
SAMPLE_HZ      = 50             # Hz — coincide con INTERVAL_MS=20 en .ino

# ── Ventana de procesamiento ──────────────────────────────────
WINDOW_SECONDS  = 30            # Ventana deslizante principal
WINDOW_SAMPLES  = WINDOW_SECONDS * SAMPLE_HZ   # 1500 muestras
WINDOW_STEP_SEC = 5             # cada cuántos segundos se re-evalúa el estado

# ── Pines físicos (referencia, ya fijados en el .ino) ─────────
PIN_ECG     = "A1"
PIN_AIRFLOW = "A2"
PIN_GSR     = "A3"

# ── Umbrales fisiológicos (literatura científica) ─────────────
# GSR — Nourbakhsh et al. 2012
GSR_OPTIMAL_LOW  = 2.0   # µS — concentración óptima
GSR_OPTIMAL_HIGH = 8.0
GSR_STRESS_HIGH  = 12.0  # µS — estrés elevado
GSR_RELAX_LOW    = 0.5   # µS — demasiado relajado

# ECG / HRV
BPM_CALM_LOW   = 58
BPM_CALM_HIGH  = 80
BPM_FOCUS_HIGH = 90
BPM_STRESS_MIN = 90
RMSSD_HEALTHY  = 30.0   # ms — tono vagal saludable
RMSSD_STRESS   = 20.0   # ms — posible estrés
NN50_FOCUS_MIN = 5      # n° de intervalos NN que difieren > 50ms

# Respiración
RESP_OPTIMAL_LOW  = 10.0  # resp/min
RESP_OPTIMAL_HIGH = 16.0
RESP_STRESS_HIGH  = 22.0
RESP_RELAX_LOW    = 8.0

# ── Clasificador ─────────────────────────────────────────────
CLASSIFIER_HEURISTIC_UNTIL = 50   # muestras antes de activar RF
CLASSIFIER_STATES = ["alta_conc", "media_conc", "baja_conc", "estres"]
RF_N_ESTIMATORS   = 100
RF_MAX_DEPTH      = 6
RF_RETRAIN_EVERY  = 25            # ventanas procesadas

# ── Motor musical ─────────────────────────────────────────────
MUSIC_FOLDERS = {
    "alta_conc"  : MUSIC_DIR / "focus",
    "media_conc" : MUSIC_DIR / "calm",
    "baja_conc"  : MUSIC_DIR / "energize",
    "estres"     : MUSIC_DIR / "stress_relief",
}

# BPM objetivo por estado (Gonzalez & Aiello 2019; Hasegawa 2004)
MUSIC_BPM_TARGET = {
    "alta_conc"  : (60, 80),   # foco profundo
    "media_conc" : (70, 90),   # concentración media
    "baja_conc"  : (90, 120),  # energizar
    "estres"     : (45, 70),   # alivio de estrés
}

# RL — epsilon-greedy
RL_EPSILON_START  = 0.15   # 15% exploración
RL_EPSILON_MIN    = 0.05
RL_EPSILON_DECAY  = 0.995
RL_REWARD_GOOD    = 0.15
RL_PENALTY_BAD    = -0.08
RL_REWARD_CLIP    = (-2.0, 2.0)
RL_MIN_PLAY_SEC   = 45     # segundos mínimos antes de evaluar
RL_EVAL_DELAY_SEC = 60     # espera tras cambio de canción

# ── LLM ───────────────────────────────────────────────────────
LLM_MODEL         = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS    = 1000
LLM_INSIGHT_EVERY = 10     # cada N ventanas procesadas
LLM_ENDPOINT      = "https://api.anthropic.com/v1/messages"

# ── Señal / Filtros ───────────────────────────────────────────
ECG_BANDPASS_LOW  = 0.5    # Hz
ECG_BANDPASS_HIGH = 40.0   # Hz
GSR_LOWPASS_FREQ  = 1.0    # Hz — GSR es señal lenta
AIRFLOW_LOWPASS   = 2.0    # Hz

# Detección de picos R
RPEAK_MIN_HEIGHT_STD = 0.5    # umbral = media + N*std de la señal
RPEAK_REFRACTORY_MS  = 300    # ms — período refractario fisiológico
RPEAK_MIN_BPM        = 35
RPEAK_MAX_BPM        = 200

# Rechazo de artefactos
ARTIFACT_GSR_MIN    = 0.05   # µS — por debajo = sin contacto
ARTIFACT_GSR_MAX    = 100.0  # µS — por encima = artefacto
ARTIFACT_ECG_CLAMP  = (0.0, 5.0)  # V — rango válido ADC
ARTIFACT_AIR_CLAMP  = (0.0, 5.0)

# ── Dashboard ─────────────────────────────────────────────────
DASH_UPDATE_MS    = 120    # intervalo de refresco matplotlib
DASH_WINDOW_SEC   = 15     # ventana visible en gráficas
DASH_METRIC_EVERY = 25     # frames entre actualizaciones de métricas

# ── Logging ───────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
