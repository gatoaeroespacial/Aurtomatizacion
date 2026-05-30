"""
================================================================
MYSIGNALS HW v2 — Dashboard en Tiempo Real
GSR + Airflow (respiración) + ECG (electrocardiograma)
================================================================
pip install pyserial matplotlib numpy scipy
python mysignals_dashboard.py
================================================================
"""

import serial
import serial.tools.list_ports
import threading
import time
import csv
import os
from datetime import datetime
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch

# ── Config ─────────────────────────────────────────────────────
PORT      = "COM4"
BAUD      = 115200
WIN_SEC   = 15
SAMPLE_HZ = 50           # 50 Hz (INTERVAL_MS=20 en el .ino)
WIN_LEN   = WIN_SEC * SAMPLE_HZ

# ── Buffers ────────────────────────────────────────────────────
t_buf      = deque(maxlen=WIN_LEN)
gsr_c_buf  = deque(maxlen=WIN_LEN)
air_buf    = deque(maxlen=WIN_LEN)
ecg_buf    = deque(maxlen=WIN_LEN)

data_lock     = threading.Lock()
running       = True
connected     = False
total_samples = 0

# ── CSV ────────────────────────────────────────────────────────
SAVE_DIR = "sesiones_mysignals"
os.makedirs(SAVE_DIR, exist_ok=True)
CSV_PATH = os.path.join(SAVE_DIR,
    f"sesion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# ──────────────────────────────────────────────────────────────
def detect_port():
    for p in serial.tools.list_ports.comports():
        if any(k in (p.description or "")
               for k in ["Arduino", "CH340", "USB Serial", "UART"]):
            return p.device
    pts = serial.tools.list_ports.comports()
    return pts[0].device if pts else PORT

# ──────────────────────────────────────────────────────────────
def serial_reader():
    global running, connected, total_samples
    port = detect_port()
    print(f"[Serial] Conectando en {port} ...")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["ts_real", "ts_ms",
                     "gsr_conductance_uS", "gsr_resistance_ohm",
                     "gsr_voltage_V", "airflow_voltage_V", "ecg_voltage_V"])

        try:
            ser = serial.Serial(port, BAUD, timeout=2)
            time.sleep(2)
            connected = True
            print(f"[Serial] OK en {port}")
            print(f"[CSV]    {CSV_PATH}")

            while running:
                try:
                    raw = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not raw or raw.startswith("#"):
                        continue
                    parts = raw.split(",")
                    if len(parts) != 6:
                        continue

                    ts_ms     = int(parts[0])
                    gsr_cond  = float(parts[1])
                    gsr_res   = float(parts[2])
                    gsr_volt  = float(parts[3])
                    air_volt  = float(parts[4])
                    ecg_volt  = float(parts[5])

                    with data_lock:
                        t_buf.append(ts_ms / 1000.0)
                        gsr_c_buf.append(gsr_cond)
                        air_buf.append(air_volt)
                        ecg_buf.append(ecg_volt)
                        total_samples += 1

                    now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    wr.writerow([now_str, ts_ms,
                                 f"{gsr_cond:.4f}", f"{gsr_res:.2f}",
                                 f"{gsr_volt:.4f}", f"{air_volt:.4f}",
                                 f"{ecg_volt:.4f}"])
                    f.flush()

                except (ValueError, UnicodeDecodeError):
                    continue

        except serial.SerialException as e:
            print(f"[Serial] Error: {e} — modo SIMULACIÓN")
            simulate()

def simulate():
    global total_samples
    t = 0.0
    while running:
        t += 0.02
        # GSR: deriva lenta
        gsr_s = max(0.1, 3.5 + 1.2*np.sin(t*0.04) + np.random.normal(0, 0.05))
        # Airflow: onda respiratoria ~15/min
        air_s = 2.5 + 0.4*np.sin(t * 15/60 * 2*np.pi)
        # ECG: onda cardíaca sintética ~70 bpm
        phase = (t * 70/60) % 1.0
        if phase < 0.05:
            ecg_s = 2.45 + 0.8*np.exp(-((phase-0.025)**2)/0.0002)
        elif phase < 0.15:
            ecg_s = 2.45 - 0.15*np.sin((phase-0.05)*np.pi/0.1)
        else:
            ecg_s = 2.45 + np.random.normal(0, 0.004)

        with data_lock:
            t_buf.append(t)
            gsr_c_buf.append(gsr_s)
            air_buf.append(air_s)
            ecg_buf.append(ecg_s)
            total_samples += 1
        time.sleep(0.02)

# ──────────────────────────────────────────────────────────────
def calc_bpm(ecg_arr, fs=SAMPLE_HZ):
    """Detección de picos R y cálculo de BPM."""
    if len(ecg_arr) < fs * 3:
        return 0
    arr = np.array(ecg_arr)
    arr -= arr.mean()
    # Solo calcular si hay suficiente amplitud (señal real ECG > 0.3V pk-pk)
    if arr.max() - arr.min() < 0.15:
        return 0
    threshold = arr.mean() + 0.65 * arr.std()
    peaks = []
    i = 1
    while i < len(arr) - 1:
        if arr[i] > threshold and arr[i] > arr[i-1] and arr[i] > arr[i+1]:
            peaks.append(i)
            i += int(fs * 0.35)  # refractario 350ms
        else:
            i += 1
    if len(peaks) < 3:
        return 0
    rr = np.diff(peaks) / fs
    rr_valid = rr[(rr > 0.33) & (rr < 1.5)]
    if len(rr_valid) < 2:
        return 0
    return int(60.0 / np.mean(rr_valid))

def calc_resp(air_arr, fs=SAMPLE_HZ):
    """Frecuencia respiratoria en resp/min."""
    if len(air_arr) < fs * 6:
        return 0.0
    a = np.array(air_arr, dtype=float)
    a -= a.mean()
    fft  = np.abs(np.fft.rfft(a))
    freq = np.fft.rfftfreq(len(a), d=1.0/fs)
    mask = (freq >= 0.1) & (freq <= 0.6)
    if not mask.any():
        return 0.0
    return round(freq[mask][np.argmax(fft[mask])] * 60, 1)

def estado_cognitivo(gsr_mean, bpm):
    """Heurística de nivel de concentración."""
    if gsr_mean <= 0:
        return "Sin contacto GSR", "#555555"
    score = 0
    if 1.5 <= gsr_mean <= 6.0:
        score += 3
    elif gsr_mean < 1.5:
        score += 1
    else:
        score += 2
    if 0 < bpm <= 75:
        score += 3
    elif 75 < bpm <= 90:
        score += 2
    elif bpm > 90:
        score += 1
    if score >= 5:
        return "Concentración alta", "#3FB950"
    if score >= 3:
        return "Concentración media", "#F0A500"
    return "Baja concentración", "#E05050"

# ──────────────────────────────────────────────────────────────
def build_dashboard():
    plt.rcParams.update({
        "figure.facecolor": "#0D1117",
        "axes.facecolor":   "#161B22",
        "axes.edgecolor":   "#30363D",
        "axes.labelcolor":  "#8B949E",
        "axes.grid":        True,
        "grid.color":       "#21262D",
        "grid.linewidth":   0.5,
        "xtick.color":      "#8B949E",
        "ytick.color":      "#8B949E",
        "text.color":       "#E6EDF3",
        "font.family":      "monospace",
    })

    fig = plt.figure(figsize=(17, 11))
    fig.canvas.manager.set_window_title(
        "MySignals HW v2 — GSR · Airflow · ECG")

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.55, wspace=0.32,
        left=0.07, right=0.97,
        top=0.91, bottom=0.06
    )

    ax_gsr = fig.add_subplot(gs[0, :])
    ax_air = fig.add_subplot(gs[1, :])
    ax_ecg = fig.add_subplot(gs[2, :])

    ax_m1 = fig.add_subplot(gs[3, 0])  # BPM
    ax_m2 = fig.add_subplot(gs[3, 1])  # Respiración
    ax_m3 = fig.add_subplot(gs[3, 2])  # Estado cognitivo
    for ax in [ax_m1, ax_m2, ax_m3]:
        ax.axis("off")

    def style(ax, title, ylabel, color):
        ax.set_title(title, color=color, fontsize=11,
                     fontweight="bold", loc="left", pad=5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#30363D")
        ax.spines["bottom"].set_color("#30363D")

    style(ax_gsr, "GSR — Respuesta Galvánica de la Piel",
          "Conductancia (µS)", "#58A6FF")
    style(ax_air, "AIRFLOW — Flujo de Aire / Respiración",
          "Voltaje (V)", "#3FB950")
    style(ax_ecg, "ECG — Electrocardiograma (señal cruda)",
          "Voltaje (V)", "#FF6B6B")

    # Zona óptima GSR
    ax_gsr.axhspan(1.5, 6.0, alpha=0.07, color="#58A6FF")

    line_gsr, = ax_gsr.plot([], [], color="#58A6FF", lw=1.2)
    line_air, = ax_air.plot([], [], color="#3FB950", lw=1.0)
    line_ecg, = ax_ecg.plot([], [], color="#FF6B6B", lw=0.8)

    # Cabecera
    fig.text(0.5, 0.968,
             "● MYSIGNALS HW v2 — GSR · AIRFLOW · ECG",
             ha="center", fontsize=13, fontweight="bold",
             color="#E6EDF3", fontfamily="monospace")
    status_txt = fig.text(0.5, 0.948, "Conectando...",
                          ha="center", fontsize=9,
                          color="#8B949E", fontfamily="monospace")

    def metric_box(ax, label, init, color):
        r = FancyBboxPatch((0.05, 0.05), 0.90, 0.90,
                           boxstyle="round,pad=0.02",
                           lw=1, edgecolor=color,
                           facecolor="#161B22",
                           transform=ax.transAxes, clip_on=False)
        ax.add_patch(r)
        ax.text(0.5, 0.80, label, ha="center", va="center",
                transform=ax.transAxes, fontsize=9,
                color="#8B949E", fontfamily="monospace")
        v = ax.text(0.5, 0.38, init, ha="center", va="center",
                    transform=ax.transAxes, fontsize=26,
                    fontweight="bold", color=color,
                    fontfamily="monospace")
        return v

    txt_bpm  = metric_box(ax_m1, "FRECUENCIA CARDÍACA", "-- BPM",    "#FF6B6B")
    txt_resp = metric_box(ax_m2, "RESPIRACIÓN",          "-- r/min",  "#3FB950")
    txt_est  = metric_box(ax_m3, "ESTADO COGNITIVO",     "Calculando...", "#F0A500")

    frame_n = [0]

    def update(frame):
        frame_n[0] += 1

        with data_lock:
            if len(t_buf) < 5:
                return line_gsr, line_air, line_ecg
            t_arr   = np.array(t_buf)
            g_arr   = np.array(gsr_c_buf)
            a_arr   = np.array(air_buf)
            e_arr   = np.array(ecg_buf)
            n       = total_samples

        line_gsr.set_data(t_arr, g_arr)
        line_air.set_data(t_arr, a_arr)
        line_ecg.set_data(t_arr, e_arr)

        t_max = t_arr[-1]
        t_min = max(0, t_max - WIN_SEC)
        for ax in [ax_gsr, ax_air, ax_ecg]:
            ax.set_xlim(t_min, t_max + 0.3)

        def ylim(ax, arr, margin=0.15):
            v = arr[np.isfinite(arr)]
            if len(v) == 0: return
            lo, hi = v.min(), v.max()
            rng = max(hi - lo, 0.01)
            ax.set_ylim(lo - rng*margin, hi + rng*margin)

        gsr_valid = g_arr[g_arr > 0]
        ylim(ax_gsr, gsr_valid) if len(gsr_valid) > 0 else ax_gsr.set_ylim(0, 5)
        ylim(ax_air, a_arr)
        ylim(ax_ecg, e_arr)

        # Métricas cada 25 frames (~0.5s)
        if frame_n[0] % 25 == 0:
            bpm  = calc_bpm(list(ecg_buf))
            resp = calc_resp(list(air_buf))
            gsr_mean = float(np.mean(gsr_valid)) if len(gsr_valid) > 0 else -1

            lbl, color = estado_cognitivo(gsr_mean, bpm)

            txt_bpm.set_text(f"{bpm} BPM" if bpm > 0 else "-- BPM")
            txt_resp.set_text(f"{resp:.1f} r/min" if resp > 0 else "-- r/min")
            txt_est.set_text(lbl)
            txt_est.set_color(color)

            m, s = divmod(int(t_arr[-1]), 60)
            conn = "🟢 ARDUINO" if connected else "🟡 SIMULACIÓN"
            status_txt.set_text(
                f"{conn}  |  Sesión: {m:02d}:{s:02d}  |  Muestras: {n:,}"
                f"  |  GSR: {gsr_mean:.2f} µS  |  BPM: {bpm}"
            )

        return line_gsr, line_air, line_ecg

    ani = FuncAnimation(fig, update, interval=100,
                        blit=False, cache_frame_data=False)
    plt.show()
    return ani

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("  MYSIGNALS HW v2 — GSR + Airflow + ECG")
    print()
    print("  Pines (MySignals.h oficial):")
    print("    A1 → ECG  |  A2 → Airflow  |  A3 → GSR")
    print()
    print("  Carga: mysignals_3sensores.ino")
    print(f"  CSV:   {CSV_PATH}")
    print("=" * 62)

    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()

    try:
        ani = build_dashboard()
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        print(f"\n[Listo] CSV: {CSV_PATH}")
