"""
================================================================
DIAGNÓSTICO DE PINES — MySignals HW v2
Lee A0-A5 y muestra cuál responde al sensor Airflow
================================================================
Uso:
    pip install pyserial matplotlib numpy
    python diagnostico_pines.py

Instrucciones:
    1. Carga diagnostico_pines.ino en el Arduino
    2. Corre este script
    3. Conecta y desconecta el sensor Airflow
    4. Observa cuál gráfica cambia — ese es el pin correcto
    5. También agita / aprieta el sensor físicamente
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

# ── Config ────────────────────────────────────────────────────
PORT     = "COM4"
BAUD     = 115200
WIN_LEN  = 200   # últimas 200 muestras visibles (~10 segundos)

# ── Buffers para los 6 pines ──────────────────────────────────
PIN_NAMES = ["A0", "A1", "A2", "A3", "A4", "A5"]
PIN_COLORS = ["#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#FF9F43", "#C77DFF"]
buffers = {name: deque([512]*WIN_LEN, maxlen=WIN_LEN) for name in PIN_NAMES}

# ── Estadísticas en vivo ──────────────────────────────────────
stats = {name: {"min": 512, "max": 512, "mean": 512.0, "var": 0.0}
         for name in PIN_NAMES}

data_lock   = threading.Lock()
running     = True
connected   = False
total_lines = 0

# ── CSV de guardado ───────────────────────────────────────────
SAVE_DIR = "diagnostico_sesiones"
os.makedirs(SAVE_DIR, exist_ok=True)
ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = os.path.join(SAVE_DIR, f"pines_{ts_str}.csv")

# ─────────────────────────────────────────────────────────────
# DETECCIÓN DE PUERTO
# ─────────────────────────────────────────────────────────────
def detect_port():
    for p in serial.tools.list_ports.comports():
        desc = p.description or ""
        if any(k in desc for k in ["Arduino", "CH340", "USB Serial", "UART"]):
            return p.device
    ports = serial.tools.list_ports.comports()
    return ports[0].device if ports else PORT

# ─────────────────────────────────────────────────────────────
# THREAD SERIAL
# ─────────────────────────────────────────────────────────────
def serial_reader():
    global running, connected, total_lines

    port = detect_port()
    print(f"[Serial] Conectando en {port} ...")

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_real", "ts_ms",
                         "A0", "A1", "A2", "A3", "A4", "A5"])

        try:
            ser = serial.Serial(port, BAUD, timeout=2)
            time.sleep(2)
            connected = True
            print(f"[Serial] ✓ Conectado en {port}")
            print(f"[CSV]    Guardando en {CSV_PATH}")

            while running:
                try:
                    raw = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not raw or raw.startswith("#"):
                        continue

                    parts = raw.split(",")
                    if len(parts) != 7:
                        continue

                    ts_ms = int(parts[0])
                    vals  = [int(p) for p in parts[1:]]  # A0..A5

                    with data_lock:
                        for i, name in enumerate(PIN_NAMES):
                            buffers[name].append(vals[i])
                            arr = np.array(buffers[name])
                            stats[name] = {
                                "min":  int(arr.min()),
                                "max":  int(arr.max()),
                                "mean": float(arr.mean()),
                                "var":  float(arr.var()),
                            }
                        total_lines += 1

                    now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    writer.writerow([now_str, ts_ms] + vals)
                    f.flush()

                except (ValueError, UnicodeDecodeError):
                    continue

        except serial.SerialException as e:
            print(f"[Serial] Error: {e}")
            print("[Serial] Modo SIMULACIÓN activado")
            simulate()

def simulate():
    """Datos simulados: un solo pin cambia para demostrar la detección."""
    global total_lines
    t = 0.0
    while running:
        t += 0.05
        vals = [512, 512, 512, 512, 512, 512]
        # Simula que A1 es el Airflow
        vals[1] = int(512 + 180 * np.sin(t * 0.3))
        vals[2] = int(512 + np.random.normal(0, 3))  # GSR ruido

        with data_lock:
            for i, name in enumerate(PIN_NAMES):
                buffers[name].append(vals[i])
                arr = np.array(buffers[name])
                stats[name] = {
                    "min":  int(arr.min()),
                    "max":  int(arr.max()),
                    "mean": float(arr.mean()),
                    "var":  float(arr.var()),
                }
            total_lines += 1
        time.sleep(0.05)

# ─────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────
def build_dashboard():
    plt.rcParams.update({
        "figure.facecolor": "#0D1117",
        "axes.facecolor":   "#0D1117",
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
    fig.canvas.manager.set_window_title("Diagnóstico de Pines — MySignals HW v2")

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        hspace=0.55, wspace=0.32,
        left=0.07, right=0.97,
        top=0.91,  bottom=0.06
    )

    axes  = []
    lines = []

    for idx, name in enumerate(PIN_NAMES):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        color = PIN_COLORS[idx]

        ax.set_title(f"Pin {name}", color=color,
                     fontsize=11, fontweight="bold", loc="left", pad=4)
        ax.set_ylim(0, 1023)
        ax.set_xlim(0, WIN_LEN)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#30363D")
        ax.spines["bottom"].set_color("#30363D")
        ax.set_ylabel("ADC (0–1023)", fontsize=8)

        # Zona de referencia central (valores en reposo ~512)
        ax.axhspan(490, 535, alpha=0.06, color=color)

        lne, = ax.plot(range(WIN_LEN), list(buffers[name]),
                       color=color, lw=1.0, alpha=0.9)
        # Texto de estadísticas dentro del gráfico
        stat_txt = ax.text(
            0.99, 0.97, "",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8, color=color,
            fontfamily="monospace"
        )
        # Indicador de actividad (círculo que se ilumina si hay varianza alta)
        act_txt = ax.text(
            0.01, 0.97, "●",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=14, color="#333333",
            fontfamily="monospace"
        )

        axes.append((ax, stat_txt, act_txt))
        lines.append(lne)

    # Título y estado
    fig.text(0.5, 0.965,
             "DIAGNÓSTICO DE PINES ANALÓGICOS — Identificar sensor Airflow",
             ha="center", fontsize=13, fontweight="bold", color="#E6EDF3",
             fontfamily="monospace")

    instrucciones = fig.text(
        0.5, 0.945,
        "Conecta/desconecta el sensor  •  Aprieta la banda  •  El pin que cambia es el Airflow",
        ha="center", fontsize=9, color="#8B949E", fontfamily="monospace"
    )

    status_txt = fig.text(
        0.5, 0.927,
        "Esperando datos...",
        ha="center", fontsize=8, color="#58A6FF", fontfamily="monospace"
    )

    # ── Candidato detectado automáticamente ──────────────────
    candidate_txt = fig.text(
        0.5, 0.908,
        "",
        ha="center", fontsize=10, fontweight="bold",
        color="#3FB950", fontfamily="monospace"
    )

    # ── Update ────────────────────────────────────────────────
    frame_n = [0]

    def update(frame):
        frame_n[0] += 1

        with data_lock:
            snap_buf = {n: list(buffers[n]) for n in PIN_NAMES}
            snap_sta = {n: dict(stats[n])   for n in PIN_NAMES}
            n_total  = total_lines

        # Varianza de cada pin — el sensor activo tendrá varianza alta
        variances = {n: snap_sta[n]["var"] for n in PIN_NAMES}
        max_var_pin = max(variances, key=variances.get)
        max_var_val = variances[max_var_pin]

        for idx, name in enumerate(PIN_NAMES):
            lne = lines[idx]
            ax, stat_txt, act_txt = axes[idx]

            data = snap_buf[name]
            lne.set_ydata(data)

            s = snap_sta[name]
            rng = s["max"] - s["min"]
            stat_txt.set_text(
                f"min:{s['min']:4d}  max:{s['max']:4d}  "
                f"rng:{rng:4d}  var:{s['var']:6.0f}"
            )

            # Indicador de actividad
            color = PIN_COLORS[idx]
            if variances[name] > 500 and name == max_var_pin:
                act_txt.set_color("#00FF88")   # verde brillante = activo
                act_txt.set_text("◉ ACTIVO")
            elif variances[name] > 200:
                act_txt.set_color("#FFD93D")   # amarillo = algo de señal
                act_txt.set_text("● señal")
            else:
                act_txt.set_color("#333333")   # gris = plano
                act_txt.set_text("○ plano")

        # Candidato automático
        if max_var_val > 500:
            candidate_txt.set_text(
                f"▶  PIN CON MÁS ACTIVIDAD: {max_var_pin}  "
                f"(var={max_var_val:.0f})  ←  probablemente es el Airflow"
            )
            candidate_txt.set_color("#3FB950")
        elif max_var_val > 100:
            candidate_txt.set_text(
                f"▶  Señal moderada en {max_var_pin} (var={max_var_val:.0f})"
                f"  — mueve el sensor para confirmar"
            )
            candidate_txt.set_color("#F0A500")
        else:
            candidate_txt.set_text(
                "▶  Todos los pines están planos — "
                "verifica la conexión del sensor"
            )
            candidate_txt.set_color("#E05050")

        conn = "🟢 ARDUINO" if connected else "🟡 SIMULACIÓN"
        status_txt.set_text(
            f"{conn}  |  Muestras: {n_total:,}  |  "
            f"Puerto: {PORT}  |  Baud: {BAUD}"
        )

        return lines

    ani = FuncAnimation(
        fig, update,
        interval=120,
        blit=False,
        cache_frame_data=False
    )

    plt.show()
    return ani


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  DIAGNÓSTICO DE PINES — MySignals HW v2")
    print("=" * 60)
    print(f"  Puerto : {PORT}")
    print(f"  CSV    : {CSV_PATH}")
    print()
    print("  INSTRUCCIONES:")
    print("  1. El sketch 'diagnostico_pines.ino' debe estar cargado")
    print("  2. Conecta el sensor Airflow a la placa")
    print("  3. Observa cuál pin se ilumina en VERDE (ACTIVO)")
    print("  4. Aprieta/suelta la banda para confirmar")
    print("  5. Anota el pin — eso va en el código principal")
    print("=" * 60)

    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()

    try:
        ani = build_dashboard()
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        print(f"\n[Listo] CSV guardado en: {CSV_PATH}")
