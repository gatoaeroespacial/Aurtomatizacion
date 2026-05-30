/*
 * ================================================================
 * SISTEMA ADAPTATIVO DE RECOMENDACIÓN MUSICAL
 * Basado en Bioseñales: GSR, ECG y Respiración (Airflow)
 * ================================================================
 * Hardware : MySignals HW v2 (Libelium) + Arduino UNO/Mega
 * NOTA     : Lee los pines analógicos DIRECTAMENTE sin librería
 *            eHealth, que no es compatible con MySignals HW v2.
 *
 * Pines (ajustar si tu shield tiene distribución diferente):
 *   ECG     → A0   (señal cruda, convertida a voltios)
 *   AIRFLOW → A1   (0-1023 ADC, sensor de flujo de aire)
 *   GSR     → A2   (voltaje en voltios, 0-5V)
 *
 * Protocolo: Espera comando 'S' por Serial antes de enviar datos.
 *            Comando 'X' para detener.
 * Salida   : Serial 115200 baud → formato CSV
 *            timestamp_ms,gsr_V,ecg_V,airflow_raw
 * ================================================================
 */

// ── Configuración de pines ─────────────────────────────────────
#define PIN_ECG     A0
#define PIN_AIRFLOW A1
#define PIN_GSR     A2

// ── Configuración de muestreo ──────────────────────────────────
const unsigned long SAMPLE_INTERVAL_MS = 20;  // 50 Hz
unsigned long lastSample = 0;
unsigned long startTime  = 0;
bool monitoring = false;

// ── Buffer de suavizado GSR (media móvil de 5 muestras) ────────
#define GSR_BUFFER 5
float gsrBuf[GSR_BUFFER];
int   gsrIdx = 0;
bool  gsrFull = false;

float smoothedGSR() {
  float sum = 0;
  int n = gsrFull ? GSR_BUFFER : gsrIdx;
  if (n == 0) return 0.0;
  for (int i = 0; i < n; i++) sum += gsrBuf[i];
  return sum / n;
}

// ── Setup ──────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  // Configurar referencia analógica (DEFAULT = Vcc del Arduino)
  analogReference(DEFAULT);

  // Indicar qué pines se usarán (ayuda al diagnóstico)
  Serial.println("# READY — MySignals HW v2 (lectura directa ADC)");
  Serial.print("# Pines: ECG=A"); Serial.print(PIN_ECG - A0);
  Serial.print("  AIRFLOW=A");    Serial.print(PIN_AIRFLOW - A0);
  Serial.print("  GSR=A");        Serial.println(PIN_GSR - A0);
  Serial.println("# Esperando comando S para iniciar monitoreo...");
}

// ── Loop ───────────────────────────────────────────────────────
void loop() {

  // Escuchar comandos del PC
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'S' && !monitoring) {
      monitoring = true;
      startTime  = millis();
      lastSample = startTime;
      gsrIdx = 0;
      gsrFull = false;
      Serial.println("# MONITORING_START");
    } else if (cmd == 'X' && monitoring) {
      monitoring = false;
      Serial.println("# MONITORING_STOP");
    }
  }

  if (!monitoring) return;

  unsigned long now = millis();
  if (now - lastSample < SAMPLE_INTERVAL_MS) return;
  lastSample = now;

  // ── Leer sensores directamente del ADC ──────────────────────
  int rawECG     = analogRead(PIN_ECG);
  int rawAirflow = analogRead(PIN_AIRFLOW);
  int rawGSR     = analogRead(PIN_GSR);

  // Convertir ECG y GSR a voltios (referencia 5V, 10 bits)
  float ecgV = rawECG  * 5.0 / 1023.0;
  float gsrV = rawGSR  * 5.0 / 1023.0;

  // Suavizado GSR
  gsrBuf[gsrIdx] = gsrV;
  gsrIdx = (gsrIdx + 1) % GSR_BUFFER;
  if (gsrIdx == 0) gsrFull = true;
  float gsrSmooth = smoothedGSR();

  // Tiempo relativo desde inicio de sesión
  unsigned long ts = now - startTime;

  // CSV: timestamp_ms , gsr_V , ecg_V , airflow_raw
  Serial.print(ts);
  Serial.print(",");
  Serial.print(gsrSmooth, 4);
  Serial.print(",");
  Serial.print(ecgV, 6);
  Serial.print(",");
  Serial.println(rawAirflow);
}

