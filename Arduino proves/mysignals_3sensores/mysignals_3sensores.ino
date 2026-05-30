/*
 * ================================================================
 * MYSIGNALS HW v2 — GSR + AIRFLOW + ECG
 * Librería oficial: MySignals (Libelium)
 * ================================================================
 * PINES CONFIRMADOS (MySignals.h oficial):
 *   A0 → EMG
 *   A1 → ECG      ← este
 *   A2 → Airflow  ← este
 *   A3 → GSR      ← este
 *   A4 → Temperatura
 *   A5 → Snore
 *
 * FORMATO CSV:
 *   ts_ms, gsr_conductance, gsr_resistance, gsr_voltage,
 *          airflow_voltage, ecg_voltage
 * ================================================================
 */

#include <MySignals.h>
#include "Wire.h"
#include "SPI.h"

const unsigned long INTERVAL_MS = 20; // 50 Hz — importante para ECG
unsigned long lastSample = 0;
unsigned long startTime  = 0;

void setup()
{
  Serial.begin(115200);
  MySignals.begin();
  startTime = millis();

  Serial.println("# MYSIGNALS_GSR_AIRFLOW_ECG");
  Serial.println("# A3=GSR  A2=Airflow  A1=ECG");
  Serial.println("ts_ms,gsr_conductance,gsr_resistance,gsr_voltage,airflow_voltage,ecg_voltage");
}

void loop()
{
  unsigned long now = millis();
  if (now - lastSample < INTERVAL_MS) return;
  lastSample = now;

  // GSR (10 Hz es suficiente — submuestrear cada 5 ciclos)
  float gsr_cond = MySignals.getGSR(CONDUCTANCE);
  float gsr_res  = MySignals.getGSR(RESISTANCE);
  float gsr_volt = MySignals.getGSR(VOLTAGE);

  // Airflow
  float air_volt = MySignals.getAirflow(VOLTAGE);

  // ECG — señal cruda en voltios
  float ecg_volt = MySignals.getECG(VOLTAGE);

  Serial.print(now - startTime);
  Serial.print(",");
  Serial.print(gsr_cond,  4); Serial.print(",");
  Serial.print(gsr_res,   2); Serial.print(",");
  Serial.print(gsr_volt,  4); Serial.print(",");
  Serial.print(air_volt,  4); Serial.print(",");
  Serial.println(ecg_volt, 4);
}
