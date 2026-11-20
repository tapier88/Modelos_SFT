# AXXEL CAPITAL — ESTADO ACTUAL KRONOS

**Última auditoría verificada:** 20 de noviembre de 2026  
**Proyecto:** AXXEL CAPITAL — Kronos SFT XAUUSD  
**Modelo:** NeoQuasar/Kronos-small + Kronos-Tokenizer-base  
**Estado operativo:** RESEARCH_ONLY — ORDERS DISABLED

## 1. Misión principal

Entrenar y validar Kronos-small para predecir los próximos cuatro cierres M15 de XAUUSD, equivalentes a un horizonte de 60 minutos.

Prioridad: mejorar la precisión predictiva frente a persistencia y benchmarks establecidos, sin data leakage.

El desarrollo de estrategias de trading es una etapa posterior.

## 2. Estado científico

| Fase | Estado |
|---|---|
| I — Diagnóstico y baselines | COMPLETADA |
| II — Inferencia Monte Carlo | COMPLETADA |
| III — Regresión Huber/Laplace | COMPLETADA |
| IV — Pilotos LoRA | COMPLETADA Y AUDITADA |
| V — LoRA + regresión conjunta | PROPUESTA, NO EJECUTADA |
| Evaluación final 2023–2024 | TEST SELLADO |

Los 37 artefactos del manifiesto de Fase IV fueron verificados mediante SHA-256.

## 3. Resultados predictivos verificados

Validación diagnóstica 2020: 950 ventanas, 3.800 predicciones y 243 sesiones NY.

| Modelo | MAE |
|---|---:|
| Persistencia | 3.560039 |
| Kronos base | 3.821930 |
| Mejor cabeza Huber | 3.552607 |
| LoRA QV + cabeza congelada | 3.558846 |
| LoRA QV greedy | 3.899300 |

La mejor cabeza Huber obtuvo una mejora puntual aproximada del 0,21 % frente a persistencia.

Los intervalos de confianza incluyen cero. No existe todavía evidencia estadística suficiente de superioridad predictiva reproducible.

Los cinco pilotos de Fase IV ya se ejecutaron y no deben repetirse sin una justificación experimental.

## 4. Arquitectura de Fase V

**Protocolo:** M06_FV_JOINT_RETURN_V1

- Brazo A: backbone Kronos congelado + cabeza Huber entrenable.
- Brazo B: backbone congelado + LoRA Q/V rank 8 + cabeza Huber entrenable.
- Semillas: 271828, 271829, 271830.
- Presupuesto: seis corridas de 300 pasos.
- Contexto: L=56.
- Horizonte: H=4, temporalidad M15.
- Entrenamiento: 2010–2019.
- Desarrollo: 2020.
- Test: 2023–2024, estrictamente reservado.

## 5. Infraestructura

**Desarrollo local:** Visual Studio Code + Codex CLI en `modelo_SFT`.

**Automatización:** Python + Playwright + Chrome CDP 9222.

**Entrenamiento:** Google Colab Pro + PyTorch + GPU, cuando esté disponible.

**Notebook maestro:**
https://colab.research.google.com/drive/1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss

**Artefactos auditados de Fase IV:**

`/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571/fase_IV/`

La conexión local de Playwright con el notebook se verificó desde Windows. La recuperación de modelos en el runtime activo y la GPU actual todavía requieren comprobación.

## 6. Próxima tarea ejecutable

**MISIÓN 07 — Recuperación y validación de Fase V**

1. Verificar el runtime actual de Colab y su GPU.
2. Recuperar los checkpoints auditados de Fases III–IV.
3. Comprobar SHA-256 y compatibilidad de arquitectura.
4. Reproducir predicciones del paso cero.
5. Verificar causalidad, gradientes e inexistencia de leakage.
6. Detectar entrenamientos equivalentes ya realizados.
7. Ejecutar las seis corridas únicamente después de superar las verificaciones.
8. Guardar resultados, curvas, checkpoints y métricas.

## 7. Reglas permanentes

- `RESEARCH_ONLY = true`
- `orders_enabled = false`
- `test_read = false`
- No repetir Fases I–IV.
- No sobrescribir checkpoints originales.
- No declarar mejora predictiva sin evidencia.
- Mantener un solo agente escritor sobre el notebook.
- Actualizar este documento después de cada tarea significativa.

## 8. Historial que debe conservarse

La evaluación M15 local del 11 de noviembre sigue siendo evidencia histórica válida para su población y configuración originales.

Sus 5.065 ventanas y métricas de 2020–2022 no son directamente comparables con las 950 ventanas diagnósticas de 2020 utilizadas en Misión 06.

No mezclar resultados de distintas poblaciones o protocolos.

**Último hito científico verificado:** Fase IV completada.  
**Siguiente hito:** preejecución y entrenamiento controlado de Fase V.