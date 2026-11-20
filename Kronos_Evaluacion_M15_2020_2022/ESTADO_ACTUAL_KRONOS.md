# Estado actual de Kronos — 2026-11-11

La tarea pendiente de evaluación M15 2020–2022 está completada en el equipo local (CPU).
La prueba corta anterior de Colab ya estaba terminada; no debe repetirse como siguiente tarea.

- 5.065 ventanas con objetivos no solapados; 20.260 cierres predichos.
- 2.179 ventanas NY en 751 días. Decisiones NY: 10:00–12:00; horizontes hasta las 13:00.
- MAE total: Kronos 2.246910; último cierre 2.032728.
- MAE NY: Kronos 2.943721; último cierre 2.661346.
- Mejora MAE total: -10.54%; NY: -10.61%.
- RUN_FULL=False; ningún entrenamiento en esta evaluación; parámetros originales intactos.
- Test 2023–2024 sin leer ni evaluar en este experimento.
- Inferencia: adaptador causal auditado con greedy; no es el predictor oficial con muestreo.

Hallazgo para el siguiente paso: las 64 velas continuas de contexto dejan fuera la apertura de las 08:30.
Un contexto de 56 velas habilita estructuralmente esa hora en 724 días de validación;
todavía no se ha medido su capacidad predictiva.

La siguiente tarea es preparar y fijar una comparación para la apertura: contexto de 56 velas,
horizonte de 4 y decisiones ancladas a 08:30, 09:30, 10:30 y 11:30 America/New_York.
Usar las mismas ventanas para persistencia, inferencia oficial y adaptador auditado.
Separar por años y días, mantener el test reservado y no activar entrenamiento completo todavía.

Evidencia: evaluacion_m15/resultados_20261111/INFORME.md, status.json,
config.json, predictions.parquet, summary.csv, independent_verification.json.
Código: evaluacion_m15/evaluar_base_m15.py.
