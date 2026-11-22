# AXXEL CAPITAL — ESTADO ACTUAL KRONOS

Actualizado: 20-11-2026, Misión07. RESEARCH_ONLY=true; orders_enabled=false; test_read=false.

Objetivo: Kronos-small + Kronos-Tokenizer-base, XAUUSD M15, L56/H4; mejorar predicción continua frente a persistencia. No trading.

## Estado científico
I–II completadas según memoria histórica. III–IV completadas; reauditoría local M07 superada: 37 archivos IV y 11 seleccionados III, cinco adaptadores/resume y tres cabezas Huber. No reentrenar. Fase V NO INICIADA: seis corridas bloqueadas por preflight, cero pasos nuevos. Existencia de nuevos parciales remotos aún no verificada.

950 ventanas/243 sesiones NY de desarrollo2020: MAE persistencia3,560039; base3,821930; mejor Huber271830=3,552607; QV+cabeza congelada3,558846. IC95 delta mejor Huber−persistencia=[−0,022115;0,007603]: sin superioridad demostrada. Métricas recalculadas desde predicciones, no inferencia nueva.

## Runtime comprobado
CDP9222/Playwright: notebook exacto 1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss, título Kronos_SFT_verificado.ipynb. Desconectado por inactividad inicialmente; reconectado sin reinicio. Celda nueva de diagnóstico ejecutada una vez: Tesla T4, 15.360MiB totales/14.913MiB libres, PID516; Drive NO montado, carpeta canónica inaccesible, sin variables M6. No asumir persistencia de este runtime en próxima sesión.

## Evidencia y recuperación
Informe/evidencia/registro de seis corridas/manifiesto nuevos: `m06_staging/m07_audit_20261120T222639017959Z/`.
Respaldo previo: `m06_staging/m07_runtime_20261120T223151636075Z/notebook_before.ipynb`.
Respaldo con diagnóstico y salida: `m06_staging/m07_runtime_20261120T223326852891Z/notebook_before.ipynb`.
Fuentes III–IV locales: `m06_staging/audit_20261120/`.
Remota: `/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571/`.
SHA256 manifiestoIV: `e9e0e3754deaafb0153d11dfce2bf0abd024669d2a253f2f7768c8a6bfa7effc`.

## Próxima acción obligatoria
Montar/autorizar Drive, verificar artefactos actuales y deduplicar por huella; recuperar modelos/checkpoints sin entrenar III–IV. Reproducir paso0, probar recarga/causalidad/gradientes/congelamiento/separación de datos/RNG/guardado. Solo entonces Fase V según `m06_staging/audit_20261120/PROTOCOLO_FASE_V_JOINT_PROPUESTO_20261120.json`: dos brazos×tres semillas,300pasos,micro16×4,AdamW sin weight decay,LR cabeza3e-4/LoRA1e-4; evaluar dev paso0/300. Motor de Fase V y persistencia todavía no implementados/validados. Sin gráficas ni checkpoints nuevos de entrenamiento.

## Reglas e historia
Leer AGENTS.md. Un solo escritor del notebook; preservar fuentes/checkpoints; test2023–2024 sellado. No ejecutar automatización antigua que repite celdas1–8. Informes08-11 y evaluación11-11 (5.065 ventanas,L64,2020–2022) son históricos y no comparables directamente con M06. Estado anterior preservado en la carpeta M07.

## Continuación M07 — autorización Drive pendiente
20-11-2026 ~22:43 UTC: runtime GPU sigue conectado. Nueva celda M07_MOUNT_DRIVE_V1 (id SOeTIq7YHoEB) enviada una sola vez, pendiente de autorización OAuth de Google abierta en Chrome. No reenviar mientras siga activa. No hay entrenamiento nuevo. Automatización automation/m07_colab_session.py respalda y contrasta código antes de ejecutar; la sangría añadida por Colab fue revisada antes de autorizar esa celda. Evidencia UI: output/playwright/m07_20261120T224252215911Z/snapshot.txt. Inventario remoto de solo lectura preparado en m06_staging/m07_remote_inventory_cell.py; NO ejecutado. Próxima acción: comprobar que termina el montaje, descargar notebook para capturar M07_MOUNT_RESULT y luego auditar artefactos remotos.

## Reconciliación actual — 20-11-2026 ~22:55 UTC
Source mismatch explicado: cuatro espacios añadidos al print final por el editor. Celda30 diagnóstico intacta; montaje SOeTIq7YHoEB único y completado, Drive/carpeta canónica disponibles. No reejecutar montaje. Controlador corregido, comparación conservada y 5/5 pruebas. Informe/hashes/diff: m06_staging/m07_reconciliation_20261120T224952Z/. Verificación remota III70/70 e IV37/37 aprobada; modelos, tres cabezas y cinco adaptadores/optimizadores restaurados sin entrenar. Evidencia remota nueva: MISION06_ALINEACION/m07_recovery/20261120T225342149867Z/. Preflight datos/paso0 enviado; recoger resultado antes de cualquier otra ejecución. Fase V expresamente prohibida por instrucción actual.
