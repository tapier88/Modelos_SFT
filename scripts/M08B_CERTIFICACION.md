# Misión 08B — diagnóstico de solo lectura

Fecha: 20-11-2026 Bogotá / 21-11-2026 UTC. Sin entrenamiento, ejecución de celdas, reinicio, modificación de checkpoints, commit ni push.

## Auditoría y preservación

Las 98 líneas eliminadas eran la diferencia acumulada respecto a Git HEAD, no una eliminación atribuible exclusivamente a M08. Se revisó el diff completo y se conservó el archivo HEAD íntegro, el estado previo a M08B y el diff en `output/m08b_20261121T043026752194Z/`. El estado actual recupera los benchmarks y parámetros omitidos, distinguiendo auditoría de M06 y recuperación de M07.

El manifiesto local M07 se comparó archivo a archivo. Todos los artefactos de `m06_staging/` y el respaldo histórico del notebook coinciden. Difieren dos archivos editables respecto a aquel manifiesto: `ESTADO_ACTUAL_KRONOS.md` y `automation/test_m07_integrity.py` (prueba temporal ya modificada antes de M08B). No se reescribió el manifiesto. La verificación local no acredita disponibilidad remota actual de checkpoints.

## Lecturas reales

| Componente | Estado | Evidencia | Fecha UTC |
|---|---|---|---|
| Notebook | CURRENT: identidad exacta y única | URL de Colab con ID `1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss`, título Kronos_SFT_verificado.ipynb | 2026-11-21 04:31:43 |
| Runtime | CURRENT: interfaz desconectada | Control «Volver a conectar T4 Hacer clic para conectarse» | 2026-11-21 04:31:43 |
| GPU/VRAM/RAM | UNKNOWN | La etiqueta T4 desconectada no prueba asignación actual | 2026-11-21 04:31:43 |
| CUDA/PyTorch | UNKNOWN | No se interrogó el kernel ni se usaron salidas guardadas | 2026-11-21 04:31:43 |
| Drive/AXXEL_CAPITAL | UNKNOWN | Montaje y acceso no verificables con esta lectura | 2026-11-21 04:31:43 |
| Modelos cargados | UNKNOWN | Sin consulta de variables del runtime | 2026-11-21 04:31:43 |
| Celdas/procesos activos | UNKNOWN | Ningún indicador positivo observado; no equivale a ausencia de procesos | 2026-11-21 04:31:43 |
| Última recuperación verificada | HISTORICAL | M07, paso cero de tres cabezas Huber; `final_result.json` | 2026-11-20 22:59:16 |
| Persistencia/checkpoints | HISTORICAL; disponibilidad actual UNKNOWN | Manifiesto M07 `f3ab5334…da4cc0`, carpeta `m07_recovery/20261120T225342149867Z`; rutas de checkpoints originales en protocolo M06 | 2026-11-20 22:59:16 |

La primera inspección manual de la interfaz mostró conexión GPU y RAM 1,59/12,67 GB. Las lecturas posteriores cambiaron a «Volver a conectar»; no se atribuye una causa ni se conserva la primera lectura como estado vigente. No había herramienta oficial de Colab disponible en el catálogo consultado; se reutilizó Python Playwright/CDP instalado y validado.

## Aceptación

Parser nativo PowerShell aprobado. **18/18 pruebas aprobadas**: las 12 previas y seis de clasificación/seguridad. Los cuatro comandos siguen cubiertos por 16 casos simulados; `recover` y `close` no se operaron contra Colab durante esta misión y sus ramas y función de respaldo no se modificaron.

Pruebas reales: `start`, `start`, `status`. Registros:

- `output/axxel/20261121T0431167677516Z/`
- `output/axxel/20261121T0431326109664Z/`
- `output/axxel/20261121T0431435151148Z/`

Los tres muestran una sola pestaña exacta, 61 regiones de celda y el mismo SHA-256 de sus etiquetas: `0d4dc01c86921813da699955fed9868f22305522fa4f34df6f87bc2f21e0e35a`. Esto verifica ausencia de duplicados observables, no una comparación byte a byte del notebook. El diagnóstico carece de operaciones de navegación, clic, teclado o ejecución. Los registros completos, comparación de hashes y salidas de pruebas están en `output/m08b_20261121T043026752194Z/acceptance.json` y los archivos `.log` vecinos.

## Alcance certificado y pendiente

Certificados: reutilización de la pestaña exacta, lectura de conexión, separación de evidencia actual e histórica, explicación de desconocidos, repetición sin duplicados observables y protección de `recover`.

Recuperación automática no certificada ni implementada: requiere conexión autorizada, comprobación de procesos, Drive accesible, SHA-256 remotos, compatibilidad de modelos/checkpoints, restauración aislada y equivalencia de predicciones/optimizador/RNG. Esa tarea no debe habilitar ejecución científica en `start` o `status`.
