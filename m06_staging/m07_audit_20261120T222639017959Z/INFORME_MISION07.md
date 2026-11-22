# MISIÓN 07 — recuperación y diagnóstico de preejecución

Fecha: 20-11-2026. RESEARCH_ONLY=true; orders_enabled=false; test_read=false.

## Dictamen
Fases III–IV recuperadas y verificadas localmente. Fase V no iniciada: Drive no está montado en el runtime reconectado. No se aprobaron los controles que dependen de artefactos remotos y modelos vivos. No se reentrenó ninguna fase ni se leyeron datasets originales/test.

## Estado científico y cronología
- 08-11: pruebas locales de tres pasos M5/M15; no equivalen a entrenamiento completo.
- 11-11: evaluación base L64, 5.065 ventanas 2020–2022. MAE 2,2469 frente a persistencia 2,0327. Población distinta; no mezclar con M06.
- INFORME_RESULTADOS registra otro piloto Colab de tres pasos del 12-11 UTC sobre 24 ventanas; tampoco representa Fase IV.
- Auditoría 20-11: cinco pilotos Fase IV de 100 pasos, seis salidas; tres cabezas Huber de Fase III. Validación diagnóstica L56/H4: 950 ventanas, 243 sesiones NY de 2020. Fases I–II constan como completadas en la memoria; esta misión no reaudita todos sus artefactos.
- Se reprodujo la auditoría local en destino nuevo, sin ejecutar sus escrituras sobre la evidencia histórica.

## Integridad superada
37/37 archivos de Fase IV: SHA-256 y tamaños. Cinco adaptadores finales y cinco resume.pt: nombres, shapes, finitud, 147.456 parámetros por variante, paso100, semilla271828, hashes declarados de modelo/tokenizador; adaptador final idéntico al de reanudación. Estados Adam y RNG NumPy/torch/CUDA presentes y comprobados según el auditor histórico. Esto no acredita restauración viva del optimizador ni RNG Python.

11 archivos seleccionados de Fase III verificados, incluidas tres cabezas de 33.092 parámetros y scalers. No se dispone de una verificación nueva de todo el caché III. Compatibilidad estructural no equivale a inferencia restaurada.

Manifiesto IV SHA-256: e9e0e3754deaafb0153d11dfce2bf0abd024669d2a253f2f7768c8a6bfa7effc.
Se recalcularon seis métricas IV, tres métricas Huber y 12 intervalos IV desde predicciones persistidas. Tolerancias históricas: métricas 1e-10; CE 1e-6. Los IC históricos usan 2.000 réplicas, no las 10.000 previstas para Fase V.

## Acceso y runtime observado
CDP9222: Chrome157; Playwright conectado a una única pestaña del notebook maestro exacto. Inicialmente desconectado por inactividad; se pulsó Volver a conectar sin reiniciar ni ejecutar celdas históricas. Se respaldó el notebook mediante descarga UI antes del diagnóstico.

Una celda nueva M07_RUNTIME_READONLY_PREFLIGHT ejecutada: PID516, Tesla T4, 15.360 MiB totales y 14.913 MiB libres; Drive disponible=false, carpeta canónica disponible=false, variables con prefijos m6_/M6_=[] . El inventario de procesos no demuestra ausencia absoluta de trabajos externos. No se observó entrenamiento activo en metadatos del respaldo. La GPU no prueba que se hayan recuperado modelos.

El primer intento restringido de Playwright falló por WinError5; el acceso autorizado fuera del sandbox funcionó. Un selector de descarga requirió adaptación. La guarda de diagnóstico existente impidió repetir la celda: el respaldo posterior confirma ejecución1 y su salida. No se creó un segundo diagnóstico.

## Preflight
APROBADO local: integridad III seleccionada/IV completa, estructura de checkpoints, métricas persistidas, identidad del notebook, GPU observada, protocolo localizado.
NO APROBADO/PENDIENTE: Drive y hashes remotos actuales; modelos/tokenizador vivos; recarga estricta dos veces en namespace vacío; predicciones paso0 y tolerancia CPU/GPU justificada; invariancia futura, normalización contextual, causalidad, alineación H1–H4 y separación train/dev en la nueva ejecución; gradientes LoRA/cabeza y congelamiento efectivo; restauración RNG/optimizador; guardado remoto verificado; deduplicación por huella científica. Las pruebas históricas no sustituyen estas comprobaciones.

## Protocolo confirmado y seis corridas
M06_FV_JOINT_RETURN_V1: A=base congelada+cabeza Huber (33.092); B=base congelada+QV rank8 alpha16 en ocho bloques+cabeza (164.164). Inicialización: cabeza III correspondiente a cada semilla; LoRA nuevo B=0, sin continuar adaptadores CE IV. Semillas271828/271829/271830. Train2010–2019 (8.428 ventanas), dev2020 (950), L56/H4. 300 pasos adicionales, micro16×acumulación4; AdamW, weight_decay0, LR cabeza3e-4/LoRA1e-4, clip1; SmoothL1 beta1 sobre retornos acumulados bps escalados con scalers III train-only. Dropout LoRA0,1; base eval con autograd en ruta conjunta. Paso0 y300 para desarrollo, checkpoints cada25; no curvas de validación intermedia contrarias al protocolo.

| Brazo | Semilla | Pasos nuevos | Estado |
|---|---:|---:|---|
| A | 271828 | 0 | Bloqueada por preflight |
| A | 271829 | 0 | Bloqueada por preflight |
| A | 271830 | 0 | Bloqueada por preflight |
| B | 271828 | 0 | Bloqueada por preflight |
| B | 271829 | 0 | Bloqueada por preflight |
| B | 271830 | 0 | Bloqueada por preflight |

No se puede excluir que existan resultados remotos nuevos bajo otros nombres. No se crearon checkpoints ni gráficas Fase V, ni se implementó o validó su motor de entrenamiento/persistencia. runs_status.json es un registro de bloqueo local, no un registro remoto de deduplicación.

## Comparación histórica verificada, no resultados Fase V
| Referencia | MAE | RMSE |
|---|---:|---:|
| Persistencia | 3,560039 | 5,391019 |
| Kronos base | 3,821930 | 5,681343 |
| Huber271828 | 3,555176 | 5,401539 |
| Huber271829 | 3,553271 | 5,395150 |
| Huber271830 | 3,552607 | 5,401648 |
| QV + cabeza congelada | 3,558846 | 5,399713 |
| QV greedy | 3,899300 | 5,766090 |

Mejor Huber: deltaMAE −0,007433; IC95 histórico [−0,022115; 0,007603]. Incluye cero. No se demuestra superioridad. Los archivos evidence.json conservan sesgo, dirección y los detalles de hashes/arquitectura. Las predicciones y figuras IV históricas se preservan en audit_20261120/fase_IV; no se presentan como gráficas generadas hoy en Colab.

## Recuperación y próxima acción
Checkpoints locales: m06_staging/audit_20261120/fase_IV/checkpoints/ y fase_III/checkpoints/fase_III/7587400db64558e6/. Ruta remota canónica: /content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571/ . Restauración viva pendiente.

Montar/autorizar Drive en el runtime actual sin ejecutar la celda histórica de preparación; inspeccionar artefactos y posibles parciales, verificar hashes y recuperar III–IV de forma idempotente. Después completar paso0 y controles científicos antes de implementar/lanzar Fase V. No proponer aún evaluación2021–2022 ni abrir test. La hipótesis Joint LoRA+Huber sigue sin probar.
