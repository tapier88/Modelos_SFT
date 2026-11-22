# M09 — restauración aislada y reconciliación de artefactos

Ejecución real: 21-11-2026, aproximadamente 20:08–20:11 UTC. Dictamen: BLOCKED por alcance pendiente de deduplicación.

Dos espacios de nombres nuevos restauraron Kronos y tokenizador desde revisiones fijadas, tres cabezas Huber y diez paquetes IV. Se aplicaron los cinco adaptadores a copias de sus arquitecturas, comprobando claves y tensores estrictamente. Los hashes de los cinco modelos adaptados coincidieron entre las dos rondas. Se cargaron optimizadores y RNG NumPy/torch CPU IV en aislamiento, sin pasos de optimizador. Las tres cabezas reprodujeron las predicciones de desarrollo cacheadas con diferencia máxima y media cero en ambas rondas. Esta prueba no recalcula las predicciones de los pilotos IV ni publica los objetos temporales en AXXEL_SESSION.

El inventario remoto recorrió MISION06_ALINEACION y leyó 67 archivos JSON sin errores. Calculó hashes de 23 checkpoints: 22 coincidieron con los manifiestos III–IV auditados y el restante coincidió con la prueba sintética de M09. Se excluyeron directorios de datos, años reservados, credenciales y cachés antes de recorrerlos. Los checkpoints se leyeron para hashing, sin sobrescritura.

La comparación del contenido de las configuraciones distingue Fase III (LR cabeza 0,001, ajuste original) de la continuación propuesta (0,0003, inicialización desde best), y Fase IV (100 pasos, CE y cabeza auxiliar congelada) del joint propuesto (300 pasos, Huber, cabeza entrenable). Se guardaron seis huellas candidatas sin fechas ni rutas. No se certificaron experimentos fuera de MISION06_ALINEACION; por ello no se declara deduplicación global ni READY_FOR_PHASE_V.

Evidencias locales:

- `output/axxel/m09_restore_twice_20261121_retry/bootstrap_restore_twice/result.json`: SHA256 `afbe5c9a7171f549a23b401ac6bef38e0e7be625a494ddff35b2b088317dedde`.
- `output/axxel/m09_dedup_20261121/bootstrap_dedup/result.json`: SHA256 `e76653641e85a5917bcff2b6ebf10c82db44fd92acffce6f678b829d63e823a5`.
- `output/axxel/m09_dedup_20261121/proposed_fingerprints.json`: SHA256 `fa9a82e2cc11cb7e36453cc2571df122948f8c1ff080170c61f3f6787a5a5a35`.
- `output/axxel/m09_dedup_20261121/reconciliation.json`: comparación reproducible de checkpoints y alcance pendiente.

El primer intento falló antes de insertar código porque Colab tenía oculto el encabezado y el menú Archivo. Mostrarlo permitió guardar el respaldo; no fue necesario cambiar el controlador. Las celdas nuevas tienen fuentes verificadas y transacciones preservadas. No se reinició Colab, no se entrenó Fase V y no se accedió al test reservado. Los archivos nuevos de validación están en m06_staging; los invocadores temporales reutilizan el transporte existente y quedan en output, excluido de Git. No se hizo commit ni push.
