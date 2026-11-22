# M09 — deduplicación externa completada; activación BLOCKED

21 de noviembre de 2026. Última restauración: 20:42:48 UTC.

Se inspeccionaron 99 directorios y 21 configuraciones/manifiestos fuera de MISION06, dentro de AXXEL_CAPITAL. Se localizaron 68 checkpoints; todos quedaron clasificados como métodos diferentes de M06_FV_JOINT_RETURN_V1 mediante evidencia de configuración o estructura. No quedan candidatos sin clasificar en el conjunto inspeccionado. Las carpetas de datasets, test, años reservados, credenciales y tokenizadores quedaron excluidas. No se afirma cobertura de otras ubicaciones de Drive ni de los directorios excluidos.

Los protocolos M04/M05 adaptan atención cruzada y difieren de ambos brazos propuestos. Cuatro ejecuciones históricas documentan objetivo sobre tokens. En la quinta ejecución faltaba una configuración encontrada; sus cuatro checkpoints se inspeccionaron estáticamente: 127 nombres de parámetros del backbone, sin claves LoRA ni de la cabeza Huber esperada. El intento anterior con `weights_only=True` rechazó objetos NumPy. Se preservó ese error y se resolvió la clasificación con `zipfile`/`pickletools`, sin deserialización insegura, carga de modelo ni ejecución de código del checkpoint.

La reconciliación es por contenido, no exclusivamente por carpeta: basta una diferencia científica verificable para descartar equivalencia. El registro local contiene cada checkpoint, la diferencia y el hash de su evidencia. Las seis huellas candidatas de Fase V siguen propuestas, sin corridas iniciadas.

`scripts/axxel.ps1 start` finalizó RESTORED, reused=false, publicando AXXEL_SESSION. Lecturas actuales: Tesla T4; 14.754 MiB libres antes de la restauración; PyTorch 2.14.1+cu130; CUDA 13.0; Drive montado y accesible. Los checkpoints III–IV coincidieron con sus hashes. Las tres cabezas reprodujeron las predicciones cacheadas con diferencias máxima/media 0. Se reutilizaron las celdas health/restore, sin duplicarlas. Las fuentes y salidas de las demás celdas permanecieron idénticas en cada operación verificada.

## Dictamen y límite operativo

La deduplicación pendiente queda aprobada dentro del ámbito documentado. Se conservan las aprobaciones anteriores de restauración aislada, causalidad, normalización, paso cero, gradientes y reanudación sintética.

**BLOCKED para activar Fase V:** el diagnóstico actual devuelve `training_active=UNKNOWN`. La restauración comprueba que no haya procesos GPU ajenos, pero no demuestra ausencia de entrenamiento de fondo dentro del proceso del kernel. No se convierte esa ausencia de evidencia en una comprobación aprobada. La próxima acción es comprobar exclusivamente la actividad actual del kernel/procesos antes de programar entrenamiento. No hace falta repetir la deduplicación ni los experimentos I–IV.

## Evidencias

- Reconciliación de los 68 candidatos: `output/axxel/m09_external_20261121/scientific_reconciliation.json`.
- Inventario externo: `output/axxel/m09_external_20261121/bootstrap_external/result.json`, SHA256 `9e0e5214c089c4609f3189261397f50e18b0575f4330f14032199448fd0cb2d9`.
- Esquemas estáticos: `output/axxel/m09_legacy_static_20261121/bootstrap_legacy_static/result.json`, SHA256 `38519af749badc1d9c2652159e5fcb47207524b65b5b100ad2a66098824af5a4`.
- Salud actual: `output/axxel/20261121T2042284299137Z/bootstrap/result.json`, SHA256 `3633d1dc4943307c20a4d9086cbe3a159fbc052b6bc25b4a8219be62a14e77ab`.
- Restauración actual: `output/axxel/20261121T2042284299137Z/bootstrap_restore/result.json`, SHA256 `d27f8396944b2432fc300358d07b157d4408755a360b55bc8b0b2b7a0c0650d4`.
- Comprobaciones de conservación: `output/axxel/m09_external_20261121/final_evidence.json`.

Cero entrenamiento, cero acceso al test, ningún checkpoint modificado, ningún commit ni push. Los informes anteriores permanecen como evidencia histórica.
