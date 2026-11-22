# Misión 07 — Source mismatch reconciliado

20 de noviembre de 2026. Investigación; órdenes deshabilitadas; test cerrado; Fase V prohibida en esta ejecución.

## Causa exacta y estado del notebook

Falló la segunda parte de `len(matches)==1 and source.strip()==code.strip()`: había una coincidencia, pero las cadenas eran distintas. El editor agregó cuatro espacios al último `print`, dentro del bloque `if`. No fue un error de identidad del notebook, ni un fallo de la celda 30, ni una inserción parcial. El control evitó ejecutar automáticamente ese código divergente.

La diferencia cambia el comportamiento si Drive ya está montado: el original imprime el estado siempre; la variante insertada solo lo imprime después de entrar al `if`. Por eso no se declara equivalencia semántica general. Se conservaron el archivo original, `m07_mount_drive_reviewed.py`, la celda ejecutada y el diff explícito `mount_source.diff`.

El notebook exacto tiene URL `https://colab.research.google.com/drive/1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss` y título `Kronos_SFT_verificado.ipynb - Colab`. En el respaldo fresco de reconciliación:

- Índice 30: `M07_RUNTIME_READONLY_PREFLIGHT`, ID `9wcbmH4SFnx3`, código sin cambios.
- Índice 31: `M07_MOUNT_DRIVE_V1`, ID `SOeTIq7YHoEB`, insertada completamente y única; código idéntico al archivo reviewed.
- `execution_count=2` es el número de ejecución del kernel, no evidencia de dos ejecuciones de esa celda. El diagnóstico anterior tuvo contador 1. En esta reconciliación no se reejecutó ninguna de ellas.
- Los IDs, orden relativo y fuentes de las celdas anteriores se conservaron. Los índices posteriores pueden cambiar al insertar diagnósticos; la identidad se mantiene por ID.

## SHA-256 (fuente UTF-8)

| Objeto | SHA-256 |
|---|---|
| m07_mount_drive.py original | 8c61f393480907a432c2cf9f9c5b059ab22b5a36d5e6d766835cb94922933b26 |
| Celda montaje y variante reviewed | e0ea94c026312b9e6acf49df065c60b53c3df4f8fef34f72f2595e0d14cc7632 |
| Celda diagnóstico índice 30 | 03eb06bfc43517b5cb3c9d0852a926987d7c785c6b0372d7b9b861e321cc45e6 |
| Respaldo fresco completo | be8b28b37d391956159680c34a6283093309a0ac71857de1bb8794fad60eed6e |

El archivo `evidence.json` contiene rutas, IDs, hashes, salida del montaje y comprobaciones completas.

## Corrección del controlador

`automation/m07_colab_session.py` conserva el rechazo `Source mismatch`. Para inserciones nuevas transmite el código como un literal Python dentro de `exec(compile(...))`, en dos líneas físicas: el editor no puede reindentar las líneas internas del literal. Guarda la fuente original y la fuente de transporte, descarga el notebook y compara el transporte antes de ejecutar. No modifica celdas existentes para ocultar diferencias.

El módulo `automation/m07_integrity.py` registra ambos hashes y un diff cuando falla la comparación. Comprueba también IDs únicos, preservación de todas las fuentes/tipos/orden de celdas previas y estado conectado/libre del runtime en las rutas de ejecución. No permite reenviar una celda con salidas o contador de ejecución.

Cinco pruebas pasan: rechazo de sangría alterada, transporte sin cambio de comportamiento, rechazo de eliminación de celda, rechazo de IDs duplicados y rechazo de runtime ocupado. El primer intento de pruebas encontró una denegación de Windows en su carpeta temporal; la repetición autorizada fuera del sandbox pasó 5/5.

Verificación real adicional: la fuente de transporte de recuperación descargada coincide exactamente con la esperada, SHA-256 `3707bee86f2e8cac886ae2fd1c2d35e68a7a61cb48199566967eb971c1cc8c87`.

## Montaje y actividad

La salida real `M07_MOUNT_RESULT` confirma `drive_available=true` y `canonical_run_available=true`, T4 con 15.360 MiB totales y 14.913 MiB libres en ese instante. Se recogió el resultado existente, sin reiniciar ni remontar.

Antes de la recuperación el kernel estaba libre, sin variables M6; `nvidia-smi` no reportó procesos de cómputo GPU. El inventario de procesos fue conservado. Esto confirma que no había un entrenamiento GPU observable en ese runtime; no certifica actividad en otros runtimes o máquinas. Durante esta misión solo se enviaron celdas nuevas de auditoría/recuperación, sin pasos de optimizador.

## Recuperación posterior

Se comprobaron remotamente **70/70 archivos de Fase III y 37/37 de Fase IV**, tamaños y SHA-256, incluidos los cachés III que no estaban completos en la copia local anterior. Manifiesto III: `731d1c5412d4836e16961ff2e3747e6fd4da124db81eaf20265a638b6c90d7a1`. Manifiesto IV: `e9e0e3754deaafb0153d11dfce2bf0abd024669d2a253f2f7768c8a6bfa7effc`.

Se restauraron Kronos-small y tokenizer con revisiones fijadas y hashes vivos idénticos a los auditados. Runtime: PyTorch 2.14.1+cu130, NumPy 2.4.6, Tesla T4. No se instalaron dependencias. Se descargaron los pesos fijados necesarios en `/content/m07_pinned_weights`.

Las tres cabezas Huber pasaron carga estricta y doble restauración de sus pesos. Los cinco adaptadores IV pasaron restauración de arquitectura y estado de optimizador, con RNG NumPy/torch reconstruido en objetos aislados. No se realizó ningún paso de optimizador. Presencia de RNG CUDA no equivale a demostrar reanudación bit a bit entre CPU histórica y GPU actual.

Evidencia nueva remota: `/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/m07_recovery/20261120T225342149867Z/`. No se sobrescribieron experimentos ni checkpoints. El inventario encontró 12 metadatos candidatos en MISION06_ALINEACION; su enumeración no constituye aún deduplicación científica completa.

El preflight de datos/paso cero se registra por separado al finalizar. No se autoriza entrenamiento por la mera recuperación correcta.

## Cierre del preflight ejecutado

Las 950 ventanas de 2020 fueron inferidas de nuevo en GPU. Se reprodujeron los scalers usando únicamente los cachés train2010–2019; se verificaron hashes fuente, índices, continuidad, separación train/dev y 243 sesiones NY. Se aprobaron las pruebas causales train/eval, invariancia a futuros y congelamiento de base/tokenizer.

| Semilla | MAE paso0 | Diferencia máxima de precio | Diferencia media absoluta |
|---|---:|---:|---:|
| 271828 | 3.555175810613 | 9.92095920083e-07 | 1.19105950788e-07 |
| 271829 | 3.553271414308 | 6.3738116296e-07 | 9.37488955874e-08 |
| 271830 | 3.552606559062 | 8.58803332449e-07 | 8.58990103169e-08 |

Tolerancias fijadas antes de inferir: máximo0,001 y media1e-5 unidades de precio; los errores observados son menores que1e-6 máximo. Se comprobaron paso0 neutral LoRA y gradientes finitos/no nulos de cabeza y LoRA_B, tanto en eval como con dropout LoRA activo. No hubo actualizaciones del optimizador. La prueba de aprendizaje posterior de LoRA_A sigue pendiente y no se declara aprobada.

Predicciones por ventana de las tres semillas guardadas como NPZ y releídas, junto al informe y manifiesto final, en `/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/m07_recovery/20261120T225342149867Z`. Se verificaron 10 archivos; SHA-256 de final_checksums.json: `f3ab5334362062805a61faa69a62413e7edb01462c9c487009853231c3da4cc0`.

Los códigos y salidas de las 57 celdas previas a esta reconciliación permanecen iguales en el respaldo final. No se reinició el runtime, no se reejecutaron diagnóstico/montaje, no se leyeron años2023–2024 y no se inició Fase V.

Pendiente antes de cualquier entrenamiento futuro: deduplicación científica completa, restauración integral repetida en namespaces nuevos, equivalencia de reanudación de RNG/optimizador, comprobación posterior de LoRA_A e implementación/pruebas transaccionales de Fase V. El preflight aprobado se limita a las comprobaciones efectivamente ejecutadas. No se declara preparación completa para entrenar.
