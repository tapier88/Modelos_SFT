# AXXEL CAPITAL — Reconciliación real y propuesta de Fase V

Fecha de auditoría: 20 de noviembre de 2026 (Bogotá). Estado: investigación; órdenes deshabilitadas.

## Dictamen

**La Fase IV está completada y sus artefactos son íntegros. No debe repetirse.**
Se verificaron los 37 archivos de su manifiesto SHA-256 y tamaños, incluidos cinco adaptadores
finales, cinco estados de reanudación, cinco cachés de evaluación, métricas, configuración y PNG.
Se recalcularon seis filas de métricas y 12 intervalos pareados desde predicciones existentes.
No hubo entrenamiento, nueva inferencia ni restauración del Transformer durante esta auditoría.

**Fase V conjunta: propuesta, no ejecutada.** No se encontró un experimento equivalente terminado
en el directorio de ejecución auditado ni una celda de Fase V en las 55 celdas del maestro.
La búsqueda exacta de carpetas fase_V/FASE_V tampoco devolvió resultados. Esto no demuestra ausencia
de experimentos fuera del alcance accesible o bajo nombres distintos: antes de entrenar deberá
compararse la huella científica, no solo el nombre o la ruta.

## 1. Ubicación reconciliada

Ruta solicitada: `/content/drive/MyDrive/kronos_sft/fase_IV/`.
No se localizó la carpeta literal `kronos_sft` en Drive. No se verificó que sea un alias o enlace
en un runtime. No se creó ni movió ninguna carpeta para hacer coincidir el nombre.

Ruta real, corroborada por la cadena de padres de Drive y el código del notebook:

`/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571/fase_IV/`

[Abrir Fase IV](https://drive.google.com/drive/folders/1BejHUVVpZvjsrcUXy7zqJbYeaaQa-7yt).
Padres: `AXXEL_CAPITAL / MISION06_ALINEACION / 20261114T222455Z_32d5e571`.
Esta ejecución es la fuente de verdad experimental. La carpeta nueva creada por la celda histórica
de hoy no sustituye a esta ejecución.

## 2. Integridad y arquitectura

`phase_status.json`: complete=true, pilots=5, candidate_outputs=6, steps_each=100,
original_base_tokenizer_M05_unchanged=true, test_read=false, orders_enabled=false.
Estos indicadores se contrastaron con archivos reales; no se aceptaron por sí solos.

SHA-256 del manifiesto: `e9e0e3754deaafb0153d11dfce2bf0abd024669d2a253f2f7768c8a6bfa7effc`.
Todos sus 37 archivos coinciden; no se detectaron archivos extra en la enumeración recursiva de Fase IV,
salvo el propio `checksums.json`, que no puede contener su hash de forma autorreferente.

| model | sha256 | size | tensor_count | parameters |
|---|---|---|---|---|
| Pilot_QV_CE_s271828 | 8bcfcd5fafbf84650293ac45b215d67fe1f488483e13382309a65b3073c7e74b | 602295 | 36 | 147456 |
| Pilot_FF_CE_s271828 | 7ba538edb54b4b6f2aefeed63519ff59ced4addd98c618567ffaf6f4331585d5 | 605587 | 48 | 147456 |
| Pilot_QV_FF_CE_s271828 | d9fda1a81aebdedd7680f187168b76314f95031cd4e8eab05c1b950b742e8d04 | 617639 | 84 | 147456 |
| Pilot_QV_CE_s2x2_s271828 | 8e85655e370f7790b53dd7812cc0cbffa7ec0312060cfed9b0191f495d0bf382 | 602295 | 36 | 147456 |
| Pilot_QV_CE_aux_s271828 | d6f1c4d620b203ae62ab6e58ef358016ea428b87d39c30b6c26444d6702cdff2 | 602295 | 36 | 147456 |

Deserialización local de tensores con `torch.load(weights_only=True, map_location='cpu')`,
solo después de verificar SHA y tamaño. Se comprobaron nombres, shapes, FP32, finitud, paso 100,
semilla 271828, rango y presupuesto; los tensores finales son idénticos a los de resume.pt.

- Q/V: 18 proyecciones de 512→512, rank 8, 36 tensores A/B.
- FFN: 24 matrices SwiGLU 512↔1024, rank 4, 48 tensores.
- Combinado: Q/V rank 4 y FFN rank 2, 84 tensores.
- Las cinco variantes tienen 147.456 parámetros LoRA. No son cinco entrenamientos multisemilla:
  los cinco se ejecutaron únicamente con 271828.

Hash de base declarado por todos los checkpoints y coincidente con la auditoría del modelo:
`a3c79ed2ca1f5b35c921fce7cf673db515dd7593e5552c643a977f8dc96ae6a9`.
Hash de tokenizer: `e7885d9d7874cde53113b1cc43cbdf8f05941fb8d95bce0c38d03183665e4740`.
Código ejecutado de pilotos verificado: `3ee88503167cbbef2171189f59efea31b3bb7105c27c8293caa837366aca55a5`.
La implementación conserva la corrección causal de cross-attention en entrenamiento y evaluación.

También se verificaron 11 artefactos seleccionados de Fase III, incluidos las
tres cabezas Huber, scalers, predicciones y métricas. Cada cabeza tiene 33.092 parámetros y shapes
512→64→4. La lista exacta está en audit_evidence.json. **No se auditó de nuevo todo el caché de
representaciones de Fase III ni se recalculó el hash de los pesos base cargados en un runtime vivo.**
La compatibilidad estructural está comprobada; la recarga estricta en Colab y la equivalencia
de predicción al paso cero siguen siendo condiciones de preejecución.

## 3. Resultados reconciliados

Misma validación: 950 ventanas, cuatro horizontes, 3.800 predicciones y 243 sesiones NY de 2020.
Se verificaron fechas, unicidad ventana/horizonte, coincidencia de truth con las referencias,
igualdad NPZ/Parquet, MAE/RMSE/sesgo/dirección y CE de los cachés. Tolerancia de métricas: 1e-10;
CE FP32: 1e-6. Los 12 IC reproducen bootstrap circular de bloques de cinco sesiones, 2.000 réplicas,
semilla 271828. Son cálculos sobre resultados existentes, no nuevos experimentos.

| model | MAE | RMSE |
|---|---|---|
| Persistencia | 3.560039 | 5.391019 |
| Kronos_base | 3.821930 | 5.681343 |
| Kronos_LoRA_M05 | 4.051262 | 5.919283 |
| Head_Huber_s271828 | 3.555176 | 5.401539 |
| Head_Huber_s271829 | 3.553271 | 5.395150 |
| Head_Huber_s271830 | 3.552607 | 5.401648 |
| Pilot_QV_CE_s271828 | 3.899300 | 5.766090 |
| Pilot_FF_CE_s271828 | 3.903771 | 5.748853 |
| Pilot_QV_FF_CE_s271828 | 3.911072 | 5.772227 |
| Pilot_QV_CE_s2x2_s271828 | 3.899524 | 5.774412 |
| Pilot_QV_CE_aux_s271828 | 3.905302 | 5.775275 |
| Pilot_QV_CE_aux_s271828_continuous | 3.558846 | 5.399713 |

Los cinco greedy de Fase IV empeoran frente a Kronos base y persistencia; sus IC pareados
frente a ambas referencias quedan por encima de cero. Cambiar ubicación o ponderación de CE
no produjo mejora de precio en estos pilotos de 100 pasos; no descarta toda configuración LoRA.

La sexta fila no es un sexto checkpoint: usa el adaptador QV_CE_aux y una cabeza Huber congelada.
Su delta MAE frente a persistencia es -0.001193753, IC95
[-0.020720961, 0.018596877]. No demuestra mejora reproducible.
Además, queda por detrás de las tres cabezas Huber originales de Fase III.
La gradiente auxiliar medida fue 0,001377902: había ruta diferenciable a LoRA, pero la cabeza
no se optimizaba. Por tanto, **la optimización conjunta LoRA+cabeza aún no fue probada** en esta ejecución.

Las cabezas Huber de Fase III reducen el MAE puntual de persistencia entre aproximadamente 0,14%
y 0,21%; sus tres IC incluyen cero. Mejor valor observado: semilla 271830, paso 150, MAE 3,552607.
No se propone promoción ni ventaja económica basándose en estos resultados.

## 4. Notebook maestro y estado operativo actual

[Notebook maestro](https://colab.research.google.com/drive/1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss).
55 celdas. Celda de Fase III: `YVYd1iJAsnsf`; de Fase IV: `qtn1qIqmtknc`.
La salida de Fase IV contiene FASE_IV_COMPLETA, sus cinco pasos 100 y sus métricas; coincide con Drive.
Última modificación de la copia leída: 2026-11-20 16:29:10 UTC (11:29:10 Bogotá).
SHA-256 de los bytes descargados: `0148815b8eadf87a5f9a5e41bff54cf5cfd6760c91a33efceafea02c078ddedb`.

Contra el respaldo local anterior: **40/40 IDs y códigos históricos coinciden; 39/40 salidas
y metadatos coinciden**. La diferencia está en la celda índice 4, ID `a401cc0e76`, de montaje y
creación de carpeta. La salida antigua apuntaba a `20261114T123309Z_907b42`; la actual apunta
a `20261120T162855Z_87cae5`. Hay evidencia de reejecución, no de pérdida de resultados Fase IV.
Se preservan ambas versiones; no se restaura automáticamente una salida encima de la otra.

El indicador histórico M06_ENVIRONMENT_STABLE=TRUE corresponde a las pruebas ejecutadas el 15 de
noviembre, PID 541. **No demuestra que el runtime del 20 de noviembre esté estable o recuperado.**
El endpoint CDP9222 no respondió en esta auditoría; no se pudo inspeccionar ejecución activa,
variables, GPU ni cambios de la versión aún no guardada del cliente. La comparación actual usa
la versión persistida de Drive. No se reinició el runtime ni se modificó el notebook.

La celda de recuperación v2 del maestro restaura I–II. Existe código de extensión para III–IV,
pero no está conectado al loader de recuperación en esta copia de 55 celdas. No declarar recuperadas
III–IV en RAM solo porque sus archivos existan. Antes de Fase V: integrar y probar la recuperación
desde archivos verificados, dos veces, conservando modelos/cachés y sin repetir entrenamiento.

## 5. Propuesta experimental Fase V: LoRA + regresión continua

Identificador: `M06_FV_JOINT_RETURN_V1`. Semillas 271828, 271829, 271830.
La denominación corresponde al nuevo objetivo solicitado; las visualizaciones/síntesis pendientes
de la misión original siguen siendo entregables separados y no se consideran ejecutadas.

### Hipótesis y comparación controlada

Hipótesis: adaptar las representaciones causales junto con la cabeza de retornos mejora MAE
respecto a persistencia y a las mejores cabezas congeladas. Comparación emparejada por semilla:

| Brazo | Inicialización | Parámetros entrenables | Pasos nuevos |
|---|---|---:|---:|
| A: cabeza sola | Cabeza Huber de Fase III de esa semilla; Kronos base congelado | 33.092 | 300 |
| B: cabeza + Q/V LoRA | Misma cabeza; LoRA nuevo neutral en ocho bloques de autoatención | 164.164 | 300 |

Los 131.072 parámetros LoRA corresponden a 16 proyecciones Q/V rank8, alpha16.
Se excluye la cross-attention de dependencia s2: no participa en `decode_s1` → cabeza continua
y tendría gradientes ausentes en este objetivo. Es una diferencia justificada frente a los
147.456 parámetros de los pilotos tokenizados. Base y tokenizer permanecen congelados e intactos.
No continuar desde los adaptadores CE de Fase IV: su degradación y cambio de representación
añadirían otro factor. Los cinco checkpoints se conservan como referencias auditadas.

### Objetivo y presupuesto fijados

Retornos acumulados H1–H4 en bps, extraídos del último hidden512 del contexto56.
Se mantienen scalers de features y retornos de Fase III ajustados solo con train.
Pérdida SmoothL1/Huber beta1 sobre retornos escalados, igual ponderación de horizontes,
sin CE en la primera comparación. Así cambia la capacidad de adaptación y se mantiene el objetivo
de Fase III. Una pérdida Huber directamente en precio sería una ablación posterior registrada aparte.

AdamW sin weight decay; LR cabeza3e-4 en ambos brazos, LR LoRA1e-4; clip de gradiente1;
lote efectivo64, micro16×4,300 pasos adicionales fijos. Dos brazos×tres semillas: seis corridas,
máximo1.800 pasos. Sin ampliar presupuesto ni buscar LR tras mirar2020. Preferir T4 si disponible.
El control de cabeza sola distingue adaptación útil de simplemente entrenar más.

El tokenizer opera no_grad; **la ruta de hidden de Kronos en el brazo conjunto debe conservar
autograd**, aunque sus pesos base estén congelados. No reutilizar `m6_features` con decorador
no_grad ni caches de hidden base para entrenar LoRA. Se pueden reutilizar tokens estrictamente
contextuales si su procedencia y hash se verifican. Recomputar hidden con adaptadores en cada paso.
Base en eval, dropout de LoRA0,1 en entrenamiento, cabeza entrenable. Labels futuros solo entran
en la pérdida; nunca en la representación. Probar invariancia al alterar/eliminar futuros.
En B=0 inicial, A puede tener gradiente cero en el primer paso; exigir gradiente no nulo de B
y de la cabeza, y comprobar aprendizaje de A posteriormente.

### Evaluación reproducible

Train exclusivamente2010–2019 (8.428 ventanas); desarrollo exclusivamente2020 (950 ventanas).
Datos y hashes de origen congelados en el JSON del protocolo. El paso0 debe reproducir las
predicciones de Fase III antes de actualizar. Evaluar y comparar el paso300 fijo: no elegir
checkpoint por consultar2020 cada25 pasos. Guardar checkpoints cada25 para recuperación,
no para selección por error de desarrollo. Registrar también los fallos.

Primaria: MAE del cierre desnormalizado H1–H4, peso igual por ventana/horizonte.
Secundarias: RMSE, sesgo, dirección, magnitud, por horizonte/hora/sesión NY y tiempo por ventana.
Referencias fijas: persistencia, Kronos base greedy, cabeza Huber original de la misma semilla,
mejor Huber observada271830 y control A emparejado. Publicar las tres semillas completas.

Bootstrap pareado por bloques circulares de cinco sesiones NY,10.000 réplicas,semilla271828.
Mostrar IC95 exploratorios y bandas Bonferroni para nueve contrastes primarios:
3semillas×[persistencia,mejorFaseIII,controlA]. Contrastes entre semillas comparten las mismas
950 ventanas: no tratarlos como tres muestras independientes de mercado. Reportar variación
entre semillas por separado. Si faltan bloques/sesiones o cambia población, bloquear comparación.

Para proponer evaluación posterior: las tres semillas deben mejorar el MAE puntual frente a
las tres referencias primarias; bandas simultáneas debajo de cero y sin deltaMAE positivo por
horizonte/hora. Es un filtro de desarrollo exigente, no prueba de generalización.
2020 ya seleccionó las cabezas iniciales: esta contaminación por selección no desaparece al
fijar ahora un protocolo. No llamar holdout a2020 ni crear un supuesto holdout nuevo a posteriori.
Evaluación2021–2022 solo mediante propuesta posterior autorizada. Test2023–2024 sellado.

### Persistencia, deduplicación y condiciones de inicio

Huella científica por brazo/semilla sobre hashes de base/tokenizer/datos/cabeza, ubicación/rank,
pérdida,optimizador,pasos,lote. Excluir timestamps y nombres de carpeta para detectar el mismo
experimento aunque haya sido guardado con otro nombre. Registro persistente de estados
planned/running/complete/failed; un solo escritor; no sobrescribir finales existentes.
Completo compatible: cargar resultados y omitir entrenamiento. Parcial compatible: reanudar
checkpoint válido con RNG/optimizador. Hash/arquitectura ambiguos: bloquear y reconciliar.

Guardar adaptador y cabeza conjuntamente, scalers, optimizer, RNG Python/NumPy/torchCPU/CUDA,
índices de entrenamiento, paso y versiones; escritura temporal, hash/tamaño, commit y lectura.
Si archivo y manifiesto divergen por corte, conservarlos y recuperar la última transacción
confirmada. No aceptar automáticamente un archivo huérfano. Notebook: edición solo por Colab.

Antes de iniciar faltan: conexión al runtime; inspección de actividad y GPU; respaldo de la
versión visible si difiere; recarga estricta; hashes de pesos vivos; recuperaciónIII–IV idempotente;
prueba de predicciones paso0,gradientes y causalidad; guardado verificado; registro de huellas.
La existencia verificada de Fase IV evita repetirla. **Esta entrega no inicia Fase V.**

## 6. Archivos de esta entrega

- Este informe: `INFORME_RECONCILIACION_FASE_IV_Y_PROPUESTA_V_20261120.md`.
- Protocolo estructurado: `PROTOCOLO_FASE_V_JOINT_PROPUESTO_20261120.json`.
- Evidencia: `audit_evidence.json`, `hashes_verified.csv`, `metrics_recomputed.csv`,
  `COMPARACION_RECONCILIADA_20261120.csv`, `drive_file_index.json` y script de auditoría.
- Bundle ZIP: incluye evidencia y propuesta; no duplica los checkpoints de entrenamiento.
- Respaldo independiente de la versión remota del notebook, sin reemplazar el maestro.

No se leyeron datos reservados, no se enviaron órdenes, no se cambió ningún checkpoint ni artefacto
experimental previo. El último estado experimental validado es **Fase IV completa**; el estado
actual de RAM y la Fase V permanecen **pendientes de preejecución**.
