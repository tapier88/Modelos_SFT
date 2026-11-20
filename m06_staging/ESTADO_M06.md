# AXXEL CAPITAL — M06: revisión inicial y bloqueo de ejecución

Fecha: 14 de noviembre de 2026, Bogotá.

**Estado: MISIÓN INCOMPLETA. No se ha ejecutado ningún experimento M06 en Colab.**
RESEARCH ONLY — ORDERS DISABLED.

## Evidencia revisada

Se leyó el cuaderno principal `Kronos_SFT_verificado.ipynb`, ID
`1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss`, con 40 celdas históricas.
Se inspeccionaron el código y las salidas de cierre de M05, y los archivos
`vendor/model/kronos.py` y `vendor/model/module.py` incluidos en el paquete del
propio cuaderno. Esta es una revisión estática; no certifica qué objetos están
actualmente en memoria en Colab.

El archivo local `kronos_pipeline.py` coincide con el hash exigido por M05:
`1424fb451942c7e1e1823808405f428283f9b1aacf7b34d08df7ca9edb503016`.
Commit de código registrado: `67b630e67f6a18c9e9be918d9b4337c960db1e9a`.

## Hallazgos verificables por código y registros existentes

1. **Las referencias históricas de precio abarcan 2020–2022.** Los valores
   3.193094, 3.430783 y 3.660541 corresponden a 2.970 ventanas y 754 sesiones NY
   en esos tres años. No constituyen las referencias de desarrollo 2020 de M06.
   El monitor de CE de M05 sí empleó 128 ventanas de 2020.
2. **La recuperación de M05 no es compatible con el alcance de M06.** Carga
   13 fuentes 2010–2022 y recupera resultados combinados. M06 debe construir
   un índice propio con una lista cerrada de archivos 2010–2020, sin ejecutar
   las celdas históricas de recuperación.
3. **Hay clipping además de cuantización.** `normalize` calcula media y
   desviación solo sobre el contexto, agrega epsilon 1e-5 y recorta a [-5,5].
   Su contribución al error de precios debe medirse por separado. No se ha
   demostrado que explique el deterioro.
4. **s1 y s2 son particiones del código latente aprendido.** El encoder proyecta
   a `s1_bits+s2_bits`, BSQ normaliza el vector y cuantiza sus signos. `half=True`
   divide los índices en dos subcódigos. La decodificación recupera signos con
   escala `1/sqrt(codebook_dim)` y usa una proyección y un decoder Transformer.
   Existe además una ruta de reconstrucción que usa la primera partición.
   El código no demuestra una semántica financiera de tendencia frente a ruido.
5. **La corrección causal forma parte del comparador.** `kp.load_models`
   reemplaza la atención cruzada por `causal_cross`, causal en train y eval.
   El rollout condiciona s2 con la secuencia s1 desplazada y el nuevo s1.
   Reemplazarlo por el predictor oficial sin conservar estas reglas alteraría
   simultáneamente la implementación y el método de decodificación.
6. **Hay un punto de extracción para la cabeza continua.** `decode_s1`
   devuelve logits y representación posterior a RMSNorm. El último estado
   obtenido exclusivamente de los 56 tokens de contexto es un candidato;
   falta ejecutar las pruebas de invariancia al futuro antes de utilizarlo.

## Preparación incorporada al cuaderno

Se agregan al final una sección M06, una celda de diagnóstico de hardware,
versiones y procedencia, una celda de carga restringida e índice temporal, y
esta revisión. El diagnóstico calcula el hash del checkpoint elegido en M05:

`MISION05_LORA/20261112T130149Z_80e373/continuacion_20261113T203705Z_d2f8d8/checkpoints/seed_271828/step_1000.pt`

Cada ejecución de la nueva celda inicial crea una carpeta única bajo
`/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/`.
Las nuevas celdas no ejecutan entrenamientos ni cargan el estado de las
celdas anteriores. Su compilación local no equivale a una prueba en Colab.

## Diseño pendiente de ejecutar

- **Fase I:** medir clipping, reconstrucción con tokens verdaderos y bypass
  continuo del cuantizador; este último es una intervención fuera del soporte
  discreto habitual del decoder, no un predictor disponible en producción.
  Comparar las cuatro combinaciones de subcódigos futuros reales/predichos
  manteniendo fijo el contexto. Son intervenciones de diagnóstico con oráculo,
  nunca candidatos predictivos ni fuentes de representaciones de la cabeza.
  Calcular efectos e interacción en MSE y diferencias en MAE. No dividir errores
  no aditivos en porcentajes independientes. Si se usa Shapley, declarar la
  regla de reparto y permitir contribuciones negativas; no identificarlo como
  porcentaje causal único del deterioro económico.
- **Fase II:** comparar greedy y cinco configuraciones fijadas de muestreo:
  temperaturas 0.6/0.8/1.0 sin truncamiento; T=0.8 con top-k=50; T=0.8 con
  top-p=0.9. Generar 32 trayectorias y derivar el resultado de 16 de las primeras
  16 trayectorias; media y mediana en precios. Dos modelos, mismas ventanas,
  seeds 271828/271829/271830 y tiempos medidos con sincronización de GPU.
  Confirmar compatibilidad en train antes de medir desarrollo y registrar
  cualquier cambio al presupuesto antes de consultar sus métricas.
- **Fase III:** congelar Kronos y tokenizador. Extraer solo representaciones de
  contexto. Comparar cabeza de retornos con Huber y Laplace NLL, y regresión
  ridge sobre retornos históricos como referencia sencilla. Escaladores y
  distribución objetivo se estiman exclusivamente en train. Fijar un presupuesto
  piloto y reglas de checkpoint antes de observar el MAE 2020. Las métricas de
  magnitud e incertidumbre requieren denominadores y calibración explícitos.
- **Fase IV:** condicionada al análisis ejecutado de I–III. Medir primero
  dimensiones reales y presupuestos de parámetros para Q/V, FFN y combinación.
  Separar ubicación de LoRA, pesos de CE y pérdida auxiliar en experimentos
  distintos. No entrenar variantes por defecto sin una hipótesis contrastable.
- **Fases V–VI:** predicciones pareadas, MAE/RMSE por horizonte y hora NY,
  histogramas y sesgo, CE/precio, trayectorias y curvas de cabeza. Ejemplos
  fijados por calendario. Bootstrap pareado por bloques de cinco sesiones NY,
  2.000 réplicas, además de dispersión entre semillas. Todo 2020 sigue siendo
  desarrollo utilizado para selección; sus intervalos no constituyen una
  evaluación final independiente ni corrigen por sí solos la búsqueda múltiple.

La dirección de persistencia queda como N/A; en otros candidatos se informan
denominadores, empates y cobertura. Si un candidato supera ambas referencias,
solo se redactará la propuesta para 2021–2022; no se ejecutará automáticamente.
Esos años ya fueron examinados en M05 y no deben describirse como un holdout
jamás observado. El test 2023–2024 permanece reservado.

## Pendientes y limitación operativa

La conexión a Drive permite revisar y actualizar el cuaderno. Esta sesión no
dispone de la herramienta de control del navegador que permite ejecutar sus
celdas ni consultar su runtime. No se ha establecido si Colab está conectado,
si tiene una T4, ni si conserva los modelos en memoria. No se ha intentado
sustituir esa ejecución por resultados de otro entorno.

| Entregable | Estado real |
|---|---|
| Revisión estática del código y cierre M05 | Realizada |
| Diagnóstico de runtime y carga restringida | Preparados; no ejecutados en Colab |
| Auditoría cuantitativa BSQ | Pendiente |
| Comparación de inferencia | Pendiente |
| Cabeza continua y pilotos LoRA | Pendientes |
| Métricas y gráficas predictivas M06 | No generadas |
| Informe científico final y recomendación M07 | Pendientes de evidencia |

Para continuar, habilitar el control de la pestaña del cuaderno principal.
Ejecutar primero **M06 · 0**, revisar el diagnóstico y después **M06 · 1**.
No usar «Ejecutar todas»: cargaría las secciones históricas con otro alcance.
Una hipótesis no queda descartada por esta revisión estática; no hay conclusiones
predictivas ni económicas nuevas.
