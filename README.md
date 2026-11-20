# AXXEL CAPITAL — Kronos SFT

### Investigación cuantitativa · Transformers · Fine-Tuning · Predicción de XAUUSD

Pipeline experimental de **Machine Learning aplicado a mercados financieros**, desarrollado para investigar, adaptar y evaluar modelos fundacionales de series temporales mediante **Supervised Fine-Tuning (SFT)**, **Low-Rank Adaptation (LoRA)** y regresión continua sobre representaciones latentes de Transformers.

El proyecto utiliza **NeoQuasar/Kronos-small** y **Kronos-Tokenizer-base** para estudiar la capacidad predictiva de movimientos futuros de XAUUSD (oro spot), principalmente en temporalidades M5 y M15.

La arquitectura incorpora preparación determinista de datos, control de causalidad, particiones cronológicas, trazabilidad mediante SHA-256, recuperación de checkpoints y evaluación estadística frente a modelos de referencia.

**Estado:** `RESEARCH_ONLY | ORDERS_DISABLED | TEST_SEALED`

**Última auditoría científica documentada:** 20 de noviembre de 2026.

**Última fase experimental completada:** Fase IV.

**Próxima fase:** entrenamiento conjunto LoRA + regresión continua (Fase V).

> AXXEL CAPITAL es actualmente un proyecto de investigación. No existe todavía evidencia suficiente para afirmar superioridad predictiva robusta, rentabilidad financiera ni capacidad de ejecución autónoma de operaciones reales.

---

## 1. Objetivo de investigación

El objetivo principal es adaptar Kronos-small para mejorar la precisión de los pronósticos sobre XAUUSD, evaluando si sus representaciones latentes contienen información predictiva adicional frente a modelos estadísticos y referencias simples.

La investigación sigue tres prioridades:

**I. Predicción de precios**

Entrenar y evaluar modelos capaces de pronosticar cierres futuros y retornos de XAUUSD mediante representaciones Transformer.

**II. Generalización y robustez**

Determinar si las mejoras observadas son reproducibles fuera del entrenamiento, resisten cambios de régimen y superan referencias estadísticas bajo comparaciones controladas.

**III. Aplicación futura a trading algorítmico**

Solo después de demostrar capacidad predictiva robusta, investigar estrategias con expectativa neta positiva, incorporando spread, comisiones, deslizamiento y restricciones de riesgo.

La investigación predictiva y la evaluación de estrategias de trading se mantienen como etapas independientes.

---

## 2. Arquitectura del modelo

### Modelo fundacional

| Componente | Configuración |
|---|---|
| Modelo base | `NeoQuasar/Kronos-small` |
| Tokenizador | `NeoQuasar/Kronos-Tokenizer-base` |
| Arquitectura | Transformer especializado en series temporales |
| Cuantización | Binary Spherical Quantization (BSQ) |
| Dimensión latente utilizada | 512 |
| Bloques Transformer | 8 |
| Fine-tuning | Adaptadores LoRA |
| Cabeza continua | MLP con regresión Huber/Laplace |
| Framework | PyTorch |
| Activo | XAUUSD |
| Temporalidades estudiadas | M5 / M15 |

### Referencias del modelo

**Repositorio original:**  
https://github.com/shiyu-coder/Kronos

**Pesos Kronos-small:**  
https://huggingface.co/NeoQuasar/Kronos-small

**Tokenizador:**  
https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base

Revisiones fijadas en la implementación auditada:

- Código Kronos: `67b630e67f6a18c9e9be918d9b4337c960db1e9a`
- Modelo: `901c26c1332695a2a8f243eb2f37243a37bea320`
- Tokenizador: `0e0117387f39004a9016484a186a908917e22426`

### Flujo arquitectónico

**Entrada de mercado**

OHLCV + canal amount, variables temporales y contexto histórico de longitud fija.

↓

**Kronos-Tokenizer-base**

Normalización contextual y extracción de representaciones cuantizadas mediante BSQ.

↓

**Kronos-small Transformer**

Modelado secuencial y extracción de representaciones latentes con controles de atención causal.

↓

**Adaptación experimental**

- Ruta discreta: predicción de subcódigos mediante Cross-Entropy.
- Ruta continua: regresión de retornos desde las representaciones latentes.
- Ruta conjunta propuesta: LoRA Q/V + cabeza continua entrenable.

↓

**Salida predictiva**

Pronósticos de retornos acumulados y cierres futuros para un horizonte operativo de hasta 60 minutos.

### Normalización y causalidad

La implementación auditada incluye normalización calculada exclusivamente con datos de contexto y controles de continuidad temporal.

El canal `amount` no está disponible en la fuente histórica procesada y se representa explícitamente mediante cero. No se interpreta como importe realmente observado.

El modelo mantiene congelados los pesos del tokenizador durante los experimentos descritos.

La prevención de fuga de información incluye separación temporal, pruebas de invariancia respecto al futuro y restricciones sobre el acceso a las etiquetas.

---

## 3. Infraestructura de datos

### Fuentes históricas

El proyecto dispone de dos conjuntos principales de información:

**Datos OHLC históricos**

- Periodo: 2010 hasta octubre de 2024.
- Resolución original: M1.
- Agregaciones: M5 y M15.
- Formato procesado: Apache Parquet.
- Política: conservar únicamente barras completas bajo las reglas del pipeline.

**Datos tick históricos**

- Fuente: Dukascopy.
- Periodo auditado: octubre de 2024 a octubre de 2026.
- Registros contabilizados: 48.875.569 ticks.
- Archivos BI5 válidos contabilizados: 2.575.

Los ticks corresponden a un conjunto de investigación independiente y **no forman parte del entrenamiento SFT descrito en los resultados actuales**.

### Almacenamiento

Directorio principal de datos procesados:

`/content/drive/MyDrive/AXXEL_CAPITAL/datasets/processed/kronos_ohlcv_v1/`

Los archivos se almacenan por año y temporalidad.

### Integridad

El pipeline utiliza:

- Comprobaciones SHA-256 de archivos y artefactos.
- Validación de tipos y columnas.
- Detección de duplicados y timestamps fuera de orden.
- Comprobación de valores nulos y no finitos.
- Verificación de coherencia OHLC.
- Validación de barras completas.
- Segmentación de ventanas en intervalos temporalmente continuos.
- Control de particiones cronológicas.

No se deben sobrescribir los datasets originales durante la experimentación.

---

## 4. Particiones temporales y evaluación

La estructura general de investigación distingue tres periodos:

| Periodo | Utilización |
|---|---|
| 2010–2019 | Entrenamiento |
| 2020–2022 | Desarrollo y validaciones históricas |
| 2023–2024 | Test reservado |

En los experimentos M06 y en el protocolo propuesto para Fase V, la selección y comparación se restringen específicamente a **2020**.

Para Fase V se ha establecido:

- Contexto M15: `L = 56`.
- Horizonte: `H = 4`.
- Entrenamiento: 2010–2019.
- Desarrollo: 2020.
- Test: 2023–2024, sellado.

Los años 2021–2022 ya participaron en evaluaciones históricas anteriores y no deben presentarse como un conjunto completamente desconocido.

La validación walk-forward ampliada es una capacidad de investigación pendiente de consolidar; no se declara como una evaluación final ya superada.

---

## 5. Estado experimental

La evolución del proyecto ha seguido un protocolo de investigación por fases.

| Fase | Investigación | Estado |
|---|---|---|
| I | Baselines y diagnóstico de representación | Completada |
| II | Inferencia probabilística Monte Carlo | Completada |
| III | Cabezas de regresión Huber y Laplace | Completada |
| IV | Pilotos de adaptación LoRA | Completada y auditada |
| V | Optimización conjunta LoRA + regresión continua | Propuesta |
| VI | Validación temporal extendida y calibración | Pendiente |
| VII | Evaluación final reservada y viabilidad económica | Pendiente |

### Fase III — Regresión continua

Se implementaron cabezas de regresión sobre representaciones internas del backbone congelado.

Arquitectura de la cabeza Huber:

`512 → 64 → 4`

Parámetros entrenables:

`33.092`

El objetivo fue comparar regresión continua contra métodos basados exclusivamente en generación discreta.

Se evaluaron diferentes semillas y checkpoints utilizando un conjunto diagnóstico común.

### Fase IV — Adaptadores LoRA

Se completaron cinco pilotos con presupuestos equiparables.

| Configuración | Objetivo |
|---|---|
| QV_CE | LoRA en proyecciones Q/V |
| FF_CE | LoRA en capas Feed-Forward |
| QV_FF_CE | Adaptación combinada |
| QV_CE_s2x2 | Mayor ponderación del subcódigo s2 |
| QV_CE_aux | Cross-Entropy + pérdida continua auxiliar |

Cada piloto utilizó:

- 147.456 parámetros LoRA entrenables.
- Semilla 271828.
- 100 pasos de entrenamiento.
- Presupuesto experimental fijo.

Los cinco experimentos finalizaron y sus artefactos fueron conservados.

La auditoría del 20 de noviembre verificó 37 archivos mediante SHA-256, incluidos adaptadores finales, estados de reanudación, métricas, configuraciones y gráficas.

**Estos pilotos no deben repetirse automáticamente.**

---

## 6. Resultados cuantitativos

### Evaluación diagnóstica M15 — Año 2020

Población de evaluación:

- 950 ventanas.
- Cuatro horizontes por ventana.
- 3.800 predicciones individuales.
- 243 sesiones de Nueva York.

### Comparación de error predictivo

| Modelo | MAE ↓ | RMSE ↓ |
|---|---:|---:|
| Persistencia | 3.560039 | 5.391019 |
| Kronos base | 3.821930 | 5.681343 |
| Kronos LoRA M05 | 4.051262 | 5.919283 |
| Head Huber — 271828 | 3.555176 | 5.401539 |
| Head Huber — 271829 | 3.553271 | 5.395150 |
| Head Huber — 271830 | **3.552607** | 5.401648 |
| Pilot QV_CE | 3.899300 | 5.766090 |
| Pilot FF_CE | 3.903771 | 5.748853 |
| Pilot QV_FF_CE | 3.911072 | 5.772227 |
| Pilot QV_CE_s2x2 | 3.899524 | 5.774412 |
| Pilot QV_CE_aux — greedy | 3.905302 | 5.775275 |
| Pilot QV_CE_aux — salida continua | 3.558846 | 5.399713 |

### Interpretación científica

El mejor resultado puntual corresponde a la cabeza Huber con semilla 271830:

**MAE = 3.552607**

Frente a persistencia:

**MAE = 3.560039**

La reducción puntual del error es aproximadamente **0,21 %**.

Sin embargo, los intervalos de confianza de las comparaciones relevantes incluyen cero.

Por tanto, no existe todavía evidencia estadística suficiente para afirmar que la mejora generalizará a datos no utilizados durante el desarrollo.

### Hallazgos de Fase IV

Los pilotos LoRA optimizados exclusivamente mediante Cross-Entropy discreta no mejoraron la precisión de precio frente a los modelos de referencia.

La disminución de la pérdida discreta no se tradujo necesariamente en un menor error de predicción continua.

La variante con cabeza auxiliar produjo un MAE continuo de 3.558846, sin demostrar una mejora estadísticamente concluyente frente a persistencia.

La principal hipótesis pendiente es evaluar si **entrenar conjuntamente la cabeza y los adaptadores LoRA** permite aprovechar mejor las representaciones latentes.

Estos resultados corresponden exclusivamente a la población y configuración experimental indicada. No constituyen evidencia de rentabilidad financiera.

---

## 7. Fase V — Optimización conjunta

**Protocolo:** `M06_FV_JOINT_RETURN_V1`

**Estado:** diseñado; entrenamiento pendiente de ejecución.

### Hipótesis

La adaptación conjunta del backbone mediante LoRA y una cabeza de regresión continua podría reducir el error predictivo frente a los modelos con backbone congelado.

### Diseño experimental

Se compararán dos brazos.

**Brazo A — Control**

- Kronos-small congelado.
- Cabeza Huber entrenable.
- 33.092 parámetros entrenables.
- Inicialización desde checkpoints auditados de Fase III.

**Brazo B — Optimización conjunta**

- Pesos base de Kronos-small congelados.
- Adaptadores LoRA Q/V de rango 8.
- Ocho bloques de autoatención.
- Cabeza Huber entrenable.
- 164.164 parámetros entrenables en total.

En el Brazo B se conservará la propagación de gradientes a través de las representaciones ocultas para actualizar los adaptadores LoRA.

### Hiperparámetros preestablecidos

| Parámetro | Valor |
|---|---|
| Semillas | 271828, 271829, 271830 |
| Brazos | 2 |
| Corridas previstas | 6 |
| Pasos por corrida | 300 |
| Presupuesto total | 1.800 pasos |
| Learning rate cabeza | 3e-4 |
| Learning rate LoRA | 1e-4 |
| Batch efectivo | 64 |
| Microbatch | 16 |
| Gradient clipping | 1.0 |
| Optimizador | AdamW |
| Weight decay | 0 |
| Función objetivo | SmoothL1/Huber |
| Temporalidad | M15 |
| Contexto | 56 barras |
| Horizonte | 4 barras |

### Evaluación prevista

Métrica principal:

**MAE del cierre futuro desnormalizado**, calculado sobre H1–H4.

Métricas secundarias:

- RMSE.
- Sesgo.
- Precisión direccional.
- Error por horizonte.
- Error por hora y sesión NY.
- Error de magnitud.
- Tiempos de inferencia.
- Dispersión entre semillas.

El protocolo contempla bootstrap pareado por bloques de sesiones y ajustes por comparaciones múltiples.

No se declarará superioridad predictiva basándose únicamente en mejoras puntuales del MAE.

La ejecución está condicionada a recuperar y validar previamente los modelos, checkpoints y condiciones del runtime.

---

## 8. Infraestructura de ejecución

### Desarrollo local

- Windows.
- Visual Studio Code.
- Codex CLI.
- Python.
- Playwright.
- Chrome DevTools Protocol.

Directorio principal de desarrollo:

`modelo_SFT/`

### Entorno de entrenamiento

- Google Colab Pro.
- PyTorch.
- GPU disponible según asignación del runtime.
- Google Drive para persistencia experimental.

### Notebook maestro

https://colab.research.google.com/drive/1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss

### Almacenamiento auditado de Fase IV

`/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571/fase_IV/`

La automatización local mediante CDP y Playwright permite inspeccionar y controlar el navegador.

La conexión local con el notebook fue verificada desde Windows. Esto no implica que todos los modelos estén actualmente cargados en la memoria de Colab.

El estado del runtime debe comprobarse en cada nueva ejecución.

---

## 9. Estructura del repositorio

El repositorio local organiza los componentes históricos y de investigación de la siguiente manera:

| Archivo o directorio | Descripción |
|---|---|
| `automation/` | Automatización y control de Google Colab mediante Playwright y CDP |
| `evaluacion_m15/` | Scripts de evaluación y resultados históricos M15 |
| `m06_staging/` | Preparación, snapshots y herramientas de recuperación M06 |
| `informe_tecnico/` | Informes de arquitectura, auditorías y reconciliación |
| `kronos_pipeline.py` | Preparación determinista y validación de datos |
| `INFORME_DE_EJECUCION.md` | Documentación histórica de implementación |
| `ESTADO_ACTUAL_KRONOS.md` | Memoria operativa y científica del proyecto |
| `LÉAME.md` | Documentación complementaria en español |

### Archivos de continuidad recomendados

`AGENTS.md`

Instrucciones permanentes para Codex. Debe crearse y mantenerse separado de los informes experimentales.

`ESTADO_ACTUAL_KRONOS.md`

Documento de seguimiento técnico que debe actualizarse al completar tareas significativas.

La existencia, contenido y vigencia de los archivos deben comprobarse en el directorio local antes de utilizarlos como referencia operativa.

---

## 10. Reproducibilidad y control de experimentos

Cada experimento debe registrar:

- Identificador de ejecución.
- Huella científica de configuración.
- Versión del código y revisiones del modelo.
- Semillas de aleatoriedad.
- Periodos de entrenamiento y validación.
- Arquitectura y parámetros entrenables.
- Función objetivo y optimizador.
- Métricas y predicciones.
- Checkpoints y estados del optimizador.
- Estados RNG.
- Checksums SHA-256.
- Estado de finalización o fallo.

Los checkpoints deben persistirse en almacenamiento duradero, verificando integridad antes de cada restauración.

### Prevención de duplicados

Antes de entrenar una nueva configuración:

1. Inspeccionar los experimentos ya registrados.
2. Comparar sus huellas científicas.
3. Detectar ejecuciones completas o parciales compatibles.
4. Recuperar checkpoints válidos cuando sea apropiado.
5. No sobrescribir artefactos anteriores.

### Recuperación de ejecución

Una desconexión del navegador o la pérdida del runtime no implica necesariamente que los experimentos hayan desaparecido.

El estado debe reconstruirse exclusivamente a partir de evidencias persistidas y verificadas.

La existencia de un checkpoint no demuestra por sí sola que el modelo esté cargado o que un entrenamiento haya terminado.

---

## 11. Seguridad y gobernanza

El proyecto permanece bajo las siguientes restricciones:

`RESEARCH_ONLY = true`

`orders_enabled = false`

`test_read = false`

### Reglas permanentes

- Prohibido enviar órdenes de trading reales.
- Prohibido utilizar el test reservado sin autorización explícita.
- Prohibido introducir datos futuros en las representaciones de contexto.
- Prohibido sobrescribir los modelos y checkpoints históricos.
- Prohibido repetir experimentos completados sin justificación.
- Prohibido presentar resultados de desarrollo como validación final independiente.
- Prohibido declarar rentabilidad financiera sin simulaciones realistas.
- Mantener trazabilidad de decisiones experimentales.
- Mantener un único proceso escritor sobre el notebook maestro.

Las mejoras de precisión predictiva y la rentabilidad de una estrategia constituyen objetivos diferentes, que deben evaluarse por separado.

---

## 12. Limitaciones conocidas

El desarrollo actual presenta varias limitaciones relevantes:

**Superioridad predictiva no demostrada**

Las mejoras puntuales observadas frente a persistencia son pequeñas y no han superado de forma concluyente los contrastes estadísticos establecidos.

**Posible efecto de cuantización**

Las pruebas sugieren una desalineación entre optimizar tokens discretos y minimizar errores de precio. No está demostrado que la cuantización explique por sí sola todo el deterioro observado.

**Selección sobre datos de desarrollo**

El conjunto 2020 ha participado en la selección de configuraciones. No representa una evaluación final independiente.

**Procedencia del preentrenamiento**

La partición temporal del proyecto evita ciertos tipos de fuga durante el ajuste fino, pero no certifica que los datos utilizados para evaluar el modelo abierto sean completamente ajenos a su preentrenamiento original.

**Ausencia de validación económica**

No existe todavía evidencia suficiente de expectativa neta positiva, estabilidad de Sharpe Ratio, Profit Factor o control de drawdown para una estrategia ejecutable.

**Infraestructura de experimentación**

Las asignaciones de GPU y los estados de memoria de Colab son temporales. Las ejecuciones deben diseñarse para recuperación y persistencia.

---

## 13. Hoja de ruta

### Etapa actual

**Recuperación y entrenamiento de Fase V**

- Verificar runtime y GPU.
- Recuperar modelos y cabezas de regresión.
- Comprobar hashes y equivalencia predictiva del paso cero.
- Validar causalidad y gradientes.
- Ejecutar los seis entrenamientos preestablecidos, si pasan las pruebas.
- Generar métricas, gráficas y checkpoints auditables.

### Siguiente etapa

**Validación extendida**

- Comparaciones temporales adicionales.
- Análisis por volatilidad y sesión.
- Calibración de incertidumbre.
- Evaluación de estabilidad entre semillas y regímenes.

### Etapa posterior

**Evaluación final reservada**

Tras congelar el protocolo y autorizar formalmente su utilización:

- Evaluación final sin ajuste posterior sobre el test.
- Comparación frente a persistencia y benchmarks.
- Análisis de robustez y degradación predictiva.

### Investigación futura

**Aplicación al trading algorítmico**

Estudio de señales y políticas de ejecución, sujeto a:

- Ventaja estadística demostrada.
- Costos transaccionales.
- Riesgo de mercado y ejecución.
- Walk-forward y pruebas de estrés.
- Control de drawdown.
- Monitorización de deriva y deterioro del modelo.

No se habilitarán operaciones reales por el simple hecho de completar el entrenamiento.

---

## 14. Documentación técnica

Los documentos históricos se conservan para garantizar trazabilidad.

**Informe de reconciliación de Fase IV y propuesta de Fase V**

`INFORME_RECONCILIACION_FASE_IV_Y_PROPUESTA_V_20261120.md`

**Protocolo experimental propuesto**

`PROTOCOLO_FASE_V_JOINT_PROPUESTO_20261120.json`

**Evaluación histórica M15**

`evaluacion_m15/resultados_20261111/INFORME.md`

**Memoria operativa**

`ESTADO_ACTUAL_KRONOS.md`

La ubicación efectiva de los informes y protocolos debe verificarse antes de ejecutar procedimientos dependientes de ellos.

---

## 15. Stack tecnológico

**Lenguaje y análisis**

Python · NumPy · Pandas · Apache Parquet

**Deep Learning**

PyTorch · Transformers · LoRA · SFT · Kronos-small · BSQuantizer

**Evaluación estadística**

MAE · RMSE · Bootstrap pareado · Análisis por horizonte · Comparaciones temporales

**Automatización**

Playwright · Chrome CDP · PowerShell

**Desarrollo**

Visual Studio Code · Codex CLI · Git · GitHub

**Infraestructura experimental**

Google Colab Pro · Google Drive

---

## Estado final documentado

**Fase IV:** COMPLETADA Y AUDITADA.

**Fase V:** PROTOCOLO DISEÑADO, ENTRENAMIENTO NO CONFIRMADO.

**Test 2023–2024:** RESERVADO.

**Trading:** DESHABILITADO.

**Próximo objetivo científico:** determinar si la optimización conjunta LoRA + regresión continua mejora significativamente la predicción de XAUUSD frente a persistencia y las mejores cabezas de regresión existentes.

---

**AXXEL CAPITAL — Investigación cuantitativa y desarrollo experimental de inteligencia artificial aplicada a mercados financieros.**
