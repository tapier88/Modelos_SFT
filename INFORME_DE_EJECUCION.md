# Kronos SFT — implementación y verificación

Fecha de cierre técnico: 8 de noviembre de 2026 (UTC).

## Estado real

La preparación y las pruebas cortas se ejecutaron localmente con los pesos reales de Kronos-small. Se hicieron tres pasos de optimización en M5 y tres en M15, con cambios comprobados en los pesos. No se ha completado el entrenamiento ni una evaluación estadística del modelo. El conjunto test permanece sin evaluar.

La copia de trabajo en Colab contiene el procedimiento completo, ejecutable por secciones. La instalación de dependencias terminó; la ejecución está detenida en la autorización de Google Drive. Las pruebas locales no se presentan como resultados ejecutados en Colab.

- Cuaderno de trabajo: https://colab.research.google.com/drive/1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss
- Carpeta de resultados: https://drive.google.com/drive/folders/1kT5uMFHSEJrm5FcY6zwE0G5h5IrQEGHg
- Respaldo previo de la copia de trabajo: https://colab.research.google.com/drive/1d-Fpl4wquJ03Xej6UDNlky_L5wuO8DaU
- Cuaderno original, conservado: https://colab.research.google.com/drive/14qbg7mIZ6VLhC-HakE5WO9w8KCV6O9RE

## 1. Datos utilizados y conservación

Se verificaron los 30 archivos Parquet anuales de XAUUSD BID M5/M15 (15 por temporalidad) contra las huellas SHA-256 de la auditoría. Se comprobaron columnas, valores finitos, orden temporal, duplicados, alineación temporal, coherencia OHLC, volumen no negativo y cantidad completa de velas M1 por barra. Las huellas se volvieron a calcular después de la ejecución: sin cambios.

La auditoría previa reconstruyó las barras a partir de 178 CSV M1, con 5.264.297 filas. El procedimiento entregado usa los Parquet ya auditados y no vuelve a escribir el ZIP M1, los CSV o los Parquet fuente.

Fuente de Colab: `/content/drive/MyDrive/AXXEL_CAPITAL/datasets/processed/kronos_ohlcv_v1`.

Salidas nuevas: `/content/drive/MyDrive/AXXEL_CAPITAL/Kronos_SFT_verificado_20261107/ejecuciones/<fecha_UTC>_<identificador>/`.

| Temporalidad | Filas verificadas | Última barra UTC | Contexto | Horizonte |
|---|---:|---|---:|---:|
| M5 | 1.046.748 | 2024-10-18 20:55 | 128 barras | 12 barras, 60 minutos |
| M15 | 346.565 | 2024-10-18 20:45 | 64 barras | 4 barras, 60 minutos |

## 2. Organización y clasificación

Las particiones son temporales: train 2010–2019, validation 2020–2022 y test 2023–18 de octubre de 2024. No se mezclaron aleatoriamente filas entre particiones. El orden de las ventanas de entrenamiento puede barajarse una vez establecida esta separación.

Una ventana se acepta únicamente si contexto y objetivo están dentro del mismo segmento continuo y de la misma partición. Cualquier salto temporal distinto de 5/15 minutos corta el segmento. Se permite continuidad entre archivos anuales cuando realmente existe; un cambio de archivo por sí solo no constituye un hueco.

| Temporalidad | Ventanas train | Ventanas validation | Ventanas test | Total |
|---|---:|---:|---:|---:|
| M5 | 308.635 | 102.155 | 60.354 | 471.144 |
| M15 | 51.606 | 18.168 | 10.522 | 80.296 |

Los índices incluyen temporalidad, partición, segmento, filas inicial/final de contexto y objetivo, instante de decisión, final exclusivo del objetivo, fecha en Nueva York, archivos de procedencia, calidad y disponibilidad de amount. La fecha y la franja NY se calculan con `America/New_York`, incluyendo cambios de horario de verano.

También se identifican ventanas cuyo horizonte completo cae entre 08:30 y 13:00 NY:

| Temporalidad | Train NY | Validation NY | Test NY |
|---|---:|---:|---:|
| M5 | 100.081 | 32.534 | 19.426 |
| M15 | 17.851 | 6.565 | 3.661 |

El entrenamiento corto usa todas las ventanas elegibles de su partición; la etiqueta NY está disponible para análisis o filtrado posterior. No se ha ejecutado un experimento dedicado exclusivamente a esa sesión.

No hace falta convertir cada vela en «comprar/vender» para este objetivo de SFT: Kronos aprende a predecir los tokens de las próximas barras. Un sistema de decisiones de trading requeriría un objetivo y una evaluación adicionales; estos no están implementados por el simple hecho de disponer del modelo.

## 3. Entrada y normalización

La entrada al tokenizador usa seis canales: open, high, low, close, volume y amount. Los datos auditados no proporcionan amount; se utiliza cero explícito en la vista de entrenamiento y se conserva `amount_available=False`. Ese cero no representa un importe negociado observado ni una máscara de datos faltantes. No se escribe de vuelta a las fuentes.

Las unidades y procedencia del volumen siguen siendo una limitación de interpretación. El volumen observado no se convierte automáticamente en volumen centralizado del mercado de oro.

Cada ventana utiliza exclusivamente su contexto para calcular media y desviación estándar por canal, en float32, con epsilon 1e-5 y recorte a [-5,5]. Estas mismas estadísticas se aplican a sus objetivos y al pronóstico. La desviación tiene ddof=0. Se comprobó que alterar el futuro no modifica las estadísticas del contexto. El recorte es una decisión explícita: limita valores extremos y puede afectar pronósticos durante episodios excepcionales.

Las variables de calendario son minuto, hora, día de semana, día y mes en UTC. Son conocidas para el horizonte; no incluyen precios futuros.

## 4. Máscaras y prevención de fuga de información

Se aplican controles con funciones distintas:

1. **Elegibilidad de ventana:** se excluyen ventanas incompletas o que atraviesen huecos o límites de partición. No se rellenan barras inexistentes.
2. **Atención causal:** una posición no puede acceder a posiciones posteriores. Se identificó que la atención cruzada de la dependencia entre niveles de tokens del código fijado cambiaba su causalidad según `self.training`. El adaptador mantiene causalidad tanto en entrenamiento como en evaluación y exige secuencias alineadas de igual longitud. Conserva los pesos originales de esa capa.
3. **Máscara de pérdida:** se calcula la pérdida únicamente sobre los H tokens objetivo que siguen al contexto. El desplazamiento next-token hace que los índices activos sean 127–138 para M5 y 63–66 para M15. El primer índice activo predice precisamente la primera barra futura, no una barra anterior.

No se utiliza padding: todas las ventanas tienen longitud fija y completa. El adaptador rechaza máscaras de padding porque no se ha implementado ni probado ese caso. No se presentan los ceros de amount como padding o como valores ausentes ignorados por la pérdida.

El tokenizador preentrenado se mantiene congelado. Se comprueba su invariancia respecto al futuro y el prefijo. La pérdida promedia la entropía cruzada de los dos niveles discretos de Kronos, sobre las posiciones activas. La evaluación autoregresiva empieza únicamente con el contexto, genera ambos niveles paso a paso y nunca toma el precio real futuro como entrada. El teacher forcing se usa para la pérdida supervisada, no para construir el pronóstico libre.

## 5. Modelo y código reproducibles

Se utiliza **Kronos-small**, del proyecto financiero Kronos, con **Kronos-Tokenizer-base**. El fragmento anterior basado en `ChronosPipeline` y `amazon/chronos-2` correspondía a otro modelo; además, crear un DataLoader y cargar pesos no ejecutaba entrenamiento ni resolvía las máscaras.

- Repositorio: https://github.com/shiyu-coder/Kronos
- Commit: `67b630e67f6a18c9e9be918d9b4337c960db1e9a`
- Modelo: https://huggingface.co/NeoQuasar/Kronos-small
- Revisión modelo: `901c26c1332695a2a8f243eb2f37243a37bea320`
- Tokenizador: https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base
- Revisión tokenizador: `0e0117387f39004a9016484a186a908917e22426`

Se incluyen el código fijado y su licencia MIT. El cuaderno contiene un paquete embebido con huella de integridad; no depende de una rama mutable de GitHub para cargar el código. Los pesos se descargan por revisión fija. El ajuste causal es un adaptador de este proyecto; no debe confundirse con un cambio publicado por los autores.

## 6. Pruebas ejecutadas

Entorno local: Windows, Python 3.13.5, PyTorch 2.9.1+cpu, NumPy 2.2.6, Pandas 2.3.3, float32. Colab instaló sus dependencias, pero todavía no ha repetido las pruebas de datos/modelo por la autorización pendiente.

| Control | Resultado local |
|---|---|
| Integridad de 30 Parquet antes/después | Sin cambios |
| Conteos, continuidad y particiones | Correctos |
| Índices exactos de la máscara de pérdida | Correctos M5/M15 |
| Normalización independiente del futuro | Correcta |
| Tokenizador: invariancia futura y de prefijo | Correcta |
| Predictor causal en entrenamiento y evaluación | Diferencia máxima 0 en la prueba |
| Pronóstico independiente de valores futuros reales | Correcto |
| Backpropagation y actualización de parámetros | 3 pasos reales por temporalidad |
| Tokenizador congelado e inalterado | Correcto |
| Recarga de checkpoint y pronóstico idéntico | Correcta |
| Reanudar 3 pasos + 1 frente a 4 continuos | Diferencia máxima de parámetros 0,0 |

La comprobación exacta de reanudación se hizo en M15, en CPU y con la misma configuración; no implica identidad bit a bit en cualquier GPU o versión de PyTorch.

El optimizador es AdamW (learning rate 1e-5, weight decay 0,01), con recorte de norma de gradientes a 1. Se guardan modelo, optimizador, estado RNG, posición dentro del muestreo, paso, configuración, revisiones y referencia del manifiesto fuente.

### Métricas de comprobación, sin interpretación de rentabilidad

Cada validación corta usa solo dos ventanas fijas y espaciadas de validation. Estos números verifican que el circuito calcula resultados; no estiman rendimiento general.

| Temporalidad | MAE close, generación libre | MAE persistencia del último close | Ventanas |
|---|---:|---:|---:|
| M5 | 1,356130 | 1,223658 | 2 |
| M15 | 2,321136 | 2,338394 | 2 |

La persistencia supera a esta prueba M5. La diferencia M15 sobre dos ejemplos no permite concluir que exista ventaja. La celda de entrenamiento completo también tiene una validación limitada para seguimiento; debe ampliarse antes de seleccionar un modelo para evaluación sustantiva. No se calculó profit factor, rentabilidad neta, costes, drawdown o ejecución MT5.

## 7. Ticks y otras limitaciones pendientes

Los 48.875.569 ticks locales y 2.575 bloques BI5 válidos corresponden a la auditoría separada de la fuente de ticks. Esa colección no se incorporó al SFT actual. Permanecen 30 bloques de origen sin resolver en seis días; no se certificó la equivalencia exacta con el ZIP remoto ni las unidades/escala necesarias para una integración fiable.

Una colección limitada a 4,5 horas NY por día produce como máximo 54 barras M5 o 18 M15 por sesión. Esto no alcanza las longitudes totales actuales de 140 y 68 barras. No es correcto concatenar días como si fueran tiempo continuo para completar esas ventanas.

El posible solapamiento entre datos históricos de evaluación y el preentrenamiento del modelo abierto impide afirmar que test sea un holdout completamente nuevo para el modelo base. La separación temporal evita fuga por el ajuste de este proyecto, pero no certifica la historia del preentrenamiento.

El paquete prepara una base de investigación. Aún faltan entrenamiento suficiente, validación temporal amplia, evaluación fuera de muestra adecuada y pruebas con costes de ejecución antes de afirmar utilidad para operar.

## 8. Continuar en Colab

1. En la ventana abierta, completar la autorización **Conectar con Google Drive**. Google solicita este permiso para que el código del cuaderno pueda acceder a Drive; la interfaz advierte que también permite modificar archivos.
2. La ejecución iniciada podrá continuar con la preparación, descarga de pesos, pruebas cortas y escritura de resultados en una nueva carpeta de ejecución. Si Colab hubiera interrumpido la cola, ejecutar las secciones desde el montaje de Drive en orden.
3. Confirmar en la sección 7 `data_verified`, `windows_prepared`, `model_loaded`, `smoke_passed` y `sources_unchanged` con valor true.
4. El entrenamiento completo está preparado en la sección 8 y desactivado mediante `RUN_FULL=False`. No es necesario activarlo para terminar la verificación corta. Cambiarlo a true inicia una ejecución larga para la temporalidad elegida; no se ha ejecutado en esta entrega.
5. La sección 9 contiene un ejemplo acotado de reanudación. Deben mantenerse temporalidad y batch size del checkpoint. No activar simultáneamente entrenamientos redundantes.

## 9. Archivos de entrega y checkpoints

El ZIP contiene el cuaderno, el núcleo Python, el verificador de reanudación, el código oficial fijado, manifiestos, índices de ventanas, resultados JSON y este informe. No duplica los datos fuente ni incluye las dependencias instaladas o los pesos base, que se recuperan por revisión fija.

Los dos checkpoints de prueba se guardan en la carpeta de Drive indicada, cada uno dividido en tres partes para respetar el límite de transferencia. Son artefactos de tres pasos, no modelos terminados. `checkpoint_parts.json` contiene tamaños y SHA-256 de cada parte y del archivo completo.rgar checkpoints de procedencia desconocida con deserialización Python.


Para reconstruirlos, descargar las seis partes en una carpeta y ejecutar desde el ZIP extraído:

```bash
python restaurar_checkpoints.py --manifest checkpoint_parts.json --parts RUTA_PARTES --output RUTA_CHECKPOINTS_NUEVA
```

El restaurador verifica cada parte y el resultado completo antes de producir el archivo `.pt`; rechaza sobrescribir archivos existentes. La restauración local se verifica por comparación de SHA-256. Para reanudar se necesitan también los pesos/configuración del tokenizador y modelo base y los datos originales auditados. No ca
Evidencia principal: `resultados/status.json`, `resultados/source_manifest.json`, `resultados/tests_M5.json`, `resultados/tests_M15.json`, `resultados/resume_verification.json` y los historiales por temporalidad.
