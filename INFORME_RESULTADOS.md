# AXXEL CAPITAL — KRONOS LoRA | INFORME DE RESULTADOS

**Ejecución real:** 20261112T024524Z_6f6ef2, Google Colab. **Decisión: LoRA técnicamente viable; ventaja predictiva no demostrada.**

## 1. Arquitectura verificada
Kronos-small: 8 bloques Transformer, d_model=512, 8 cabezas, FFN=1024, RMSNorm, RoPE, embeddings jerárquicos y doble cabezal s1/s2 (1024 clases cada uno). Tokenizador BSQ congelado, entrada de 6 variables, dimensión 256, 20 bits (10+10), 3 bloques encoder y 3 decoder reales. La configuración nominal de 4 capas genera n-1 bloques en la fuente oficial: no se confunde el número configurado con el instanciado.
Fuente oficial del commit 67b630e67f6a18c9e9be918d9b4337c960db1e9a: tres archivos verificados byte a byte. Revisión de pesos {'Kronos-small': '901c26c1332695a2a8f243eb2f37243a37bea320', 'Kronos-Tokenizer-base': '0e0117387f39004a9016484a186a908917e22426'}. Existe una modificación previa explícita: atención cruzada causal en train/eval y rollout greedy con s1 alineado. Se conserva idéntica en base y LoRA.

## 2. Configuración LoRA
Rank 8, alpha 16 (escala 2), dropout 0.10, LR 0.0001 fijo, AdamW, weight decay 0.01, clipping de gradiente 1.0, batch 1, FP32, seed 271828. Dieciocho módulos Q/V: 16 en autoatención + 2 en dependencia. Sin entrenar el tokenizador ni los pesos originales. Contexto 56 velas (14 horas) y horizonte 4 (1 hora).

## 3. Hardware empleado
CPU de Colab, 2 hilos, RAM total 12.671 GiB. Pico RSS del proceso muestreado durante pasos 1.441 GiB; incluye otras variables de la sesión. Piloto 1.631 s; media por paso 0.517 s. Evaluación base 13.602 s y LoRA 12.396 s. GPU no disponible: memoria y consumo GPU N/A.
CPU actual es el mínimo **demostrado**; no se probaron CPUs menores. Presupuesto estático de parámetros/gradientes/Adam: 111.73 MiB, excluyendo activaciones y memoria del runtime. T4 16 GB es candidata inicial a medir para ampliar ensayos; L4 24 GB, A100 40/80 GB y H100 80 GB no se midieron ni son necesarios para este piloto. Aumentar batch de 1 a 2/4/8 solo tras medir pico reservado <80% y estabilidad. Escalar cuando tiempo o memoria excedan el presupuesto; más GPU no demuestra mejores predicciones.

## 4. Parámetros
Predictor base congelado: 24,741,376. Adaptadores entrenables: 147,456. Predictor adaptado total: 24,888,832; fracción entrenable 0.5925%. Tokenizador congelado adicional: 3,958,042. Total combinado: 28,846,874.

## 5. Prueba piloto
Tres pasos, tres ventanas train 2010–2019 registradas exactamente. Todas las pérdidas y gradientes fueron finitos. Los 36 tensores A/B cambiaron. Hashes de todos los pesos/buffers originales y del tokenizador permanecen idénticos. Inicialización B=0 reprodujo exactamente la salida base. Checkpoint de adaptadores recargado y predicción idéntica.
Pérdida eval de la misma ventana train: 3.86023331 → 3.85743475; esta observación no mide generalización. Las pérdidas de entrenamiento por paso corresponden a ventanas distintas, por lo que su trayectoria no demuestra convergencia.

## 6. Gráficas de entrenamiento
Panel B interactivo y PNG real: pérdida, LR, norma de gradiente, RAM, tiempo y GPU N/A. Panel A: arquitectura, parámetros congelados/entrenables y distribución por módulo. Tablas completas y hashes en auditoria/ y metricas/.

## 7. Predicción y validez
Panel C interactivo: selector de las 24 ventanas con 56 velas de contexto y 4 objetivos, errores por horizonte e histograma. 24 días distintos de validación 2020–2022, 96 cierres por modelo. Anclas 08:30/09:30/10:30/11:30 America/New_York con DST. Estadísticas de normalización exclusivamente del contexto; 60 velas continuas, ninguna cruza particiones. Pruebas de invariancia a futuro pasan en normalización, tokenizer, atención train/eval y rollout. Test 2023–2024 no se leyó en esta misión.

## 8. Comparación contra base y persistencia
LoRA: MAE 3.80863444, RMSE 6.24949998, precisión direccional 45.8333%.
Base: MAE 3.80863444, RMSE 6.24949998, precisión direccional 45.8333%.
Persistencia: MAE 3.56029129, RMSE 6.20043013. Predice cambio cero, por lo que no emite dirección (cobertura 0%; acierto condicionado N/A).
Diferencia máxima de precios base/LoRA: 0.00000000. Tres pasos cambiaron adaptadores y pérdida, pero no los tokens ganadores greedy de esta muestra. LoRA no supera persistencia en MAE. Resultados por NY y año incluidos. Base se refiere a pesos originales con el mismo protocolo causal greedy, no al muestreador oficial KronosPredictor.

## 9. Errores y limitaciones
Se corrigió un error de sintaxis durante la inserción de la primera celda; todas las celdas finales se ejecutaron correctamente. Veinticuatro ventanas no bastan para establecer un efecto estadístico; los cuatro horizontes por ventana están correlacionados. No hay intervalos de confianza, prueba final, backtest ni costes de transacción. Posible solapamiento con preentrenamiento externo desconocido. Datos BID; volumen de procedencia pendiente y amount ausente codificado con cero. No existe auditoría económica de rentabilidad. GPU no perfilada. Los checkpoints históricos se preservaron y el último se cargó estrictamente; su L=64 impide confundirlo con este ensayo L=56.

## 10. Conclusión y siguiente fase
**GO técnico:** la integración LoRA funciona y el laboratorio está preparado para diseñar un SFT supervisado controlado. **NO-GO predictivo/trading:** no se ha demostrado mejora, ventaja ni rentabilidad. Antes de un entrenamiento amplio, fijar presupuesto, validación temporal/walk-forward, semillas, métricas y criterio de parada; ampliar la comparación base/persistencia en validación y resolver volumen/amount. Conservar test 2023–2024 para una evaluación final sin ajuste de hiperparámetros. Ningún entrenamiento completo ni orden de trading fue iniciado.

### Reproducción y artefactos
En runtime nuevo: ejecutar dependencias originales, montar Drive y ejecutar las celdas Misión 04 1–7 en orden. La sección 2 usa la implementación histórica fijada y recupera pesos por revisión si faltan. Cada sección 1 crea un directorio nuevo. La celda de piloto restaura los adaptadores iniciales al repetirse: no acumula pasos.
Carpeta: /content/drive/MyDrive/AXXEL_CAPITAL/MISION04_LORA/20261112T024524Z_6f6ef2. Checkpoint: checkpoints/lora_step003.pt; contiene solo adaptadores, optimizador, configuración y RNG, y exige pesos base fijados para recargar. El RNG guardado corresponde al momento posterior a las pruebas de auditoría; este piloto no certifica una reanudación de entrenamiento largo.
Respaldo previo: https://colab.research.google.com/drive/12247HRxRoitH1eD8HEdKPqC_rAhvd6Eq
Fuentes: https://github.com/shiyu-coder/Kronos/tree/67b630e67f6a18c9e9be918d9b4337c960db1e9a ; https://huggingface.co/NeoQuasar/Kronos-small ; https://docs.cloud.google.com/compute/docs/gpus
