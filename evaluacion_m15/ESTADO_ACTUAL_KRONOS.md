# AXXEL CAPITAL — ESTADO ACTUAL DE KRONOS

**Fecha de referencia:** 20 de noviembre de 2026  
**Modelo:** NeoQuasar/Kronos-small  
**Tokenizador:** Kronos-Tokenizer-base  
**Activo:** XAUUSD  
**Estado:** RESEARCH_ONLY — ORDERS DISABLED

## 1. Objetivo principal

Entrenar y validar Kronos-small para predecir el comportamiento futuro de XAUUSD con la mayor precisión posible.

Objetivo operativo de investigación: predecir los próximos cuatro cierres M15, equivalentes a 60 minutos.

Antes de desarrollar estrategias de trading, el modelo debe superar benchmarks predictivos mediante evaluaciones reproducibles y sin data leakage.

## 2. Estado científico verificado

| Fase | Estado |
|---|---|
| I — Diagnóstico y baselines | Completada |
| II — Inferencia Monte Carlo | Completada |
| III — Cabezas Huber/Laplace | Completada |
| IV — Adaptadores LoRA | Completada y auditada |
| V — Entrenamiento conjunto LoRA + Huber | Propuesta; pendiente |
| Test 2023–2024 | Sellado |

La auditoría del 20 de noviembre verificó los 37 artefactos del manifiesto de Fase IV, incluidos cinco adaptadores LoRA y sus estados de recuperación.

## 3. Resultados actuales

Evaluación diagnóstica 2020:

- 950 ventanas.
- 3.800 predicciones.
- 243 sesiones de Nueva York.

| Modelo | MAE |
|---|---:|
| Persistencia | 3.560039 |
| Kronos base | 3.821930 |
| Mejor cabeza Huber | 3.552607 |
| LoRA Q/V + cabeza congelada | 3.558846 |
| LoRA Q/V greedy | 3.899300 |

La mejor cabeza Huber consiguió una mejora puntual aproximada del 0,21 % frente a persistencia.

**No existe todavía evidencia estadística suficiente de una mejora predictiva reproducible.**

Los experimentos de Fase IV no justifican continuar optimizando exclusivamente Cross-Entropy discreta.

## 4. Próxima misión — Fase V

**Protocolo:** `M06_FV_JOINT_RETURN_V1`

Hipótesis: adaptar conjuntamente las representaciones latentes causales de Kronos y la cabeza continua Huber puede reducir el error de predicción.

Experimentos:

- Brazo A: Kronos congelado + cabeza Huber entrenable.
- Brazo B: Kronos congelado + LoRA Q/V rank 8 + cabeza Huber entrenable.
- Semillas: 271828, 271829 y 271830.
- Seis corridas experimentales.
- 300 pasos adicionales por corrida.
- Contexto L=56.
- Horizonte H=4.
- Temporalidad M15.
- Entrenamiento: 2010–2019.
- Desarrollo: 2020.
- Test 2023–2024: prohibido acceder.

## 5. Infraestructura de trabajo

**Proyecto local:** `C:\Users\HP\OneDrive\Escritorio\modelo_SFT`

**Editor:** Visual Studio Code.

**Agente:** Codex CLI local.

**Control del navegador:** Python + Playwright + Chrome CDP 9222.

**Notebook maestro:**  
https://colab.research.google.com/drive/1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss

La conexión local de Playwright al notebook está verificada.

La recuperación de los modelos en el runtime activo, el estado actual de la GPU y las pruebas de preejecución permanecen pendientes de confirmación.

## 6. Próxima tarea ejecutable

1. Reconciliar archivos locales con el informe más reciente.
2. Verificar el estado real del runtime de Colab.
3. Restaurar Fases III–IV sin reentrenar.
4. Comprobar integridad SHA-256.
5. Reproducir las predicciones del paso cero.
6. Verificar causalidad y gradientes.
7. Detectar experimentos equivalentes ya realizados.
8. Ejecutar Fase V únicamente cuando pasen las pruebas obligatorias.
9. Generar métricas, gráficas y checkpoints recuperables.

## 7. Reglas de seguridad

- `RESEARCH_ONLY = true`
- `orders_enabled = false`
- `test_read = false`
- No repetir experimentos completados.
- No sobrescribir checkpoints originales.
- No modificar datasets fuente.
- No permitir dos agentes escritores simultáneos.
- No declarar rentabilidad sin evidencia.
- Guardar el estado después de cada tarea significativa.

## 8. Historial preservado

La evaluación local del 11 de noviembre de 2026 permanece como evidencia histórica:

- 5.065 ventanas M15.
- 20.260 cierres predichos.
- MAE Kronos: 2.246910.
- MAE persistencia: 2.032728.
- MAE NY Kronos: 2.943721.
- MAE NY persistencia: 2.661346.

Esta evaluación corresponde a otra configuración y población de validación. Sus métricas no son directamente comparables con la evaluación diagnóstica de Misión 06.

**Último hito completado:** Fase IV auditada.

**Próxima misión:** recuperación del runtime y ejecución controlada de Fase V.

**Norma de continuidad:** actualizar este documento después de cada avance importante, utilizando solamente resultados comprobados.