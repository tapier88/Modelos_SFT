# Modelos_SFT — AXXEL CAPITAL

Pipeline de investigación cuantitativa y **Supervised Fine-Tuning (SFT)** sobre **Kronos-small** (`NeoQuasar/Kronos-small` + `Kronos-Tokenizer-base`) adaptado con **LoRA** para el pronóstico y análisis de series temporales financieras en **XAUUSD (M5/M15)**.

Diseñado con enfoque en reproducibilidad, control de *data leakage*, trazabilidad experimental mediante hashes **SHA-256** y evaluación estructurada en Google Colab.

> **Estado del proyecto:** `RESEARCH_ONLY — PEDIDOS DESHABILITADOS`  
> Entorno enfocado exclusivamente en investigación, experimentación y validación cuantitativa. No ejecuta operaciones ni envía órdenes a mercados reales.

---

## Propuesta de valor técnica

* **Adaptación eficiente de parámetros (LoRA):** Ajuste fino supervisado sobre `Kronos-small` manteniendo la base congelada para reducir costo computacional y memoria en experimentos iterativos.
* **Prevención de fuga de datos (*no data leakage*):** Validación temporal estricta (*walk-forward*) y separación clara entre etapas de preparación, entrenamiento y evaluación.
* **Auditoría y trazabilidad:** Verificación de integridad de artefactos y estados mediante **SHA-256**, *snapshots* de avance y registros de control de ejecución.
* **Evaluación en M15:** Scripts y métricas dedicadas para contrastar el comportamiento del modelo base frente a iteraciones ajustadas.

---

## Estructura del repositorio

* `automation/` — Automatización y supervisión de sesiones en Google Colab (`controlar_kronos_colab.py`, `notebook_guard.json`, `resultado_control_colab.json`).
* `evaluacion_m15/` — Evaluación en temporalidad M15, cálculo de métricas (`test_metricas.py`), scripts de evaluación (`evaluar_base_m15.py`, `finalizar_entrega.py`) y documentación de ejecución (`ESTADO_ACTUAL_KRONOS.md`, `README_EJECUCION.md`).
* `m06_staging/` — *Snapshots* de experimentación, núcleo de ejecución (`stable_runtime_core.py`) y archivos de verificación de respaldo y estado.
* `informe_tecnico/` — Informes técnicos, auditorías y documentos de reconciliación del pipeline.
* `LÉAME.md` — Documentación complementaria en español.

---

## Flujo del pipeline

1. **Ingesta y control de integridad:** Validación de datos de series temporales y verificación de hashes SHA-256 para garantizar consistencia entre etapas.
2. **Entrenamiento SFT + LoRA:** Fine-tuning supervisado sobre representaciones tokenizadas (`Kronos-Tokenizer-base` + `Kronos-small`) para series XAUUSD.
3. **Validación *walk-forward* y evaluación:** Comparación sistemática de métricas en M15 y registro de estado (`status.json` y reportes técnicos) para auditar cada iteración.

---

## Stack tecnológico

* **Lenguaje y ciencia de datos:** Python, NumPy, Pandas
* **Deep Learning y fine-tuning:** PyTorch, Supervised Fine-Tuning (SFT), LoRA, Kronos-small / Kronos-Tokenizer-base
* **Flujo de trabajo y control:** Google Colab, Git, GitHub, Visual Studio Code
