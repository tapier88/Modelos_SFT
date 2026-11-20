# Evaluación M15 de Kronos, 2020–2022

Abra resultados/INFORME.md para ver conclusiones, alcance y limitaciones.
Esta ejecución se completó en Windows CPU, fuera de Colab, sin entrenamiento.
Los datos 2023–2024 no se abrieron ni puntuaron.

## Repetir en el equipo original

```powershell
python evaluar_base_m15.py --bundle C:\workspace\kronos_sft_20261107 --source C:\workspace\kronos_audit_20261107\parquet --output resultados_nuevos
```

Use un directorio nuevo para otra ejecución. Si se interrumpe, repita exactamente
el comando y configuración para reanudar los fragmentos existentes. No cambie
el código ni el tamaño de lote al reanudar. No ejecute sobre los archivos fuente.

## Contenido y portabilidad

- evaluar_base_m15.py: protocolo, inferencia, métricas y reporte.
- test_metricas.py: verificaciones analíticas de métricas.
- finalizar_entrega.py: comprobación independiente y empaquetado del equipo original.
- bundle/: copia exacta del pipeline y código vendor usado, con licencia.
- datos/: únicamente los tres Parquet M15 2020, 2021, 2022.
- resultados/: predicciones, intervalos, hashes, pruebas, cobertura y gráfico.
- requirements_entorno.txt: versiones de las dependencias usadas.

Los pesos no se duplican en este ZIP. En otro equipo instale las dependencias,
obtenga los pesos de las revisiones indicadas en resultados/config.json y colóquelos
en bundle/weights/Kronos-small y bundle/weights/Kronos-Tokenizer-base
(config.json y model.safetensors). Compruebe sus hashes con
resultados/provenance_verified.json. Después ejecute:

```powershell
python evaluar_base_m15.py --bundle bundle --source datos --output resultados_nuevos
```

Se conserva la inferencia greedy del adaptador causal auditado del proyecto.
No es el predictor oficial con muestreo. Los resultados solo describen este
protocolo y estos datos; no constituyen un backtest de operaciones.
