# AXXEL CAPITAL — ESTADO ACTUAL KRONOS

## Diagnóstico M11 — 22-11-2026 12:20 UTC

Auditoría local completada, sin entrenamiento, inferencia nueva ni acceso al test. Informe y propuesta pendiente de ejecución: `scripts/M11_DIAGNOSTICO_GENERALIZACION_20261122.md`. Evidencia con SHA-256 y respaldo íntegro del estado anterior: `output/axxel/m11_audit_20261122T122015431003Z/`.

Hashes de resultado/protocolo/fuentes/scalers y nueve archivos de reconciliación verificados; resultado idéntico a salida del notebook archivado; agregados de seis corridas y nueve contrastes reconciliados. No se revalidaron checkpoints remotos ni se recalcularon métricas desde predicciones individuales. Diferencia media control–cabeza original +0,017127025; Joint–control +0,000272717. Son diagnósticos descriptivos sobre desarrollo reutilizado; no prueban sobreajuste, equivalencia ni superioridad.

Lectura actual de interfaz 12:19:03 UTC: única pestaña maestra, runtime conectado, backend GPU. Drive, modelos en memoria, GPU exacta y procesos activos UNKNOWN; no se ejecutaron celdas. Registro: `output/axxel/20261122T1219024663942Z/`. Las verificaciones remotas anteriores siguen siendo HISTORICAL.

Próxima acción propuesta: `M11_ADAPTER_OFF_FIXED_HEAD_V1`, ablación de inferencia ON/OFF del adaptador con la misma cabeza Joint final, tres semillas y dev2020; verificar primero commits/hashes y equivalencia. Propuesta NO EJECUTADA. No repetir I–V ni entrenar automáticamente. RESEARCH_ONLY=true; orders_enabled=false; test_read=false.

---
Estado de cierre M10 conservado a continuación; sus referencias a conexión «actual» corresponden a la fecha histórica indicada.

Actualizado: 22-11-2026 12:04 UTC / 07:04 Bogotá.
RESEARCH_ONLY=true; orders_enabled=false; test_read=false.

## Fase V completada — no repetir

M06_FV_JOINT_RETURN_V1: **6/6 corridas COMPLETED,300pasos cada una**. Resultado capturado02:03:30UTC y reconciliado12:04UTC contra la salida actual de la única celda de FaseV del notebook maestro. No se reenvió entrenamiento.

|Semilla|Control MAE|Joint MAE|
|---|---:|---:|
|271828|3,577933976|3,578414298|
|271829|3,574401693|3,574432133|
|271830|3,560099197|3,560406587|

Persistencia3,560039287; mejorHuberIII3,552606561. Media±SD: control3,570811622±0,009443849; joint3,571084339±0,009459133. Joint no mejoró el MAE frente a persistencia ni frente al control en ninguna semilla. Los9IC simultáneos incluyen0; followup_gate=false. Sin superioridad demostrada. Desarrollo2020 reutilizado; test2023–2024 sellado. Baseline greedy no verificado en esta evaluación.

Próxima misión: diagnóstico científico usando checkpoints/predicciones existentes antes de proponer otra ablación; no extender pasos ni repetir estas corridas automáticamente. Informe: `scripts/M10_CIERRE_FASE_V.md`.

## Evidencia y artefactos

- Resultado: `output/axxel/m10_phase_v_20261122/bootstrap_phase_v/result.json`, SHA256 `e78f0bbbf46ab0693ced209b1b2642bd6b23960e922b0774463a83a000b89321`.
- Reconciliación actual, notebook y seisPNG reales: `output/axxel/m10_reconciled_20261122T120350/`.
- Estado anterior íntegro: misma carpeta/`ESTADO_ANTERIOR.md`.
- Carpeta Drive reportada: `/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571/fase_V_joint_v1/f7efc796337d3684b0641688c1fa98eca67d0a01814d714ce99cce78a49f1213/`.
- Evaluación: `evaluation/20261122T020302277505Z/`; manifiesto SHA256 `0f02af8aeefc2f794f5b39516835b471fec36484b1c920c300a12c2a285427f4`.
- Checkpoints cada25pasos, estados de optimizador/RNG y manifests reportados por la ejecución. Verificar commits/hashes antes de restaurar; no se hizo una nueva relectura completa de Drive en esta reconciliación.

## Conexión actual e inicio diario

CDP127.0.0.1:9222 responde y conecta con timeout60000ms. Notebook exacto identificado; ventana visible en estado normal; runtime conectado según interfaz. El bloqueo local `output/axxel_start.lock` no existe. El fallo histórico de conexión fue un timeout después de conectar el WebSocket, no falta de puerto expuesto. Colab no se reinició.

`.\scripts\axxel.ps1 start` conecta/verifica/restaura; nunca entrena. `status` consulta interfaz; `recover/close` mantienen diagnóstico/respaldo. FaseV usa entrada explícita autorizada `automation/m10_execute_phase_v.py`; consultar transacciones y completados antes de usarla. No reenviar una entrega incierta. Un solo escritor del notebook.

## Memoria científica preservada

I–IV completadas; no repetir. Train2010–2019:8.428 ventanas;dev2020:950/243sesionesNY;L56,H4,M15. Evaluaciones08-11/11-11(L64,5.065ventanas2020–2022) no directamente comparables.

M09 aprobó causalidad, normalización contextual/scalers train-only, alineación, paso cero desde precios(max9,9209592e-7), gradientes y reanudación sintética. Dos restauraciones aisladas aprobadas; deduplicación interna/externa documentada. Certificación: `scripts/M09_READY_FOR_PHASE_V.md`.

Protocolo original: `m06_staging/audit_20261120/PROTOCOLO_FASE_V_JOINT_PROPUESTO_20261120.json`. Control33.092/joint164.164parámetros;QVrank8alpha16;micro16×acumulación4;AdamW weight_decay0;LRcabeza3e-4/LoRA1e-4;dev0/300. No modificar resultados históricos.

Pruebas de esta reconciliación: sintaxis Python y6tests de seguridad del transporte aprobados;29tests/parser aprobados antes del entrenamiento. Sin commit ni push.
