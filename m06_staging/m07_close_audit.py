"""Persist evidence-only mission report; preserve previous project state."""
import datetime, hashlib, json, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
stage = root / 'm06_staging'
out = stage / 'm07_audit_20261120T222639017959Z'
evidence = json.loads((out / 'evidence.json').read_text())
nbpath = stage / 'm07_runtime_20261120T223326852891Z/notebook_before.ipynb'
nb = json.loads(nbpath.read_text(encoding='utf-8'))
cell = next(c for c in nb['cells'] if 'M07_RUNTIME_READONLY_PREFLIGHT' in ''.join(c.get('source',[])))
runtime = json.loads(''.join(cell['outputs'][0]['text']).splitlines()[1])
def save(name, text):
    with (out / name).open('x', encoding='utf-8') as f:
        f.write(text)
save('runtime_evidence.json', json.dumps(runtime, indent=2))
protocol = stage / 'audit_20261120/PROTOCOLO_FASE_V_JOINT_PROPUESTO_20261120.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
save('runs_status.json', json.dumps({'protocol': 'M06_FV_JOINT_RETURN_V1', 'protocol_sha256': sha(protocol), 'training_authorized_by_preflight': False, 'remote_duplicate_search_complete': False, 'runs': [{'arm': a, 'seed': s, 'status': 'blocked_preflight', 'steps_executed_this_mission': 0, 'remote_previous_status': 'unverified'} for a in ['control_head_only','joint_QV_head'] for s in [271828,271829,271830]]}, indent=2))
save('ESTADO_ANTERIOR.md', (root / 'ESTADO_ACTUAL_KRONOS.md').read_text(encoding='utf-8'))
report = '''# MISIÓN 07 — recuperación y diagnóstico de preejecución

Fecha: 20-11-2026. RESEARCH_ONLY=true; orders_enabled=false; test_read=false.

## Dictamen
Fases III–IV recuperadas y verificadas localmente. Fase V no iniciada: Drive no está montado en el runtime reconectado. No se aprobaron los controles que dependen de artefactos remotos y modelos vivos. No se reentrenó ninguna fase ni se leyeron datasets originales/test.

## Estado científico y cronología
- 08-11: pruebas locales de tres pasos M5/M15; no equivalen a entrenamiento completo.
- 11-11: evaluación base L64, 5.065 ventanas 2020–2022. MAE 2,2469 frente a persistencia 2,0327. Población distinta; no mezclar con M06.
- INFORME_RESULTADOS registra otro piloto Colab de tres pasos del 12-11 UTC sobre 24 ventanas; tampoco representa Fase IV.
- Auditoría 20-11: cinco pilotos Fase IV de 100 pasos, seis salidas; tres cabezas Huber de Fase III. Validación diagnóstica L56/H4: 950 ventanas, 243 sesiones NY de 2020. Fases I–II constan como completadas en la memoria; esta misión no reaudita todos sus artefactos.
- Se reprodujo la auditoría local en destino nuevo, sin ejecutar sus escrituras sobre la evidencia histórica.

## Integridad superada
37/37 archivos de Fase IV: SHA-256 y tamaños. Cinco adaptadores finales y cinco resume.pt: nombres, shapes, finitud, 147.456 parámetros por variante, paso100, semilla271828, hashes declarados de modelo/tokenizador; adaptador final idéntico al de reanudación. Estados Adam y RNG NumPy/torch/CUDA presentes y comprobados según el auditor histórico. Esto no acredita restauración viva del optimizador ni RNG Python.

11 archivos seleccionados de Fase III verificados, incluidas tres cabezas de 33.092 parámetros y scalers. No se dispone de una verificación nueva de todo el caché III. Compatibilidad estructural no equivale a inferencia restaurada.

Manifiesto IV SHA-256: e9e0e3754deaafb0153d11dfce2bf0abd024669d2a253f2f7768c8a6bfa7effc.
Se recalcularon seis métricas IV, tres métricas Huber y 12 intervalos IV desde predicciones persistidas. Tolerancias históricas: métricas 1e-10; CE 1e-6. Los IC históricos usan 2.000 réplicas, no las 10.000 previstas para Fase V.

## Acceso y runtime observado
CDP9222: Chrome157; Playwright conectado a una única pestaña del notebook maestro exacto. Inicialmente desconectado por inactividad; se pulsó Volver a conectar sin reiniciar ni ejecutar celdas históricas. Se respaldó el notebook mediante descarga UI antes del diagnóstico.

Una celda nueva M07_RUNTIME_READONLY_PREFLIGHT ejecutada: PID516, Tesla T4, 15.360 MiB totales y 14.913 MiB libres; Drive disponible=false, carpeta canónica disponible=false, variables con prefijos m6_/M6_=[] . El inventario de procesos no demuestra ausencia absoluta de trabajos externos. No se observó entrenamiento activo en metadatos del respaldo. La GPU no prueba que se hayan recuperado modelos.

El primer intento restringido de Playwright falló por WinError5; el acceso autorizado fuera del sandbox funcionó. Un selector de descarga requirió adaptación. La guarda de diagnóstico existente impidió repetir la celda: el respaldo posterior confirma ejecución1 y su salida. No se creó un segundo diagnóstico.

## Preflight
APROBADO local: integridad III seleccionada/IV completa, estructura de checkpoints, métricas persistidas, identidad del notebook, GPU observada, protocolo localizado.
NO APROBADO/PENDIENTE: Drive y hashes remotos actuales; modelos/tokenizador vivos; recarga estricta dos veces en namespace vacío; predicciones paso0 y tolerancia CPU/GPU justificada; invariancia futura, normalización contextual, causalidad, alineación H1–H4 y separación train/dev en la nueva ejecución; gradientes LoRA/cabeza y congelamiento efectivo; restauración RNG/optimizador; guardado remoto verificado; deduplicación por huella científica. Las pruebas históricas no sustituyen estas comprobaciones.

## Protocolo confirmado y seis corridas
M06_FV_JOINT_RETURN_V1: A=base congelada+cabeza Huber (33.092); B=base congelada+QV rank8 alpha16 en ocho bloques+cabeza (164.164). Inicialización: cabeza III correspondiente a cada semilla; LoRA nuevo B=0, sin continuar adaptadores CE IV. Semillas271828/271829/271830. Train2010–2019 (8.428 ventanas), dev2020 (950), L56/H4. 300 pasos adicionales, micro16×acumulación4; AdamW, weight_decay0, LR cabeza3e-4/LoRA1e-4, clip1; SmoothL1 beta1 sobre retornos acumulados bps escalados con scalers III train-only. Dropout LoRA0,1; base eval con autograd en ruta conjunta. Paso0 y300 para desarrollo, checkpoints cada25; no curvas de validación intermedia contrarias al protocolo.

| Brazo | Semilla | Pasos nuevos | Estado |
|---|---:|---:|---|
| A | 271828 | 0 | Bloqueada por preflight |
| A | 271829 | 0 | Bloqueada por preflight |
| A | 271830 | 0 | Bloqueada por preflight |
| B | 271828 | 0 | Bloqueada por preflight |
| B | 271829 | 0 | Bloqueada por preflight |
| B | 271830 | 0 | Bloqueada por preflight |

No se puede excluir que existan resultados remotos nuevos bajo otros nombres. No se crearon checkpoints ni gráficas Fase V, ni se implementó o validó su motor de entrenamiento/persistencia. runs_status.json es un registro de bloqueo local, no un registro remoto de deduplicación.

## Comparación histórica verificada, no resultados Fase V
| Referencia | MAE | RMSE |
|---|---:|---:|
| Persistencia | 3,560039 | 5,391019 |
| Kronos base | 3,821930 | 5,681343 |
| Huber271828 | 3,555176 | 5,401539 |
| Huber271829 | 3,553271 | 5,395150 |
| Huber271830 | 3,552607 | 5,401648 |
| QV + cabeza congelada | 3,558846 | 5,399713 |
| QV greedy | 3,899300 | 5,766090 |

Mejor Huber: deltaMAE −0,007433; IC95 histórico [−0,022115; 0,007603]. Incluye cero. No se demuestra superioridad. Los archivos evidence.json conservan sesgo, dirección y los detalles de hashes/arquitectura. Las predicciones y figuras IV históricas se preservan en audit_20261120/fase_IV; no se presentan como gráficas generadas hoy en Colab.

## Recuperación y próxima acción
Checkpoints locales: m06_staging/audit_20261120/fase_IV/checkpoints/ y fase_III/checkpoints/fase_III/7587400db64558e6/. Ruta remota canónica: /content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571/ . Restauración viva pendiente.

Montar/autorizar Drive en el runtime actual sin ejecutar la celda histórica de preparación; inspeccionar artefactos y posibles parciales, verificar hashes y recuperar III–IV de forma idempotente. Después completar paso0 y controles científicos antes de implementar/lanzar Fase V. No proponer aún evaluación2021–2022 ni abrir test. La hipótesis Joint LoRA+Huber sigue sin probar.
'''
save('INFORME_MISION07.md', report)
state = '''# AXXEL CAPITAL — ESTADO ACTUAL KRONOS

Actualizado: 20-11-2026, Misión07. RESEARCH_ONLY=true; orders_enabled=false; test_read=false.

Objetivo: Kronos-small + Kronos-Tokenizer-base, XAUUSD M15, L56/H4; mejorar predicción continua frente a persistencia. No trading.

## Estado científico
I–II completadas según memoria histórica. III–IV completadas; reauditoría local M07 superada: 37 archivos IV y 11 seleccionados III, cinco adaptadores/resume y tres cabezas Huber. No reentrenar. Fase V NO INICIADA: seis corridas bloqueadas por preflight, cero pasos nuevos. Existencia de nuevos parciales remotos aún no verificada.

950 ventanas/243 sesiones NY de desarrollo2020: MAE persistencia3,560039; base3,821930; mejor Huber271830=3,552607; QV+cabeza congelada3,558846. IC95 delta mejor Huber−persistencia=[−0,022115;0,007603]: sin superioridad demostrada. Métricas recalculadas desde predicciones, no inferencia nueva.

## Runtime comprobado
CDP9222/Playwright: notebook exacto 1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss, título Kronos_SFT_verificado.ipynb. Desconectado por inactividad inicialmente; reconectado sin reinicio. Celda nueva de diagnóstico ejecutada una vez: Tesla T4, 15.360MiB totales/14.913MiB libres, PID516; Drive NO montado, carpeta canónica inaccesible, sin variables M6. No asumir persistencia de este runtime en próxima sesión.

## Evidencia y recuperación
Informe/evidencia/registro de seis corridas/manifiesto nuevos: `m06_staging/m07_audit_20261120T222639017959Z/`.
Respaldo previo: `m06_staging/m07_runtime_20261120T223151636075Z/notebook_before.ipynb`.
Respaldo con diagnóstico y salida: `m06_staging/m07_runtime_20261120T223326852891Z/notebook_before.ipynb`.
Fuentes III–IV locales: `m06_staging/audit_20261120/`.
Remota: `/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571/`.
SHA256 manifiestoIV: `e9e0e3754deaafb0153d11dfce2bf0abd024669d2a253f2f7768c8a6bfa7effc`.

## Próxima acción obligatoria
Montar/autorizar Drive, verificar artefactos actuales y deduplicar por huella; recuperar modelos/checkpoints sin entrenar III–IV. Reproducir paso0, probar recarga/causalidad/gradientes/congelamiento/separación de datos/RNG/guardado. Solo entonces Fase V según `m06_staging/audit_20261120/PROTOCOLO_FASE_V_JOINT_PROPUESTO_20261120.json`: dos brazos×tres semillas,300pasos,micro16×4,AdamW sin weight decay,LR cabeza3e-4/LoRA1e-4; evaluar dev paso0/300. Motor de Fase V y persistencia todavía no implementados/validados. Sin gráficas ni checkpoints nuevos de entrenamiento.

## Reglas e historia
Leer AGENTS.md. Un solo escritor del notebook; preservar fuentes/checkpoints; test2023–2024 sellado. No ejecutar automatización antigua que repite celdas1–8. Informes08-11 y evaluación11-11 (5.065 ventanas,L64,2020–2022) son históricos y no comparables directamente con M06. Estado anterior preservado en la carpeta M07.
'''
(root / 'ESTADO_ACTUAL_KRONOS.md').write_text(state, encoding='utf-8')
manifest = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in out.iterdir() if p.is_file()}
manifest['external_evidence'] = {str(p.relative_to(root)): sha(p) for p in [nbpath, protocol, root / 'AGENTS.md', root / 'ESTADO_ACTUAL_KRONOS.md']}
save('checksums.json', json.dumps(manifest, indent=2))
print(out)
print('evidence_sha256', sha(out / 'evidence.json'))
print('report_sha256', sha(out / 'INFORME_MISION07.md'))
