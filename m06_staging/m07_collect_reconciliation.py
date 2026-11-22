import datetime, hashlib, json, pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'automation'))
from m07_integrity import cell_id,source,verify_preserved
out=ROOT/'m06_staging/m07_reconciliation_20261120T224952Z'
p=pathlib.Path(sys.argv[1]).resolve()
nb=json.loads(p.read_text(encoding='utf-8'))
original=json.loads((ROOT/'output/playwright/m07_20261120T224746735062Z/before.ipynb').read_text(encoding='utf-8'))
old_ids={cell_id(c) for c in original['cells']}
filtered={'cells':[c for c in nb['cells'] if cell_id(c) in old_ids]}
verify_preserved(original,filtered)
assert all(c.get('outputs',[])==next(x for x in filtered['cells'] if cell_id(x)==cell_id(c)).get('outputs',[]) for c in original['cells']), 'Historical outputs changed'
records={}
for c in nb['cells']:
    for output in c.get('outputs',[]):
        text=''.join(output.get('text',[]))
        for line in text.splitlines():
            for prefix,name in [('M07_RECOVERY_STATUS ','recovery'),('M07_STEP0_STATUS ','step0'),('M07_FINAL_RECOVERY ','final')]:
                if line.startswith(prefix):records[name]=json.loads(line[len(prefix):])
assert records['final']['report']['status']=='reconciliation_and_recovery_verified'
assert records['step0']['training_steps']==0
for name,value in records.items():
    with (out/(name+'_result.json')).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
final=records['final']
rows=records['step0']['step0']
summary='\n## Cierre del preflight ejecutado\n\n'
summary+='Las 950 ventanas de 2020 fueron inferidas de nuevo en GPU. Se reprodujeron los scalers usando únicamente los cachés train2010–2019; se verificaron hashes fuente, índices, continuidad, separación train/dev y 243 sesiones NY. Se aprobaron las pruebas causales train/eval, invariancia a futuros y congelamiento de base/tokenizer.\n\n'
summary+='| Semilla | MAE paso0 | Diferencia máxima de precio | Diferencia media absoluta |\n|---|---:|---:|---:|\n'
for r in rows:summary+=f"| {r['seed']} | {r['MAE']:.12f} | {r['max_abs_price_diff']:.12g} | {r['mean_abs_price_diff']:.12g} |\n"
summary+='\nTolerancias fijadas antes de inferir: máximo0,001 y media1e-5 unidades de precio; los errores observados son menores que1e-6 máximo. Se comprobaron paso0 neutral LoRA y gradientes finitos/no nulos de cabeza y LoRA_B, tanto en eval como con dropout LoRA activo. No hubo actualizaciones del optimizador. La prueba de aprendizaje posterior de LoRA_A sigue pendiente y no se declara aprobada.\n\n'
summary+=f"Predicciones por ventana de las tres semillas guardadas como NPZ y releídas, junto al informe y manifiesto final, en `{final['output']}`. Se verificaron {final['files_verified']} archivos; SHA-256 de final_checksums.json: `{final['manifest_sha256']}`.\n\n"
summary+='Los códigos y salidas de las 57 celdas previas a esta reconciliación permanecen iguales en el respaldo final. No se reinició el runtime, no se reejecutaron diagnóstico/montaje, no se leyeron años2023–2024 y no se inició Fase V.\n\n'
summary+='Pendiente antes de cualquier entrenamiento futuro: deduplicación científica completa, restauración integral repetida en namespaces nuevos, equivalencia de reanudación de RNG/optimizador, comprobación posterior de LoRA_A e implementación/pruebas transaccionales de Fase V. El preflight aprobado se limita a las comprobaciones efectivamente ejecutadas. No se declara preparación completa para entrenar.\n'
with (out/'INFORME_RECONCILIACION.md').open('a',encoding='utf-8') as f:f.write(summary)
state=ROOT/'ESTADO_ACTUAL_KRONOS.md'
with (out/'ESTADO_PRECIERRE.md').open('x',encoding='utf-8') as f:f.write(state.read_text(encoding='utf-8'))
state.write_text(f'''# AXXEL CAPITAL — ESTADO ACTUAL KRONOS

Actualización: 20-11-2026, cierre reconciliación M07. RESEARCH_ONLY=true; orders_enabled=false; test_read=false.

## Restricción vigente
El usuario pidió resolver Source mismatch, recuperar III–IV y avanzar preflight. **No iniciar Fase V**, no reiniciar runtime ni ejecutar celdas históricas. Leer AGENTS.md al comenzar.

## Estado recuperado
I–II completadas según evidencia histórica; III–IV completas. Integridad remota nueva: III70/70 archivos, IV37/37. Modelos fijados recuperados con hashes vivos correctos; tres cabezas Huber cargadas estrictamente dos veces; cinco adaptadores IV y sus estados Adam/RNG NumPy/torch restaurados en objetos aislados. No entrenamiento.

Source mismatch resuelto: Colab añadió cuatro espacios al print final de montaje. Se conservan original y variante reviewed; no son idénticos. Celda30 diagnóstico ID9wcbmH4SFnx3 intacta; montaje IDSOeTIq7YHoEB único y completado. Controlador corregido para transporte sin reindentación y comparación exacta; 5/5 pruebas. No repetir montaje. Celdas históricas y salidas preservadas.

## Preflight comprobado
Train2010–2019:8.428 ventanas; dev2020:950 ventanas/243 sesiones NY. Hashes, continuidad, índices, separación temporal, normalización contextual y scalers train-only reproducidos. Causalidad train/eval e invariancia futura aprobadas. Paso0 de tres cabezas reproducido en GPU: error máximo global {max(r['max_abs_price_diff'] for r in rows):.12g} unidades de precio. LoRA neutral idéntico; gradientes cabeza/LoRA_B finitos y no nulos, también con dropout activo; base/tokenizer sin gradientes. Cero pasos de optimizador.

MAE paso0 semillas271828/271829/271830: {rows[0]['MAE']:.9f}, {rows[1]['MAE']:.9f}, {rows[2]['MAE']:.9f}. Son referencias recuperadas, no Fase V. Persistencia3,560039; IC histórico mejor Huber−persistencia=[−0,022115;0,007603]: sin superioridad demostrada.

## Runtime y artefactos
Notebook maestro: https://colab.research.google.com/drive/1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss . Drive y carpeta canónica montados/verificados. Tesla T4, PyTorch2.14.1+cu130, NumPy2.4.6; variables nuevas M7_ (no asumir que M6_ esté cargado). No asumir sesión viva al reanudar.
Canónica: /content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571/
Nueva evidencia/predicciones paso0: {final['output']}/
SHA256 final_checksums.json: {final['manifest_sha256']}
SHA256 manifiestoIV: e9e0e3754deaafb0153d11dfce2bf0abd024669d2a253f2f7768c8a6bfa7effc
Informe, diff, hashes y resultados locales: m06_staging/m07_reconciliation_20261120T224952Z/
Respaldo final: {p.relative_to(ROOT).as_posix()}

## Pendiente y próxima acción
Fase V:0/6 corridas iniciadas en esta misión; no comenzar por restricción vigente. Completar deduplicación por huella científica (inventario de12 metadatos no basta), restauración integral repetida en namespace nuevo, equivalencia RNG/optimizador y comprobación posterior LoRA_A. Motor de Fase V/persistencia aún no implementado ni validado. No declarar preflight total aprobado.
Protocolo: m06_staging/audit_20261120/PROTOCOLO_FASE_V_JOINT_PROPUESTO_20261120.json . Dos brazos×tres semillas,300pasos,micro16×4,AdamW sin weight_decay; LR cabeza3e-4/LoRA1e-4; dev solo paso0/300.

## Historia y protección
No reentrenar I–IV ni sobrescribir checkpoints. Informes08-11 y evaluación11-11 (5.065 ventanas,L64,2020–2022) son históricos y no comparables directamente con M06. Test2023–2024 sellado. Estados anteriores preservados en las carpetas de auditoría. Un solo escritor del notebook.
''',encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={str(x.relative_to(ROOT)):{'sha256':sha(x),'bytes':x.stat().st_size} for x in out.iterdir() if x.is_file()}
for x in [state,p,ROOT/'automation/m07_colab_session.py',ROOT/'automation/m07_integrity.py',ROOT/'automation/test_m07_integrity.py']:
    files[str(x.relative_to(ROOT))]={'sha256':sha(x),'bytes':x.stat().st_size}
with (out/'final_local_checksums.json').open('x',encoding='utf-8') as f:json.dump(files,f,indent=2)
for rel,value in files.items():assert sha(ROOT/rel)==value['sha256']
print('Reconciliation closed',out)
print('State sha256',sha(state))
