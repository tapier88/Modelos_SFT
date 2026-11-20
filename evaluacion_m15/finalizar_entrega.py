"""Independently verify saved predictions, render a chart and package evidence."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'resultados_20261111'
BUNDLE = Path(r'C:\workspace\kronos_sft_20261107')
SOURCE = Path(r'C:\workspace\kronos_audit_20261107\parquet')
status = json.loads((OUT/'status.json').read_text(encoding='utf-8'))
assert status['state'] == 'completed' and not status['training_done'] and not status['test_evaluated']
pred = pd.read_parquet(OUT/'predictions.parquet')
ix = pd.read_parquet(OUT/'selected_windows.parquet')
truth = pd.concat([pd.read_parquet(SOURCE/f'XAUUSD_M15_BID_{y}.parquet') for y in [2020, 2021, 2022]], ignore_index=True)
assert len(pred) == 4*len(ix) == 20260
assert pred.target_close_time.is_unique
assert pred.actual_close.notna().all() and np.isfinite(pred.predicted_close).all()
positions = np.repeat(ix.start_row.to_numpy()+64, 4) + np.tile(np.arange(4), len(ix))
last_positions = np.repeat(ix.start_row.to_numpy()+63, 4)
assert np.array_equal(pred.actual_close.to_numpy(), truth.close.iloc[positions].to_numpy())
assert np.array_equal(pred.persistence_close.to_numpy(), truth.close.iloc[last_positions].to_numpy())
times = pd.to_datetime(truth.timestamp_utc.iloc[positions], utc=True).reset_index(drop=True)+pd.Timedelta(minutes=15)
assert times.equals(pred.target_close_time.reset_index(drop=True))
summary = pd.read_csv(OUT/'summary.csv')
global_row = summary[summary.group == 'ALL'].iloc[0]
assert np.isclose(global_row.close_mae, np.abs(pred.predicted_close-pred.actual_close).mean())
assert np.isclose(global_row.persistence_close_mae, np.abs(pred.persistence_close-pred.actual_close).mean())
checks = {'prediction_rows': len(pred), 'target_times_unique': True, 'all_truth_matches_source': True,
          'all_persistence_matches_source': True, 'all_target_times_match_source': True, 'aggregate_mae_recomputed': True}
(OUT/'independent_verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')

shown = summary.set_index('group').loc[['ALL', 'NY', '2020_NY', '2021_NY', '2022_NY']]
labels = ['Total', 'Nueva York', 'NY 2020', 'NY 2021', 'NY 2022']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
x = np.arange(len(shown))
axes[0].bar(x-.19, shown.close_mae, .38, label='Kronos · configuración actual', color='#3366a1')
axes[0].bar(x+.19, shown.persistence_close_mae, .38, label='Último cierre', color='#b7c1cc')
axes[0].set_xticks(x, labels, rotation=15)
axes[0].set_ylabel('Error absoluto medio · unidades de precio')
axes[0].set_title('Predicción de las próximas 4 velas M15')
axes[0].set_ylim(0, 1.24 * max(shown.close_mae.max(), shown.persistence_close_mae.max()))
axes[0].legend(frameon=False, fontsize=9)
axes[0].grid(axis='y', alpha=.18)
axes[0].set_axisbelow(True)
delta = shown.delta_mae.to_numpy()
err = np.vstack([delta-shown.delta_ci95_low.to_numpy(), shown.delta_ci95_high.to_numpy()-delta])
axes[1].errorbar(delta, x, xerr=err, fmt='o', color='#3366a1', capsize=4)
axes[1].axvline(0, color='#6b7280', linestyle='--', linewidth=1)
axes[1].set_yticks(x, labels)
axes[1].invert_yaxis()
axes[1].set_xlabel('MAE Kronos − MAE último cierre\nNegativo favorece Kronos; positivo favorece último cierre')
axes[1].set_title('Diferencia pareada · IC 95% por días')
axes[1].grid(axis='x', alpha=.18)
fig.suptitle('XAUUSD · validación 2020–2022 · pesos sin entrenamiento', fontsize=14, fontweight='bold')
fig.savefig(OUT/'comparacion_m15.png', dpi=180)
plt.close(fig)

diagnostics = json.loads((OUT/'input_diagnostics.json').read_text(encoding='utf-8'))
report = (OUT/'INFORME.md').read_text(encoding='utf-8').split('\n## Cobertura de Nueva York\n')[0]
report += '\n## Cobertura de Nueva York\n\n| Año | Días con velas NY | Días evaluados NY | Ventanas NY |\n|---|---:|---:|---:|\n'
for row in diagnostics:
    report += f"| {row['year']} | {row['ny_days_with_bars']} | {row['ny_days_with_selected_windows']} | {row['selected_ny_windows']} |\n"
report += '\n## Verificación adicional\n\nLas 20.260 predicciones guardadas se contrastaron de nuevo con los tres archivos fuente: '
report += 'precios reales, último cierre y tiempos de cierre coinciden; no hay objetivos temporales duplicados. '
report += 'Los hashes de ambos archivos de pesos coinciden con los publicados en las revisiones oficiales indicadas en provenance_verified.json.\n\n'
report += 'Las métricas auxiliares de magnitud comparan contra movimiento cero, una referencia débil: '
report += 'no demuestran superioridad frente a una estimación histórica de volatilidad o ATR. Las unidades '
report += 'de volumen no se contrastaron con las del preentrenamiento; amount permanece en cero.\n\n'
report += '![Comparación de error](comparacion_m15.png)\n'
opening = json.loads((OUT/'opening_coverage_audit.json').read_text(encoding='utf-8'))
report += '\n## Cobertura de la apertura: limitación decisiva\n\n'
report += f"La auditoría de todas las ventanas candidatas NY, antes de quitar solapamientos, encontró "
report += f"{opening['eligible_0830_to_1000_exclusive']} decisiones entre 08:30 y antes de 10:00. "
report += f"La primera hora elegible es {opening['earliest_ny_decision']}. "
report += 'Por tanto, este experimento no valida la apertura de las 08:30. '
report += 'El contexto estricto de 64 velas continuas, junto con las interrupciones diarias, retrasa la primera decisión elegible. '
report += f"Ejemplo: contexto desde {opening['first_window_context_start_ny']} y decisión {opening['first_window_decision_ny']}.\n\n"
report += 'La siguiente tarea debe resolver primero esta cobertura: definir por adelantado un contexto más corto '
report += 'que admita la apertura real, conservar las interrupciones explícitas y comparar sobre las mismas '
report += 'ventanas la inferencia oficial y el adaptador auditado. No rellenar precios inventados ni activar '
report += 'entrenamiento completo para compensar esta falta de cobertura.\n'
feasibility = json.loads((OUT/'opening_context_feasibility.json').read_text(encoding='utf-8'))
feasible = pd.DataFrame(feasibility['rows']).groupby('context_bars').eligible_exact_0830_decisions.sum()
report += '\n| Contexto M15 | Horas de contexto | Días con decisión elegible exacta a las 08:30 |\n|---|---:|---:|\n'
for context in [64, 58, 56, 48]:
    report += f'| {context} velas | {context/4:g} | {int(feasible.loc[context])} |\n'
report += '\nEste recuento es una comprobación estructural: no se hicieron predicciones con estos contextos alternativos '
report += 'ni se eligió una longitud según sus errores. Un piloto predefinido de 56 velas conservaría 14 horas '
report += 'y habilitaría la apertura en 724 días de la muestra; su capacidad predictiva sigue sin evaluar.\n'
report += 'La próxima selección debe anclarse explícitamente a 08:30, 09:30, 10:30 y 11:30 NY '
report += 'para mantener objetivos de una hora sin solapamiento. Reducir el contexto y conservar una '
report += 'selección global iniciada en otra hora no garantiza incluir la apertura.\n'
(OUT/'INFORME.md').write_text(report, encoding='utf-8')
ny_result = summary[summary.group == 'NY'].iloc[0]
current = f'''# Estado actual de Kronos — 2026-11-11

La tarea pendiente de evaluación M15 2020–2022 está completada en el equipo local (CPU).
La prueba corta anterior de Colab ya estaba terminada; no debe repetirse como siguiente tarea.

- 5.065 ventanas con objetivos no solapados; 20.260 cierres predichos.
- 2.179 ventanas NY en 751 días. Decisiones NY: 10:00–12:00; horizontes hasta las 13:00.
- MAE total: Kronos {global_row.close_mae:.6f}; último cierre {global_row.persistence_close_mae:.6f}.
- MAE NY: Kronos {ny_result.close_mae:.6f}; último cierre {ny_result.persistence_close_mae:.6f}.
- Mejora MAE total: {global_row.mae_skill_pct:.2f}%; NY: {ny_result.mae_skill_pct:.2f}%.
- RUN_FULL=False; ningún entrenamiento en esta evaluación; parámetros originales intactos.
- Test 2023–2024 sin leer ni evaluar en este experimento.
- Inferencia: adaptador causal auditado con greedy; no es el predictor oficial con muestreo.

Hallazgo para el siguiente paso: las 64 velas continuas de contexto dejan fuera la apertura de las 08:30.
Un contexto de 56 velas habilita estructuralmente esa hora en 724 días de validación;
todavía no se ha medido su capacidad predictiva.

La siguiente tarea es preparar y fijar una comparación para la apertura: contexto de 56 velas,
horizonte de 4 y decisiones ancladas a 08:30, 09:30, 10:30 y 11:30 America/New_York.
Usar las mismas ventanas para persistencia, inferencia oficial y adaptador auditado.
Separar por años y días, mantener el test reservado y no activar entrenamiento completo todavía.

Evidencia: evaluacion_m15/resultados_20261111/INFORME.md, status.json,
config.json, predictions.parquet, summary.csv, independent_verification.json.
Código: evaluacion_m15/evaluar_base_m15.py.
'''
(ROOT/'ESTADO_ACTUAL_KRONOS.md').write_text(current, encoding='utf-8')
(ROOT.parent/'ESTADO_ACTUAL_KRONOS.md').write_text(current, encoding='utf-8')

readme = '''# Evaluación M15 de Kronos, 2020–2022

Abra resultados/INFORME.md para ver conclusiones, alcance y limitaciones.
Esta ejecución se completó en Windows CPU, fuera de Colab, sin entrenamiento.
Los datos 2023–2024 no se abrieron ni puntuaron.

## Repetir en el equipo original

```powershell
python evaluar_base_m15.py --bundle C:\\workspace\\kronos_sft_20261107 --source C:\\workspace\\kronos_audit_20261107\\parquet --output resultados_nuevos
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
'''
(ROOT/'README_EJECUCION.md').write_text(readme, encoding='utf-8')
requirements = []
for name in ['numpy', 'pandas', 'pyarrow', 'torch', 'matplotlib', 'huggingface_hub', 'tqdm']:
    requirements.append(name+'=='+importlib.metadata.version(name))
requirements += ['einops==0.8.2', 'safetensors==0.6.2']
(ROOT/'requirements_entorno.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cpu\n'+'\n'.join(requirements)+'\n', encoding='utf-8')
archive = ROOT/'Kronos_Evaluacion_M15_2020_2022.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for name in ['evaluar_base_m15.py', 'test_metricas.py', 'finalizar_entrega.py', 'README_EJECUCION.md', 'requirements_entorno.txt', 'ESTADO_ACTUAL_KRONOS.md']:
        z.write(ROOT/name, name)
    for file in sorted(OUT.rglob('*')):
        if file.is_file() and file.suffix != '.tmp':
            z.write(file, str(Path('resultados')/file.relative_to(OUT)))
    z.write(BUNDLE/'kronos_pipeline.py', 'bundle/kronos_pipeline.py')
    for file in sorted((BUNDLE/'vendor').rglob('*')):
        if file.is_file() and (file.suffix == '.py' or file.name == 'LICENSE'):
            z.write(file, str(Path('bundle')/file.relative_to(BUNDLE)))
    for year in [2020, 2021, 2022]:
        name = f'XAUUSD_M15_BID_{year}.parquet'
        z.write(SOURCE/name, 'datos/'+name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                  'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'independent_checks': checks}))
