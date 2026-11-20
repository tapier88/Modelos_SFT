from pathlib import Path
import json,ast,hashlib
r=Path(__file__).resolve().parent
nb=json.loads((r/'stabilize_primary_backup.ipynb').read_text(encoding='utf-8'))
cells={c.get('id',c.get('metadata',{}).get('id')):c for c in nb['cells']}
data=''.join(cells['m06_datos_20261114']['source']).split("for rel in ['auditoria/ventanas_m06.parquet'")[0]
models=''.join(cells['m06_models_v1']['source']).split("m6_mode('Kronos_base'); probe=")[0]
models=models.replace("snapshot_download('NeoQuasar/'+repo, revision=rev, local_dir=str(M6_WEIGHTS/repo), allow_patterns=['config.json','model.safetensors'])", "\n    if not all((M6_WEIGHTS/repo/f).is_file() for f in ['config.json','model.safetensors']):\n        snapshot_download('NeoQuasar/'+repo, revision=rev, local_dir=str(M6_WEIGHTS/repo), allow_patterns=['config.json','model.safetensors'])")
bsq=''.join(cells['m06_bsq_v1']['source']).split('fig,axes=plt.subplots')[0]
# No publicar ni reescribir artefactos durante recuperación.
bsq='\n'.join(s for s in bsq.splitlines() if '.to_csv(' not in s and '.to_parquet(' not in s)
assert "if cache.exists():" in bsq
bsq=bsq.replace("cache=audit_dir/f'batch_{start:04d}.npz'", "cache=audit_dir/f'batch_{start:04d}.npz'\n    assert cache.is_file(), 'Recuperación prohíbe recalcular cachés faltantes'")
header=(r/'stable_runtime_core.py').read_text(encoding='utf-8')
source=header+'\nDATA_SOURCE='+repr(data)+'\nMODEL_SOURCE='+repr(models)+'\nPHASE_I_SOURCE='+repr(bsq)+'\n'
for key,text in [('data',data),('models',models),('phaseI',bsq)]: ast.parse(text)
ast.parse(source)
dest=r/'m06_recovery_stable_v2.py';dest.write_text(source,encoding='utf-8')
print(json.dumps({'path':str(dest),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}))
