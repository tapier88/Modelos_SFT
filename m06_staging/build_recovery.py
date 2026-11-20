from pathlib import Path
import ast
r=Path(__file__).resolve().parent
pre='''#@title MISIÓN 06 · R. Recuperación verificada del runtime y cachés M06
import sys, os, json, hashlib, platform, importlib.metadata as m6_metadata
from pathlib import Path
from datetime import datetime, timezone
from IPython.display import display, Markdown
import torch
from google.colab import drive
M6_BASE=Path('/content/drive/MyDrive/AXXEL_CAPITAL')
M6_DIR=M6_BASE/'MISION06_ALINEACION/20261114T222455Z_32d5e571'
if not Path('/content/drive/MyDrive').exists(): drive.mount('/content/drive')
assert M6_DIR.is_dir(), 'Falta la ejecución M06 que se va a recuperar'
M6_DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert M6_DEVICE.type=='cuda', 'La recuperación requiere la GPU disponible para el presupuesto acordado'
def m6_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()
M6_POLICY=json.loads((M6_DIR/'configuraciones/politica.json').read_text())
M6_PROVENANCE=json.loads((M6_DIR/'auditoria/procedencia.json').read_text())
M6_HIST=Path(M6_PROVENANCE[0]['path']).parent
M6_CHECKPOINT=Path(M6_PROVENANCE[-1]['path'])
for item in M6_PROVENANCE: assert m6_sha(item['path'])==item['sha256']
M6_PREFLIGHT_OK=True
print('RECUPERANDO',str(M6_DIR),'GPU',torch.cuda.get_device_name(0),flush=True)
'''
data=(r/'m06_data.py').read_text(encoding='utf-8').split("for rel in ['auditoria/ventanas_m06.parquet'")[0]
data+='''\nassert M6_SOURCE_MANIFEST==json.loads((M6_DIR/'auditoria/datasets_autorizados.json').read_text())
pd.testing.assert_frame_equal(m6_ix,pd.read_parquet(M6_DIR/'auditoria/ventanas_m06.parquet'))
print('DATOS_RECUPERADOS_Y_HASHES_VERIFICADOS',len(m6_train),len(m6_dev),flush=True)
'''
models=(r/'m06_models.py').read_text(encoding='utf-8')
bsq=(r/'m06_bsq.py').read_text(encoding='utf-8').split('fig,axes=plt.subplots')[0]
checks='''
for start in range(0,len(m6_dev),16):
    assert (M6_DIR/'checkpoints/fase_I'/f'batch_{start:04d}.npz').is_file(), 'Falta caché; no recalcular silenciosamente'
'''
tail='''
M6_PHASE_I_DONE=True
M6_RECOVERY={'utc':datetime.now(timezone.utc).isoformat(),'device':torch.cuda.get_device_name(0),
 'cached_phase_I_windows':len(M6_Y),'original_files_verified':True,'reason':'Runtime perdió variables al reconectar; MC detenido por assert antes de cálculo',
 'recovered_baselines':M6_BASELINE_TABLE.to_dict(orient='records')}
m6_save_json('auditoria/recuperacion_runtime.json',M6_RECOVERY)
display(M6_BASELINE_TABLE)
print('RECUPERACION_COMPLETA; ejecutar únicamente M06 · 4 para continuar')
'''
src=pre+data+models+checks+bsq+tail
ast.parse(src)
(r/'m06_recovery.py').write_text(src,encoding='utf-8')
print('recovery source compiled',len(src))
