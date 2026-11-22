# M07_REMOTE_INVENTORY_V2
"""Read-only artifact audit; run only after Drive authorization finishes."""
from pathlib import Path
import hashlib, json, os, datetime, subprocess

ROOT = Path('/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571')
assert ROOT.is_dir(), 'Drive/canonical run unavailable: do not train'
print('M07_CURRENT_ACTIVITY',json.dumps({'pid':os.getpid(),'processes':subprocess.run(['ps','-eo','pid,ppid,comm'],capture_output=True,text=True).stdout,'gpu_processes':subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True).stdout,'loaded_model_names':[k for k in globals() if k.startswith(('m6_','M6_'))]}),flush=True)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()
iv=ROOT/'fase_IV'
assert digest(iv/'checksums.json')=='e9e0e3754deaafb0153d11dfce2bf0abd024669d2a253f2f7768c8a6bfa7effc'
verified={}
for phase in ['fase_III','fase_IV']:
    directory=ROOT/phase
    manifest=json.loads((directory/'checksums.json').read_text())
    records=[]
    for rel, expected in manifest.items():
        target=(directory/rel).resolve()
        assert directory.resolve() in target.parents, 'Manifest path outside phase'
        assert target.is_file() and target.stat().st_size==expected['bytes'], rel
        assert digest(target)==expected['sha256'], rel
        records.append(rel)
    status=json.loads((directory/'phase_status.json').read_text())
    assert status['complete'] is True
    verified[phase]={'count':len(records),'manifest_sha256':digest(directory/'checksums.json'),'status':status,'files':records}
# Enumerate experiment metadata only; no dataset directories or contents.
candidates=[]
for entry in ROOT.parent.rglob('*'):
    if entry.is_file() and entry.suffix=='.json' and any(word in entry.name.lower() for word in ['manifest','registry','status','config','protocol','complete']):
        candidates.append({'path':str(entry),'bytes':entry.stat().st_size})
print('M07_REMOTE_INVENTORY',json.dumps({'verified':verified,'metadata_candidates':candidates,'deduplication_complete':False,'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'RESEARCH_ONLY':True,'test_read':False,'orders_enabled':False},sort_keys=True),flush=True)
