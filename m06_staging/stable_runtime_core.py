"""M06: recuperación transaccional, sin inferencia experimental ni entrenamiento."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,math,os,tempfile,uuid,contextlib,importlib.metadata

ROOT=Path('/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571')
OPS=ROOT/'auditoria/estabilizacion_v2'
LOCK=OPS/'runtime_writer.lock'

def digest_bytes(data): return hashlib.sha256(data).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def json_value(value,nonfinite='raise',issues=None,path='$'):
    import numpy as np
    import torch
    issues=[] if issues is None else issues
    if isinstance(value,Path): return str(value)
    if isinstance(value,np.generic): value=value.item()
    if isinstance(value,torch.Tensor): value=value.detach().cpu().tolist()
    if isinstance(value,np.ndarray): value=value.tolist()
    if isinstance(value,float) and not math.isfinite(value):
        if nonfinite!='null': raise ValueError('Non-finite value at '+path)
        issues.append({'path':path,'reason':'nonfinite','original':str(value)})
        return None
    if value is None or isinstance(value,(str,int,float,bool)): return value
    if isinstance(value,dict):
        if not all(isinstance(k,str) for k in value): raise TypeError('JSON object keys must be strings')
        return {k:json_value(v,nonfinite,issues,path+'.'+k) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_value(v,nonfinite,issues,path+'['+str(i)+']') for i,v in enumerate(value)]
    raise TypeError('Unsupported type at '+path+': '+type(value).__name__)

def atomic_bytes(path,data,expected_sha=None):
    path=Path(path).resolve()
    if ROOT.resolve() not in path.parents: raise ValueError('Write outside current M06 run forbidden')
    if path.suffix=='.ipynb' and 'auditoria' not in path.parts: raise ValueError('Notebook writes only as explicit backups')
    path.parent.mkdir(parents=True,exist_ok=True)
    lock=path.with_name(path.name+'.writer_lock')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    tmp=None
    try:
        os.close(fd)
        if expected_sha is not None and (not path.is_file() or sha(path)!=expected_sha):
            raise RuntimeError('CONCURRENT_MODIFICATION_DETECTED')
        tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
        with tmp.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        if sha(tmp)!=digest_bytes(data): raise IOError('Temporary file verification failed')
        os.replace(tmp,path)
        if path.stat().st_size!=len(data) or sha(path)!=digest_bytes(data): raise IOError('Drive readback verification failed')
        return {'path':str(path),'bytes':len(data),'sha256':sha(path)}
    finally:
        if tmp is not None and tmp.exists(): tmp.unlink()
        lock.unlink(missing_ok=True)

def atomic_json(path,value,nonfinite='raise',expected_sha=None):
    issues=[]; value=json_value(value,nonfinite,issues)
    data=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode()
    info=atomic_bytes(path,data,expected_sha)
    assert json.loads(Path(path).read_text())==value
    return {**info,'normalizations':issues}

def inventory():
    # Solo artefactos M06; nunca enumera ni abre datasets reservados.
    files=[]
    for folder in ['metricas','graficas','checkpoints/fase_I','checkpoints/fase_II']:
        p=ROOT/folder
        if p.is_dir(): files.extend(x for x in p.rglob('*') if x.is_file() and not x.name.endswith('.tmp'))
    return {str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)}

def create_or_verify_manifest():
    OPS.mkdir(parents=True,exist_ok=True)
    path=OPS/'artifact_manifest_before_recovery.json'
    current=inventory()
    if path.exists():
        previous=json.loads(path.read_text())
        if previous!=current: raise RuntimeError('ARTIFACT_MANIFEST_CHANGED: inspect before accepting any new baseline')
    else: atomic_json(path,current)
    return current

def verify_npz(path,expected_rows):
    import numpy as np
    try:
        with np.load(path,allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        assert arrays['truth'].shape==(expected_rows,4,6)
        assert arrays['last_close'].shape==(expected_rows,)
        assert all(np.isfinite(v).all() for v in arrays.values())
        for label in ['Kronos_base','Kronos_LoRA_M05']:
            assert arrays[label].shape==(expected_rows,4,6)
            assert arrays[label+'_CE'].shape==(expected_rows,4,2)
        return arrays
    except Exception as exc: raise RuntimeError('INVALID_PHASE_I_CACHE: '+str(path)) from exc

def recover():
    """Build isolated namespace; publish no success flag until all dependencies validate."""
    import numpy as np,pandas as pd,torch,platform
    from google.colab import drive
    from IPython.display import display,Markdown
    if not Path('/content/drive/MyDrive').is_dir(): drive.mount('/content/drive')
    if not ROOT.is_dir(): raise FileNotFoundError('Persistent M06 run missing: '+str(ROOT))
    OPS.mkdir(parents=True,exist_ok=True)
    fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,json.dumps({'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat()}).encode());os.close(fd)
    try:
        manifest=create_or_verify_manifest()
        provenance=json.loads((ROOT/'auditoria/procedencia.json').read_text())
        assert provenance[0]['sha256']=='1424fb451942c7e1e1823808405f428283f9b1aacf7b34d08df7ca9edb503016'
        for p in provenance: assert Path(p['path']).is_file() and sha(p['path'])==p['sha256'], 'Original source/checkpoint mismatch'
        for start in range(0,950,16): verify_npz(ROOT/'checkpoints/fase_I'/f'batch_{start:04d}.npz',min(16,950-start))
        ns={'__name__':'m06_recovered_state','M6_PHASE_I_DONE':False,'M06_ENVIRONMENT_STABLE':False,
            'np':np,'pd':pd,'torch':torch,'Path':Path,'json':json,'hashlib':hashlib,'display':display,'Markdown':Markdown,
            'M6_BASE':ROOT.parents[1],'M6_DIR':ROOT,'M6_PROVENANCE':provenance,
            'M6_POLICY':json.loads((ROOT/'configuraciones/politica.json').read_text()),
            'M6_HIST':Path(provenance[0]['path']).parent,'M6_CHECKPOINT':Path(provenance[-1]['path']),
            'M6_PREFLIGHT_OK':True,'m6_sha':sha,'M6_DEVICE':torch.device('cuda' if torch.cuda.is_available() else 'cpu')}
        ns['M6_HW_RECOVERED']={'pid':os.getpid(),'python':platform.python_version(),'torch':torch.__version__,
            'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'free_gpu_bytes':torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None}
        assert ns['M6_POLICY']['allowed_source_years']==list(range(2010,2021))
        exec(compile(DATA_SOURCE,'m06_authorized_data.py','exec'),ns)
        assert ns['M6_SOURCE_MANIFEST']==json.loads((ROOT/'auditoria/datasets_autorizados.json').read_text())
        pd.testing.assert_frame_equal(ns['m6_ix'],pd.read_parquet(ROOT/'auditoria/ventanas_m06.parquet'))
        assert len(ns['m6_train'])==8428 and len(ns['m6_dev'])==950 and ns['m6_dev_ix'].ny_date.nunique()==243
        exec(compile(MODEL_SOURCE,'m06_frozen_models.py','exec'),ns)
        # Safe serializer replaces the legacy writer only in the recovered namespace.
        ns['m6_save_json']=lambda rel,obj:atomic_json(ROOT/rel,obj)
        ns['M6_PLAN']=json.loads((ROOT/'configuraciones/protocolo_experimentos.json').read_text())
        ns['M6_MC_DAYS']=ns['M6_PLAN']['mc_exploration_days']
        ns['M6_MC_IDS']=np.flatnonzero(ns['m6_dev_ix'].ny_date.isin(ns['M6_MC_DAYS'])).tolist()
        ns['M6_READY']=True
        exec(compile(PHASE_I_SOURCE,'m06_phase_I_cache_only.py','exec'),ns)
        saved=pd.read_csv(ROOT/'metricas/referencias_2020.csv').set_index('model')
        actual=ns['M6_BASELINE_TABLE'].set_index('model')
        for name in ['Persistencia','Kronos_base','Kronos_LoRA_M05']:
            for metric in ['MAE','RMSE','bias','magnitude_MAE']:
                assert np.isclose(actual.loc[name,metric],saved.loc[name,metric],rtol=0,atol=1e-10), (name,metric)
        # Fase II completed artifacts are read, never resampled during recovery.
        ns['M6_PHASE_II_DONE']=False
        mcpath=ROOT/'metricas/MC_replicas_2020.csv'
        if mcpath.is_file():
            table=pd.read_csv(mcpath)
            assert len(table)==6 and set(table.seed)=={271828,271829,271830}
            for row in table.to_dict(orient='records'):
                frame=pd.read_parquet(ROOT/'metricas'/(row['model']+'.parquet'))
                assert len(frame)==3800 and frame['window'].nunique()==950
                assert pd.to_datetime(frame.decision_utc,utc=True).dt.year.eq(2020).all()
                p=frame.pivot(index='window',columns='horizon',values='prediction').sort_index().to_numpy()
                assert p.shape==(950,4) and np.isfinite(p).all()
                assert abs(ns['m6_metrics'](p)['MAE']-row['MAE'])<1e-10
                ns['M6_PRED'][row['model']]=p
            ns['M6_MC_REPLICAS']=table;ns['M6_MC_REPLICA_ROWS']=table.to_dict(orient='records')
            ns['M6_MC_PILOT']=pd.read_csv(ROOT/'metricas/MC_exploracion.csv')
            ns['M6_MC_PILOT_ROWS']=ns['M6_MC_PILOT'].to_dict(orient='records')
            ns['M6_MC_SELECTED']=json.loads((ROOT/'configuraciones/MC_seleccion_fijada.json').read_text())
            ns['M6_CANDIDATE_META']=json.loads((ROOT/'configuraciones/candidatos.json').read_text())
            ns['M6_PHASE_II_DONE']=True
        assert ns['m6_hash'](ns['m6_model'],True)==ns['M6_BASE_HASH']
        assert ns['m6_hash'](ns['m6_tok'])==ns['M6_TOK_HASH']
        assert not any(p.requires_grad for p in ns['m6_model'].parameters())
        assert inventory()==manifest, 'Recovery modified persistent experimental artifacts'
        summary={'train_windows':8428,'dev_windows':950,'ny_sessions':243,'phase_I_caches':60,
         'base_hash':ns['M6_BASE_HASH'],'tokenizer_hash':ns['M6_TOK_HASH'],'M05_checkpoint_sha256':sha(ns['M6_CHECKPOINT']),
         'phase_I_metrics':json.loads(ns['M6_BASELINE_TABLE'].to_json(orient='records')),
         'phase_II_replicas':len(ns.get('M6_MC_REPLICAS',[])),
         'artifact_manifest_sha256':digest_bytes(json.dumps(manifest,sort_keys=True).encode()),
         'predictions_sha256':digest_bytes(b''.join(np.ascontiguousarray(ns['M6_PRED'][k]).tobytes() for k in sorted(ns['M6_PRED']))),
         'original_weights_unchanged':True,'experimental_inference_or_training_executed':False}
        atomic_json(OPS/'last_verified_recovery.json',{'summary':summary,'hardware':ns['M6_HW_RECOVERED']})
        ns['M6_RECOVERY_EVIDENCE']=summary
        ns['M6_PHASE_I_DONE']=True
        return ns
    finally: LOCK.unlink(missing_ok=True)

def operational_tests():
    import numpy as np, torch, gc
    OPS.mkdir(parents=True,exist_ok=True)
    results={}
    # B: no inherited M06 globals; recover() creates its own namespace.
    first=recover(); first_summary=first['M6_RECOVERY_EVIDENCE']
    results['B_empty_namespace_recovery']=first['M6_PHASE_I_DONE'] is True
    results['C_integrity']=first_summary['original_weights_unchanged'] and first_summary['phase_I_caches']==60
    del first;gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    second=recover()
    results['D_idempotence']=first_summary==second['M6_RECOVERY_EVIDENCE']
    # E: strict serialization + explicit missing-value policy + readback.
    obj={'int':np.int64(7),'array':np.array([1.,2.]),'tensor':torch.tensor([3.,4.]),'missing':float('nan')}
    probe=OPS/'persistence_probe.json'
    atomic_json(probe,{'sentinel':'intact'})
    before=sha(probe)
    try: atomic_json(probe,obj)
    except ValueError: pass
    else: raise AssertionError('NaN was not rejected')
    assert sha(probe)==before
    info=atomic_json(probe,obj,nonfinite='null')
    assert json.loads(probe.read_text())=={'int':7,'array':[1.,2.],'tensor':[3.,4.],'missing':None}
    for bad in [float('inf'),float('-inf'),object()]:
        old=sha(probe)
        try: atomic_json(probe,{'bad':bad})
        except (ValueError,TypeError): pass
        else: raise AssertionError('Unsupported/nonfinite object was not rejected')
        assert sha(probe)==old
    results['E_persistence_serialization']=True
    # Corrupted cache fixture is isolated; never alters a genuine phase cache.
    corrupt=OPS/'truncated_cache_test.npz';atomic_bytes(corrupt,b'PK\x03\x04truncated')
    try: verify_npz(corrupt,16)
    except RuntimeError: results['C_corrupt_cache_detected']=True
    else: raise AssertionError('Corrupt cache accepted')
    # F: emulate stale writer against isolated JSON, not the principal notebook.
    fixture=OPS/'concurrency_fixture.json'; atomic_json(fixture,{'version':1})
    old=sha(fixture);atomic_json(fixture,{'version':2})
    try: atomic_json(fixture,{'version':3},expected_sha=old)
    except RuntimeError as exc: assert str(exc)=='CONCURRENT_MODIFICATION_DETECTED'
    else: raise AssertionError('Stale writer accepted')
    assert json.loads(fixture.read_text())=={'version':2}
    atomic_bytes(fixture.with_name(fixture.name+'.writer_lock'),b'test_writer')
    try: atomic_json(fixture,{'version':4})
    except FileExistsError: pass
    else: raise AssertionError('Second writer accepted')
    finally: fixture.with_name(fixture.name+'.writer_lock').unlink()
    results['F_stale_and_concurrent_writer_detected']=True
    assert all(results.values())
    atomic_json(OPS/'tests_B_C_D_E_F.json',{'results':results,'summary':second['M6_RECOVERY_EVIDENCE'],
       'serialization_normalizations':info['normalizations'],'principal_notebook_mutated_by_tests':False,
       'scope':'F tests isolated files; browser single-writer and save/readback test A verified separately',
       'utc':datetime.now(timezone.utc).isoformat()})
    return second,results
