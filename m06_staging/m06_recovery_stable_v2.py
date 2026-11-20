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

DATA_SOURCE="#@title MISIÓN 06 · 1. Datos autorizados, índice nuevo y normalización causal\nassert globals().get('M6_PREFLIGHT_OK') is True, 'Ejecutar primero M06 · 0'\nimport numpy as np\nimport pandas as pd\n\nM6_SOURCE = M6_BASE / 'datasets/processed/kronos_ohlcv_v1'\nM6_SOURCE_MANIFEST, m6_parts = [], []\n# Lista cerrada; no se enumeran directorios de datos, archivos ZIP ni índices históricos.\ndef m6_authorized_source(year):\n    if type(year) is not int or year not in range(2010, 2021):\n        raise ValueError('M06 solo permite archivos fuente 2010–2020')\n    return M6_SOURCE / f'XAUUSD_M15_BID_{year}.parquet'\n\nfor m6_year in range(2010, 2021):\n    m6_path = m6_authorized_source(m6_year)\n    m6_digest = m6_sha(m6_path)\n    m6_part = pd.read_parquet(m6_path)\n    m6_part['timestamp_utc'] = pd.to_datetime(m6_part.timestamp_utc, utc=True)\n    assert len(m6_part) and m6_part.timestamp_utc.dt.year.eq(m6_year).all()\n    assert m6_part.timestamp_utc.is_monotonic_increasing and m6_part.timestamp_utc.is_unique\n    assert m6_part.m1_candles.eq(15).all()\n    m6_prices = m6_part[['open', 'high', 'low', 'close', 'volume']]\n    assert np.isfinite(m6_prices.to_numpy()).all()\n    assert (m6_part.low > 0).all() and (m6_part.volume >= 0).all()\n    assert (m6_part.low <= m6_part[['open', 'close']].min(axis=1)).all()\n    assert (m6_part.high >= m6_part[['open', 'close']].max(axis=1)).all()\n    assert (m6_part.low <= m6_part.high).all()\n    assert m6_sha(m6_path) == m6_digest, 'Fuente cambió durante lectura'\n    M6_SOURCE_MANIFEST.append({'year': m6_year, 'file': m6_path.name, 'rows': len(m6_part), 'sha256': m6_digest})\n    m6_parts.append(m6_part[['timestamp_utc', 'open', 'high', 'low', 'close', 'volume']].copy())\nm6_df = pd.concat(m6_parts, ignore_index=True)\ndel m6_parts, m6_part, m6_prices\nassert m6_df.timestamp_utc.is_unique and m6_df.timestamp_utc.is_monotonic_increasing\nm6_ts = m6_df.timestamp_utc.astype('int64').to_numpy()\nM6_STEP = 15 * 60 * 10**9\nassert np.all(m6_ts % M6_STEP == 0)\nm6_df['split'] = np.where(m6_df.timestamp_utc.dt.year <= 2019, 'train', 'development')\nm6_df['segment'] = (m6_df.timestamp_utc.diff().ne(pd.Timedelta(minutes=15)) |\n                    m6_df.split.ne(m6_df.split.shift())).cumsum()\nm6_windows = []\nfor m6_segment, m6_group in m6_df.groupby('segment', sort=False):\n    m6_first, m6_last = int(m6_group.index[0]), int(m6_group.index[-1])\n    m6_starts = np.arange(m6_first, m6_last - 59 + 1, dtype=np.int64)\n    if not len(m6_starts):\n        continue\n    m6_decisions = m6_df.timestamp_utc.iloc[m6_starts + 56]\n    m6_ny = m6_decisions.dt.tz_convert('America/New_York')\n    for s, decision, ny in zip(m6_starts, m6_decisions, m6_ny):\n        if ny.strftime('%H:%M') not in M6_POLICY['ny_anchors']:\n            continue\n        m6_windows.append({'start_row': int(s), 'split': m6_group.split.iloc[0],\n                           'decision_utc': decision, 'decision_ny': ny.isoformat(),\n                           'ny_date': ny.strftime('%Y-%m-%d'), 'ny_hour': ny.strftime('%H:%M'),\n                           'context_start': m6_df.timestamp_utc.iloc[s],\n                           'target_end': m6_df.timestamp_utc.iloc[s+59] + pd.Timedelta(minutes=15)})\nm6_ix = pd.DataFrame(m6_windows).sort_values('decision_utc').reset_index(drop=True)\nassert len(m6_ix) and m6_ix.decision_utc.is_unique\nm6_train_ix = m6_ix[m6_ix.split.eq('train')].reset_index(drop=True)\nm6_dev_ix = m6_ix[m6_ix.split.eq('development')].reset_index(drop=True)\nassert len(m6_train_ix) and len(m6_dev_ix)\nassert m6_dev_ix.decision_utc.dt.year.eq(2020).all()\nassert m6_train_ix.target_end.max() < m6_dev_ix.context_start.min()\nassert (m6_ix.decision_utc.iloc[1:].to_numpy() >= m6_ix.target_end.iloc[:-1].to_numpy()).all()\nm6_starts = m6_ix.start_row.to_numpy(np.int64)\nassert np.all(m6_ts[m6_starts[:, None] + np.arange(60)] == m6_ts[m6_starts, None] + np.arange(60) * M6_STEP)\n\ndef m6_normalize(raw):\n    raw = np.asarray(raw, dtype=np.float32)\n    if raw.shape != (60, 6):\n        raise ValueError('Contrato M06: 60 filas, OHLCV + amount=0')\n    mean = raw[:56].mean(axis=0)\n    scale = raw[:56].std(axis=0) + np.float32(1e-5)\n    return np.clip((raw - mean) / scale, -5, 5), mean, scale\n\nclass M6Dataset(torch.utils.data.Dataset):\n    def __init__(self, frame, index):\n        self.L, self.H = 56, 4\n        self.index = index.reset_index(drop=True)\n        self.starts = self.index.start_row.to_numpy(np.int64)\n        self.values = np.column_stack([frame[['open','high','low','close','volume']].to_numpy(np.float32), np.zeros(len(frame), np.float32)])\n        t = frame.timestamp_utc.dt\n        self.stamp = np.column_stack([t.minute, t.hour, t.weekday, t.day, t.month]).astype(np.float32)\n    def __len__(self):\n        return len(self.starts)\n    def __getitem__(self, i):\n        s = int(self.starts[i])\n        raw = self.values[s:s+60].copy()\n        x, mean, scale = m6_normalize(raw)\n        return {'x': torch.from_numpy(x), 'stamp': torch.from_numpy(self.stamp[s:s+60].copy()),\n                'loss_mask': torch.from_numpy(np.arange(59) >= 55),\n                'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale),\n                'truth': torch.from_numpy(raw[56:].copy()), 'last_close': torch.tensor(raw[55,3])}\n\nm6_train = M6Dataset(m6_df, m6_train_ix)\nm6_dev = M6Dataset(m6_df, m6_dev_ix)\nm6_probe = m6_train.values[m6_train.starts[0]:m6_train.starts[0]+60].copy()\nm6_changed = m6_probe.copy()\nm6_changed[56:] += np.float32(10000)\nm6_before, m6_after = m6_normalize(m6_probe), m6_normalize(m6_changed)\nassert np.array_equal(m6_before[0][:56], m6_after[0][:56])\nassert np.array_equal(m6_before[1], m6_after[1]) and np.array_equal(m6_before[2], m6_after[2])\nM6_DATA_CHECKS = {'loaded_years': list(range(2010, 2021)), 'normalization_future_invariant': True,\n                  'train_windows': len(m6_train), 'development_windows': len(m6_dev),\n                  'development_ny_sessions': int(m6_dev_ix.ny_date.nunique()),\n                  'causal_model_checks': 'PENDING', 'predictive_experiments': 'NOT_EXECUTED'}\n"
MODEL_SOURCE="#@title MISIÓN 06 · 2. Modelos congelados, causalidad y protocolo experimental\nassert globals().get('M6_PREFLIGHT_OK') and 'm6_dev' in globals()\nimport os, sys, time, math, random, copy, inspect, importlib.util\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.data import default_collate\nfrom huggingface_hub import snapshot_download\nimport matplotlib.pyplot as plt\nos.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'\ntorch.use_deterministic_algorithms(True)\ntorch.backends.cuda.matmul.allow_tf32 = False\ntorch.backends.cudnn.allow_tf32 = False\ntorch.set_num_threads(min(2, os.cpu_count() or 1))\n\ndef m6_save_json(rel, obj):\n    path = (M6_DIR / rel).resolve()\n    assert M6_DIR.resolve() in path.parents\n    path.parent.mkdir(parents=True, exist_ok=True)\n    tmp = path.with_suffix(path.suffix + '.tmp')\n    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding='utf-8')\n    os.replace(tmp, path)\n\ndef m6_seed(seed=271828):\n    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)\n    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)\n\ndef m6_hash(net, base=False):\n    h = hashlib.sha256()\n    for name, value in sorted(net.state_dict().items()):\n        if base and ('lora_A' in name or 'lora_B' in name): continue\n        name = name.replace('.base.', '.') if base else name\n        h.update(name.encode()); h.update(str(tuple(value.shape)).encode())\n        h.update(value.detach().cpu().contiguous().numpy().tobytes())\n    return h.hexdigest()\n\nassert m6_sha(M6_HIST/'kronos_pipeline.py') == M6_PROVENANCE[0]['sha256']\nspec = importlib.util.spec_from_file_location('m6_kp', M6_HIST/'kronos_pipeline.py')\nm6_kp = importlib.util.module_from_spec(spec); spec.loader.exec_module(m6_kp)\nassert m6_kp.COMMIT == '67b630e67f6a18c9e9be918d9b4337c960db1e9a'\nM6_WEIGHTS = Path('/content/kronos_weights_pinned')\nfor repo, rev in m6_kp.REVISIONS.items():\n    \n    if not all((M6_WEIGHTS/repo/f).is_file() for f in ['config.json','model.safetensors']):\n        snapshot_download('NeoQuasar/'+repo, revision=rev, local_dir=str(M6_WEIGHTS/repo), allow_patterns=['config.json','model.safetensors'])\nm6_model, m6_tok = m6_kp.load_models(M6_WEIGHTS, M6_DEVICE)\nm6_model.requires_grad_(False).eval(); m6_tok.requires_grad_(False).eval()\nassert len(m6_model.transformer) == 8 and m6_model.d_model == 512\nassert (m6_tok.s1_bits, m6_tok.s2_bits) == (10, 10)\nM6_BASE_HASH, M6_TOK_HASH = m6_hash(m6_model), m6_hash(m6_tok)\nassert m6_sha(M6_CHECKPOINT) == M6_PROVENANCE[-1]['sha256']\nm6_packet = torch.load(M6_CHECKPOINT, map_location='cpu', weights_only=False)\nassert m6_packet['base_hash'] == M6_BASE_HASH and m6_packet['tokenizer_hash'] == M6_TOK_HASH\nassert m6_packet['step'] == 1000\nM6_ADAPTER_M05 = {k:v.clone() for k,v in m6_packet['adapter'].items()}\nM6_M05_CONFIG = m6_packet['config']\ndel m6_packet\n\nclass M6LoRA(nn.Module):\n    def __init__(self, base):\n        super().__init__(); self.base = base.requires_grad_(False); self.enabled = True\n        self.lora_A = nn.Parameter(torch.zeros(8, base.in_features, device=base.weight.device), requires_grad=False)\n        self.lora_B = nn.Parameter(torch.zeros(base.out_features, 8, device=base.weight.device), requires_grad=False)\n        self.dropout = nn.Dropout(.10); self.scale = 2.\n    def forward(self, x):\n        y = self.base(x)\n        return y + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)*self.scale if self.enabled else y\n\nM6_TARGETS = []\nfor name, module in list(m6_model.named_modules()):\n    if type(module).__name__ in ['MultiHeadAttentionWithRoPE','MultiHeadCrossAttentionWithRoPE']:\n        for leaf in ['q_proj','v_proj']:\n            layer = getattr(module, leaf)\n            M6_TARGETS.append(name+'.'+leaf); setattr(module, leaf, M6LoRA(layer))\nassert len(M6_TARGETS) == 18\nM6_ADAPTER_PARAMS = {n:p for n,p in m6_model.named_parameters() if 'lora_' in n}\nassert set(M6_ADAPTER_PARAMS) == set(M6_ADAPTER_M05)\nassert sum(p.numel() for p in M6_ADAPTER_PARAMS.values()) == 147456\nwith torch.no_grad():\n    for name, param in M6_ADAPTER_PARAMS.items(): param.copy_(M6_ADAPTER_M05[name].to(param.device))\n\ndef m6_mode(label):\n    assert label in ['Kronos_base','Kronos_LoRA_M05']\n    for name in M6_TARGETS: m6_model.get_submodule(name).enabled = label == 'Kronos_LoRA_M05'\n    m6_model.eval(); m6_tok.eval()\n\ndef m6_batch(ds, ids):\n    return {k:v.to(M6_DEVICE) for k,v in default_collate([ds[int(i)] for i in ids]).items()}\n\ndef m6_sync():\n    if M6_DEVICE.type == 'cuda': torch.cuda.synchronize()\n\nM6_CAUSAL = {}\nfor label in ['Kronos_base','Kronos_LoRA_M05']:\n    m6_mode(label)\n    M6_CAUSAL[label] = m6_kp.contract_tests(m6_model, m6_tok, m6_train, M6_DEVICE)\nassert m6_hash(m6_model, True) == M6_BASE_HASH and m6_hash(m6_tok) == M6_TOK_HASH\nassert not any(p.requires_grad for p in m6_model.parameters())\n\n@torch.no_grad()\ndef m6_rollout(batch, paths=1, seed=271828, temperature=None, top_k=0, top_p=1.):\n    # El predictor y sus representaciones reciben solo contexto observado.\n    m6_model.eval(); m6_tok.eval()\n    a,b = m6_tok.encode(batch['x'][:,:56], half=True)\n    a=a.repeat_interleave(paths,0); b=b.repeat_interleave(paths,0)\n    stamp=batch['stamp'].repeat_interleave(paths,0)\n    generator=torch.Generator(device=M6_DEVICE).manual_seed(int(seed))\n    def choose(logits):\n        if temperature is None: return logits.argmax(-1, keepdim=True)\n        logits=logits/float(temperature)\n        if top_k:\n            logits=logits.masked_fill(logits < logits.topk(top_k).values[:,-1:], -float('inf'))\n        if top_p < 1:\n            values,order=logits.sort(descending=True)\n            remove=values.softmax(-1).cumsum(-1)>top_p\n            remove[:,1:]=remove[:,:-1].clone(); remove[:,0]=False\n            logits=logits.masked_fill(torch.zeros_like(remove).scatter(1,order,remove), -float('inf'))\n        return torch.multinomial(logits.softmax(-1),1,generator=generator)\n    for j in range(4):\n        logits, hidden=m6_model.decode_s1(a,b,stamp=stamp[:,:56+j])\n        na=choose(logits[:,-1])\n        nb=choose(m6_model.decode_s2(hidden,torch.cat([a[:,1:],na],1))[:,-1])\n        a=torch.cat([a,na],1); b=torch.cat([b,nb],1)\n    prediction=m6_tok.decode([a,b],half=True)[:,-4:]\n    prediction=prediction*batch['scale'].repeat_interleave(paths,0)[:,None]+batch['mean'].repeat_interleave(paths,0)[:,None]\n    return prediction.reshape(len(batch['x']),paths,4,6), (a.reshape(len(batch['x']),paths,60),b.reshape(len(batch['x']),paths,60))\n\n@torch.no_grad()\ndef m6_ce(batch):\n    a,b=m6_tok.encode(batch['x'],half=True)\n    l1,l2=m6_model(a[:,:-1],b[:,:-1],stamp=batch['stamp'][:,:-1],use_teacher_forcing=True,s1_targets=a[:,1:])\n    ce1=F.cross_entropy(l1.transpose(1,2),a[:,1:],reduction='none')[:,55:]\n    ce2=F.cross_entropy(l2.transpose(1,2),b[:,1:],reduction='none')[:,55:]\n    return torch.stack([ce1,ce2],-1)\n\n@torch.no_grad()\ndef m6_features(batch):\n    a,b=m6_tok.encode(batch['x'][:,:56],half=True)\n    return m6_model.decode_s1(a,b,stamp=batch['stamp'][:,:56])[1][:,-1]\n\n"
PHASE_I_SOURCE="#@title MISIÓN 06 · 3. Fase I: BSQ, clipping, intervenciones y referencias 2020\nassert globals().get('M6_READY') is True\nM6_LABELS=['Kronos_base','Kronos_LoRA_M05']\nM6_FIELDS=['open','high','low','close','volume','amount']\nM6_PRED={}; M6_CE={}; M6_TIMES={}\nM6_AUDIT_PARTS=[]\naudit_dir=M6_DIR/'checkpoints/fase_I'; audit_dir.mkdir(exist_ok=True)\n\n@torch.no_grad()\ndef m6_decode_latent(latent, pre=False):\n    z=m6_tok.post_quant_embed_pre(latent) if pre else m6_tok.post_quant_embed(latent)\n    for layer in m6_tok.decoder: z=layer(z)\n    return m6_tok.head(z)\n\nfor start in range(0,len(m6_dev),16):\n    cache=audit_dir/f'batch_{start:04d}.npz'\n    assert cache.is_file(), 'Recuperación prohíbe recalcular cachés faltantes'\n    if cache.exists():\n        with np.load(cache,allow_pickle=False) as cached: part={k:cached[k] for k in cached.files}\n        M6_AUDIT_PARTS.append(part); continue\n    batch=m6_batch(m6_dev,range(start,min(start+16,len(m6_dev))))\n    with torch.no_grad():\n        true_a,true_b=m6_tok.encode(batch['x'],half=True)\n        context_a,context_b=m6_tok.encode(batch['x'][:,:56],half=True)\n        assert torch.equal(true_a[:,:56],context_a) and torch.equal(true_b[:,:56],context_b)\n        z=m6_tok.embed(batch['x'])\n        for layer in m6_tok.encoder: z=layer(z)\n        continuous=F.normalize(m6_tok.quant_embed(z),dim=-1)\n        quantized=m6_tok.indices_to_bits([true_a,true_b],half=True)\n        def to_price(norm): return norm[:,-4:]*batch['scale'][:,None]+batch['mean'][:,None]\n        clip=to_price(batch['x'])\n        cont=to_price(m6_decode_latent(continuous))\n        oracle=to_price(m6_tok.decode([true_a,true_b],half=True))\n        pre=to_price(m6_decode_latent(quantized[:,:,:10],pre=True))\n        part={'truth':batch['truth'].cpu().numpy(),'last_close':batch['last_close'].cpu().numpy(),\n              'clipped':clip.cpu().numpy(),'continuous_bypass':cont.cpu().numpy(),\n              'oracle_tokens':oracle.cpu().numpy(),'oracle_s1_only':pre.cpu().numpy(),\n              'latent_quantization_mse':(quantized-continuous).square().mean((1,2)).cpu().numpy()}\n        for label in M6_LABELS:\n            m6_mode(label); m6_sync(); t0=time.perf_counter()\n            pred,(pa,pb)=m6_rollout(batch)\n            m6_sync(); elapsed=time.perf_counter()-t0\n            part[label]=pred[:,0].cpu().numpy()\n            part[label+'_seconds']=np.array([elapsed])\n            part[label+'_CE']=m6_ce(batch).cpu().numpy()\n            aa,bb=pa[:,0].clone(),pb[:,0].clone()\n            aa[:,56:]=true_a[:,56:]\n            part[label+'_real_s1_pred_s2']=to_price(m6_tok.decode([aa,bb],half=True)).cpu().numpy()\n            aa,bb=pa[:,0].clone(),pb[:,0].clone()\n            bb[:,56:]=true_b[:,56:]\n            part[label+'_pred_s1_real_s2']=to_price(m6_tok.decode([aa,bb],half=True)).cpu().numpy()\n            part[label+'_token_accuracy']=np.stack([(pa[:,0,56:]==true_a[:,56:]).cpu().numpy(),(pb[:,0,56:]==true_b[:,56:]).cpu().numpy()],-1)\n        assert all(np.isfinite(v).all() for v in part.values())\n    np.savez_compressed(cache,**part)\n    M6_AUDIT_PARTS.append(part)\n    if start%160==0: print('FASE_I',start,'/',len(m6_dev),flush=True)\n\nM6_AUDIT={k:np.concatenate([p[k] for p in M6_AUDIT_PARTS],axis=0) for k in M6_AUDIT_PARTS[0]}\nM6_Y=M6_AUDIT['truth'][:,:,3].astype(np.float64)\nM6_LAST=M6_AUDIT['last_close'].astype(np.float64)\nM6_PRED['Persistencia']=np.repeat(M6_LAST[:,None],4,axis=1)\nfor label in M6_LABELS:\n    M6_PRED[label]=M6_AUDIT[label][:,:,3].astype(np.float64)\n    M6_CE[label]=M6_AUDIT[label+'_CE']\n    M6_TIMES[label]=float(M6_AUDIT[label+'_seconds'].sum()/len(m6_dev))\n\ndef m6_metrics(pred,truth=M6_Y,last=M6_LAST):\n    error=np.asarray(pred,dtype=np.float64)-truth\n    change=pred-last[:,None]; actual=truth-last[:,None]\n    nonflat=actual!=0; signals=change!=0\n    return {'MAE':float(np.abs(error).mean()),'RMSE':float(np.sqrt(np.mean(error**2))),\n            'bias':float(error.mean()),'direction_pct':float(100*np.mean(np.sign(change)[nonflat]==np.sign(actual)[nonflat])) if signals.any() and nonflat.any() else None,\n            'direction_coverage_pct':float(100*signals.mean()),'actual_nonzero':int(nonflat.sum()),\n            'magnitude_MAE':float(np.abs(np.abs(change)-np.abs(actual)).mean()),'windows':len(truth)}\n\ndef m6_prediction_frame(label,pred,ids=None):\n    ids=np.arange(len(m6_dev)) if ids is None else np.asarray(ids)\n    ix=m6_dev_ix.iloc[ids]\n    return pd.DataFrame({'model':label,'window':np.repeat(ids,4),'horizon':np.tile(np.arange(1,5),len(ids)),\n                         'ny_date':np.repeat(ix.ny_date.to_numpy(),4),'ny_hour':np.repeat(ix.ny_hour.to_numpy(),4),\n                         'decision_utc':np.repeat(ix.decision_utc.astype(str).to_numpy(),4),\n                         'truth':M6_Y[ids].ravel(),'last_close':np.repeat(M6_LAST[ids],4),'prediction':pred.ravel()})\n\ndef m6_paired_ci(pred,reference,ids=None):\n    ids=np.arange(len(M6_Y)) if ids is None else np.asarray(ids)\n    delta=np.abs(pred-M6_Y[ids])-np.abs(reference-M6_Y[ids])\n    group=pd.DataFrame({'session':m6_dev_ix.iloc[ids].ny_date.to_numpy(),'delta_sum':delta.sum(1),'n':4}).groupby('session',sort=True).sum()\n    values=group.delta_sum.to_numpy(); counts=group.n.to_numpy(); n=len(group)\n    rng=np.random.default_rng(271828); draws=[]\n    for _ in range(2000):\n        starts=rng.integers(0,n,size=math.ceil(n/5))\n        take=((starts[:,None]+np.arange(5))%n).ravel()[:n]\n        draws.append(values[take].sum()/counts[take].sum())\n    lo,hi=np.quantile(draws,[.025,.975])\n    return {'delta_MAE':float(delta.mean()),'CI95_low':float(lo),'CI95_high':float(hi),'sessions':n,'bootstrap_replicates':2000,'block_sessions':5,'interpretation':'exploratorio; no corrige seleccion multiple'}\n\nM6_BASELINE_TABLE=pd.DataFrame([{'model':label,**m6_metrics(p),'seconds_per_window':M6_TIMES.get(label)} for label,p in M6_PRED.items()])\nreconstruction=[]\nfor name in ['clipped','continuous_bypass','oracle_tokens','oracle_s1_only']:\n    for j,field in enumerate(M6_FIELDS):\n        e=M6_AUDIT[name][:,:,j].astype(np.float64)-M6_AUDIT['truth'][:,:,j]\n        reconstruction.append({'intervention':name,'variable':field,'MAE':float(abs(e).mean()),'RMSE':float(np.sqrt((e**2).mean())),'scope':'ORACULO_DIAGNOSTICO_NO_PREDICTOR'})\nM6_RECON_TABLE=pd.DataFrame(reconstruction)\nfactorial=[]; decomposition=[]\nfor label in M6_LABELS:\n    e00=M6_AUDIT['oracle_tokens'][:,:,3]-M6_Y\n    e10=M6_AUDIT[label+'_pred_s1_real_s2'][:,:,3]-M6_Y\n    e01=M6_AUDIT[label+'_real_s1_pred_s2'][:,:,3]-M6_Y\n    e11=M6_PRED[label]-M6_Y\n    for metric,fun in [('MAE',lambda e:np.abs(e).mean()),('MSE',lambda e:np.square(e).mean())]:\n        E00,E10,E01,E11=map(fun,[e00,e10,e01,e11])\n        interaction=E11-E10-E01+E00\n        phi1=.5*((E10-E00)+(E11-E01)); phi2=.5*((E01-E00)+(E11-E10))\n        assert np.isclose(E00+phi1+phi2,E11)\n        factorial.append({'model':label,'metric':metric,'E00_real_real':E00,'E10_pred_real':E10,'E01_real_pred':E01,'E11_pred_pred':E11,'interaction':interaction,'shapley_s1':phi1,'shapley_s2':phi2})\n    stages={'clipping':M6_AUDIT['clipped'][:,:,3]-M6_Y,\n            'decoder_continuous':M6_AUDIT['continuous_bypass'][:,:,3]-M6_AUDIT['clipped'][:,:,3],\n            'quantization_intervention':M6_AUDIT['oracle_tokens'][:,:,3]-M6_AUDIT['continuous_bypass'][:,:,3],\n            'token_prediction':M6_PRED[label]-M6_AUDIT['oracle_tokens'][:,:,3]}\n    assert np.allclose(sum(stages.values()),e11)\n    terms=[]\n    for name,e in stages.items(): terms.append((name,float(np.mean(e**2))))\n    names=list(stages)\n    for i in range(len(names)):\n        for j in range(i+1,len(names)):\n            terms.append((names[i]+' x '+names[j],float(2*np.mean(stages[names[i]]*stages[names[j]]))))\n    assert np.isclose(sum(v for _,v in terms),np.mean(e11**2),rtol=1e-5,atol=1e-5)\n    for name,value in terms: decomposition.append({'model':label,'term':name,'MSE_contribution':value})\nM6_FACTORIAL=pd.DataFrame(factorial); M6_DECOMP=pd.DataFrame(decomposition)\ndegradation=M6_DECOMP.pivot(index='term',columns='model',values='MSE_contribution')\ndegradation['delta_MSE']=degradation.Kronos_LoRA_M05-degradation.Kronos_base\ntotal_delta=float(degradation.delta_MSE.sum())\ndegradation['signed_percent_of_delta']=100*degradation.delta_MSE/total_delta if abs(total_delta)>1e-10 else np.nan\n\ndef m6_figsave(name):\n    plt.tight_layout(); plt.savefig(M6_DIR/'graficas'/name,dpi=150,bbox_inches='tight'); plt.show(); plt.close()\n"
