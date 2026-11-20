import json,hashlib,re,math
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parent
IV=ROOT/'fase_IV';manifest=json.loads((IV/'checksums.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
hashes=[]
for rel,exp in manifest.items():
    p=IV/rel;assert p.stat().st_size==exp['bytes'] and sha(p)==exp['sha256'],rel
    hashes.append({'file':rel,'sha256':sha(p),'bytes':p.stat().st_size,'verified':True})
status=json.loads((IV/'phase_status.json').read_text());assert status['complete'] and status['pilots']==5 and status['candidate_outputs']==6
spec=json.loads((IV/'configuraciones/protocolo_previo.json').read_text())
models=json.loads((ROOT/'auditoria/modelos.json').read_text())
table=pd.read_csv(IV/'metricas/pilotos_resultados_2020.csv')
assert len(table)==6 and table.seed.eq(271828).all() and table.steps.eq(100).all() and table.trainable_parameters.eq(147456).all()
baseline=pd.read_parquet(ROOT/'referencias_predicciones_2020.parquet')
base={n:g.pivot(index='window',columns='horizon',values='prediction').sort_index().to_numpy() for n,g in baseline.groupby('model')}
truth=baseline[baseline.model.eq('Persistencia')].pivot(index='window',columns='horizon',values='truth').sort_index().to_numpy()
last=base['Persistencia'][:,0]
ix=pd.read_csv(ROOT/'auditoria/desarrollo_2020.csv')
assert len(ix)==950 and ix.ny_date.nunique()==243 and pd.to_datetime(ix.decision_utc,utc=True).dt.year.eq(2020).all()
assert truth.shape==(950,4)
def stats(p):
    e=p-truth;mask=truth!=last[:,None]
    return {'MAE':float(abs(e).mean()),'RMSE':float(np.sqrt(np.square(e).mean())),'bias':float(e.mean()),
      'direction_pct':float(100*(np.sign(p-last[:,None])[mask]==np.sign(truth-last[:,None])[mask]).mean())}
adapters=[];preds={};metric_checks=[]
for v in spec['variants']:
    name='Pilot_'+v['id']+'_s271828';folder=IV/'checkpoints'/name
    # File SHA/size validated above BEFORE safe tensor-only deserialization.
    pkt=torch.load(folder/'adapter_final.pt',map_location='cpu',weights_only=True)
    assert pkt['variant']==v and pkt['step']==100 and pkt['seed']==271828
    assert pkt['base_hash']==models['base_hash'] and pkt['tokenizer_hash']==models['tokenizer_hash']
    expected={}
    for module,rank in v['ranks'].items():
        if '.ffn.' in module:
            assert re.fullmatch(r'transformer\.[0-7]\.ffn\.w[123]',module)
            inp,out=(1024,512) if module.endswith('.w2') else (512,1024)
        else:
            assert re.fullmatch(r'(transformer\.[0-7]\.self_attn|dep_layer\.cross_attn)\.[qv]_proj',module)
            inp,out=512,512
        expected[module+'.lora_A']=(rank,inp);expected[module+'.lora_B']=(out,rank)
    assert set(expected)==set(pkt['adapter'])
    for n,t in pkt['adapter'].items():
        assert tuple(t.shape)==expected[n] and t.dtype==torch.float32 and torch.isfinite(t).all(),n
    count=sum(t.numel() for t in pkt['adapter'].values());assert count==147456
    resume=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
    assert resume['step']==100 and resume['variant']==v
    assert all(torch.equal(t,resume['adapter'][n]) for n,t in pkt['adapter'].items())
    assert set(resume['adapter'])==set(pkt['adapter'])
    assert all(float(s['step'])==100 and torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all() for s in resume['optimizer']['state'].values())
    assert all(k in resume for k in ['numpy_rng','torch_rng','cuda_rng'])
    completion=json.loads((folder/'complete.json').read_text())
    assert completion['fixed_final_step']==100 and completion['base_unchanged']
    with np.load(folder/'evaluation.npz',allow_pickle=False) as z:
        assert np.array_equal(z['ids'],np.arange(950)) and z['CE'].shape==(950,4,2)
        assert all(np.isfinite(z[k]).all() for k in z.files)
        for row in completion['rows']:
            label=row['model'];f=pd.read_parquet(IV/'metricas'/(label+'.parquet'))
            assert len(f)==3800 and not f[['window','horizon']].duplicated().any()
            assert pd.to_datetime(f.decision_utc,utc=True).dt.year.eq(2020).all() and f.ny_date.nunique()==243
            y=f.pivot(index='window',columns='horizon',values='truth').sort_index().to_numpy()
            p=f.pivot(index='window',columns='horizon',values='prediction').sort_index().to_numpy()
            assert np.array_equal(y,truth) and np.array_equal(p,z['continuous' if label.endswith('_continuous') else 'greedy'])
            result=stats(p);stored=table.set_index('model').loc[label]
            for k,value in result.items():assert np.isclose(value,stored[k],rtol=0,atol=1e-10),(label,k)
            assert abs(float(z['CE'].mean())-stored.CE)<1e-6
            preds[label]=p;metric_checks.append({'model':label,**result,'cache_parquet_identical':True})
    adapters.append({'model':name,'sha256':sha(folder/'adapter_final.pt'),'size':(folder/'adapter_final.pt').stat().st_size,
        'tensor_count':len(expected),'parameters':count,'architecture_compatible':True,'resume_identical':True,
        'step':100,'seed':271828,'shapes':{k:list(v) for k,v in expected.items()},'base_hash':pkt['base_hash'],'tokenizer_hash':pkt['tokenizer_hash']})
def paired(p,ref):
    delta=abs(p-truth)-abs(ref-truth)
    g=pd.DataFrame({'session':ix.ny_date,'sum':delta.sum(1),'n':4}).groupby('session',sort=True).sum()
    vals=g['sum'].to_numpy();cnt=g.n.to_numpy();n=len(g);rng=np.random.default_rng(271828);draw=[]
    for _ in range(2000):
        take=((rng.integers(0,n,size=math.ceil(n/5))[:,None]+np.arange(5))%n).ravel()[:n]
        draw.append(vals[take].sum()/cnt[take].sum())
    lo,hi=np.quantile(draw,[.025,.975]);return {'delta_MAE':float(delta.mean()),'CI95_low':float(lo),'CI95_high':float(hi)}
cis=pd.read_csv(IV/'metricas/pilotos_intervalos.csv')
for row in cis.to_dict(orient='records'):
    calc=paired(preds[row['candidate']],base[row['reference']])
    assert all(abs(calc[k]-row[k])<1e-10 for k in calc)
nb=json.loads((ROOT/'notebook_exact.ipynb').read_text(encoding='utf-8'))
assert len(nb['cells'])==55
cell=next(c for c in nb['cells'] if c.get('metadata',{}).get('id')=='qtn1qIqmtknc')
outputs=''.join(''.join(o.get('text',[])) for o in cell.get('outputs',[]))
assert 'FASE_IV_COMPLETA 6 salidas' in outputs
displayed='\n'.join(''.join(o.get('data',{}).get('text/plain',[])) for o in cell.get('outputs',[]))
for row in table.itertuples():assert f'{row.MAE:.6f}' in displayed
for row in table[~table.model.str.endswith('_continuous')].itertuples():
    match=re.search(r'PILOTO '+re.escape(row.model)+r' paso 100 CE ([\d.]+) MAE ([\d.]+) n 950',outputs)
    assert match and abs(float(match[2])-row.MAE)<1e-10
original=json.loads((ROOT.parent/'notebook_original.ipynb').read_text(encoding='utf-8'))
history=[]
for i,(a,b) in enumerate(zip(original['cells'][:40],nb['cells'][:40])):
    history.append({'cell':i,'id_equal':a.get('metadata',{}).get('id')==b.get('metadata',{}).get('id'),
       'source_equal':a.get('source')==b.get('source'),'outputs_equal':a.get('outputs',[])==b.get('outputs',[]),
       'metadata_equal':a.get('metadata',{})==b.get('metadata',{}),'execution_count_equal':a.get('execution_count')==b.get('execution_count')})
assert all(x['id_equal'] and x['source_equal'] for x in history)
# Output differences are findings to preserve, never silently overwrite them.
third=ROOT/'fase_III';m3=json.loads((third/'checksums.json').read_text());verified3=[]
for p in sorted(third.rglob('*')):
    if not p.is_file() or p.name=='checksums.json':continue
    rel=p.relative_to(third).as_posix();assert rel in m3 and sha(p)==m3[rel]['sha256'] and p.stat().st_size==m3[rel]['bytes']
    verified3.append(rel)
heads=pd.read_csv(third/'metricas/cabezas_resultados_2020.csv');head_checks=[]
for row in heads[heads.model.str.startswith('Head_Huber')].itertuples():
    f=pd.read_parquet(third/'metricas'/(row.model+'.parquet'))
    p=f.pivot(index='window',columns='horizon',values='prediction').sort_index().to_numpy()
    assert np.array_equal(f.pivot(index='window',columns='horizon',values='truth').sort_index().to_numpy(),truth)
    assert abs(stats(p)['MAE']-row.MAE)<1e-10
    head_checks.append({'model':row.model,'best_step':int(row.best_step),**stats(p),'CI_vs_persistence':paired(p,base['Persistencia'])})
    ck=third/'checkpoints/fase_III/7587400db64558e6'/(row.model+'_best.pt')
    pkt=torch.load(ck,map_location='cpu',weights_only=True)
    assert pkt['base_hash']==models['base_hash'] and pkt['tokenizer_hash']==models['tokenizer_hash']
    assert pkt['seed']==int(row.seed) and pkt['best_step']==int(row.best_step)
    shapes={'hidden.weight':(64,512),'hidden.bias':(64,),'output.weight':(4,64),'output.bias':(4,)}
    assert set(pkt['head'])==set(shapes)
    for n,t in pkt['head'].items():assert tuple(t.shape)==shapes[n] and torch.isfinite(t).all()
    head_checks[-1].update(checkpoint_sha256=sha(ck),parameters=sum(t.numel() for t in pkt['head'].values()),architecture_compatible=True)
comparison={'phase_IV_files_verified':len(hashes),'phase_IV_manifest_sha256':sha(IV/'checksums.json'),'phase_status':status,
 'adapters':adapters,'metrics_recomputed':metric_checks,'paired_intervals_recomputed':len(cis),'head_benchmarks':head_checks,
 'phase_III_files_verified':verified3,'notebook_cells':55,'notebook_sha256':sha(ROOT/'notebook_exact.ipynb'),
 'phase_IV_cell_id':'qtn1qIqmtknc','phase_IV_output_matches':True,'historical40':history,
 'runtime_live_verified':False,'new_training_executed':False,'reserved_data_read':False,'orders_enabled':False}
(ROOT/'audit_evidence.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
pd.DataFrame(hashes).to_csv(ROOT/'hashes_verified.csv',index=False)
pd.DataFrame(metric_checks).to_csv(ROOT/'metrics_recomputed.csv',index=False)
print(json.dumps({k:v for k,v in comparison.items() if k not in ['adapters','historical40','phase_III_files_verified']},indent=2))
print('HISTORICAL40', {k:sum(x[k] for x in history) for k in history[0] if k!='cell'})
