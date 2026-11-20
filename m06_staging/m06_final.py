# M06 final laboratory: all calculations use existing authorized caches/predictions.
import io,importlib.util,platform
from datetime import datetime,timezone
from IPython.display import display,Markdown
from google.colab import _message
assert M06_ENVIRONMENT_STABLE and M6_PHASE_I_DONE and M6_PHASE_II_DONE and M6_PHASE_III_DONE and M6_PHASE_IV_DONE
M6_FINAL_DIR=M6_DIR/'sintesis_final'
M6_FINAL_DIR.mkdir(exist_ok=True)
M6_FINAL_MANIFEST_PATH=M6_FINAL_DIR/'checksums.json'
M6_FINAL_MANIFEST=json.loads(M6_FINAL_MANIFEST_PATH.read_text()) if M6_FINAL_MANIFEST_PATH.exists() else {}
def final_verify():
    for rel,v in M6_FINAL_MANIFEST.items():
        p=M6_FINAL_DIR/rel
        assert p.is_file() and p.stat().st_size==v['bytes'] and m6_stable.sha(p)==v['sha256'],rel
final_verify()
def final_bytes(rel,data):
    p=M6_FINAL_DIR/rel;old=M6_FINAL_MANIFEST.get(rel,{}).get('sha256')
    assert not p.exists() or old is not None,'Untracked final artifact: '+rel
    mh=m6_stable.sha(M6_FINAL_MANIFEST_PATH) if M6_FINAL_MANIFEST_PATH.exists() else None
    info=m6_stable.atomic_bytes(p,data,expected_sha=old)
    M6_FINAL_MANIFEST[rel]={'bytes':info['bytes'],'sha256':info['sha256']}
    m6_stable.atomic_json(M6_FINAL_MANIFEST_PATH,M6_FINAL_MANIFEST,expected_sha=mh)
def final_json(rel,obj):
    issues=[];clean=m6_stable.json_value(obj,nonfinite='null',issues=issues)
    # Null is permitted only for unavailable table fields and is audited explicitly.
    final_bytes(rel,json.dumps({'data':clean,'nonfinite_to_null':issues},ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode())
def final_csv(rel,df):final_bytes(rel,df.to_csv(index=False).encode())
def final_fig(name):
    b=io.BytesIO();plt.tight_layout();plt.savefig(b,format='png',dpi=150,bbox_inches='tight')
    final_bytes('graficas/'+name,b.getvalue());plt.show();plt.close()
def md_table(df):
    def fmt(x):
        if x is None or (isinstance(x,(float,np.floating)) and not np.isfinite(x)):return 'N/A'
        if isinstance(x,(float,np.floating)):return f'{x:.6f}'
        return str(x).replace('|','/')
    return '| '+' | '.join(map(str,df.columns))+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))

# Restore from persistent files twice into fresh phase namespaces, without inference.
ep=Path('/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/m06_completed_recovery_5557a47b.py')
assert m6_stable.sha(ep)=='5557a47b0629d9ca7f3440e0f5d2b6e221c9ae2d5cdf7aaf0591527b0615d966'
spec=importlib.util.spec_from_file_location('m6_completed_recovery',ep)
extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
def recovery_probe():
    ns={k:globals()[k] for k in ['M6_DIR','m6_stable','M6_Y','M6_BASE_HASH','M6_TOK_HASH','M6_DEVICE','m6_metrics']}
    ns['M6_PRED']={};ns['M6_CANDIDATE_META']={}
    evidence=extension.restore_completed(ns)
    assert ns['M6_PHASE_III_DONE'] and ns['M6_PHASE_IV_DONE'] and len(ns['M6_PRED'])==14
    digest=hashlib.sha256()
    for n,p in sorted(ns['M6_PRED'].items()):
        assert np.array_equal(p,M6_PRED[n]);digest.update(n.encode());digest.update(p.tobytes())
    return {'phase_evidence':evidence,'prediction_sha256':digest.hexdigest(),'fresh_namespace':True}
probe1=recovery_probe();probe2=recovery_probe();assert probe1==probe2
final_json('auditoria/recuperacion_fases_III_IV.json',{'first':probe1,'second_identical':True,'retrained':False})

names=[n for n,p in M6_PRED.items() if not n.startswith('MC_DIAG_') and np.asarray(p).shape==(950,4)]
assert len(names)==23, (len(names),names)
truth=M6_Y;last=M6_LAST;days,session=np.unique(m6_dev_ix.ny_date.to_numpy(),return_inverse=True);nd=len(days)
assert nd==243
rng=np.random.default_rng(271828)
starts=rng.integers(0,nd,size=(2000,math.ceil(nd/5)))
draws=((starts[:,:,None]+np.arange(5))%nd).reshape(2000,-1)[:,:nd]
counts=np.bincount(session,minlength=nd)*4
def sums(v):return np.bincount(session,weights=np.asarray(v).sum(1),minlength=nd)
def bounds(v):return [float(x) for x in np.quantile(v,[.025,.975])]
def family(name):
    if name.startswith('MC_'):return 'MC_base' if 'Kronos_base' in name else 'MC_LoRA'
    if name.startswith('Head_'):return 'Head_Huber' if 'Huber' in name else 'Head_Laplace'
    return M6_CANDIDATE_META.get(name,{}).get('family',name)
M6_FINAL_ROWS=[];M6_PAIRED=[];M6_HORIZON=[];M6_HOUR=[];M6_SESSIONS=[]
for name in names:
    p=M6_PRED[name];error=p-truth;ae=abs(error);se=error**2
    ses_ae=sums(ae);ses_se=sums(se);ses_bias=sums(error)
    denom=counts[draws].sum(1)
    am=ses_ae[draws].sum(1)/denom;rm=np.sqrt(ses_se[draws].sum(1)/denom)
    bm=ses_bias[draws].sum(1)/denom
    lo,hi=bounds(am);rl,rh=bounds(rm);bl,bh=bounds(bm)
    row={'model':name,'family':family(name),'seed':M6_CANDIDATE_META.get(name,{}).get('seed'),**m6_metrics(p),
         'MAE_CI95_low':lo,'MAE_CI95_high':hi,'RMSE_CI95_low':rl,'RMSE_CI95_high':rh,
         'bias_CI95_low':bl,'bias_CI95_high':bh,'seconds_per_window':M6_TIMES.get(name)}
    if name in ['Kronos_base','Kronos_LoRA_M05']:
        row.update(CE=float(M6_CE[name].mean()),CE_s1=float(M6_CE[name][:,:,0].mean()),CE_s2=float(M6_CE[name][:,:,1].mean()))
    if name.startswith('Pilot_'):
        r=M6_IV_TABLE.set_index('model').loc[name]
        row.update(CE=float(r.CE),CE_s1=float(r.CE_s1),CE_s2=float(r.CE_s2))
    nonflat=truth!=last[:,None];signal=p!=last[:,None]
    if signal.any():
        hit=((np.sign(p-last[:,None])==np.sign(truth-last[:,None]))&nonflat)
        den=sums(nonflat);num=sums(hit)
        row['direction_CI95_low'],row['direction_CI95_high']=bounds(100*num[draws].sum(1)/den[draws].sum(1))
    M6_FINAL_ROWS.append(row)
    for ref in ['Persistencia','Kronos_base']:
        d=ses_ae-sums(abs(M6_PRED[ref]-truth));low,high=bounds(d[draws].sum(1)/denom)
        M6_PAIRED.append({'candidate':name,'reference':ref,'delta_MAE':float((ae-abs(M6_PRED[ref]-truth)).mean()),
                          'CI95_low':low,'CI95_high':high,'sessions':243,'block_sessions':5,'replicates':2000})
    for h in range(4):
        M6_HORIZON.append({'model':name,'horizon':h+1,'MAE':float(ae[:,h].mean()),'RMSE':float(np.sqrt(se[:,h].mean())),
                          'bias':float(error[:,h].mean()),'delta_MAE_persistence':float((ae[:,h]-abs(M6_PRED['Persistencia']-truth)[:,h]).mean())})
    for hour in sorted(m6_dev_ix.ny_hour.unique()):
        ids=np.flatnonzero(m6_dev_ix.ny_hour.eq(hour));cp=m6_paired_ci(p[ids],M6_PRED['Persistencia'][ids],ids)
        M6_HOUR.append({'model':name,'ny_hour':hour,**m6_metrics(p[ids],truth[ids],last[ids]),**cp})
    for i,day in enumerate(days):
        M6_SESSIONS.append({'model':name,'ny_date':day,'MAE':float(ses_ae[i]/counts[i]),'RMSE':float(np.sqrt(ses_se[i]/counts[i])),
                           'bias':float(ses_bias[i]/counts[i]),'predictions':int(counts[i])})
M6_FINAL_TABLE=pd.DataFrame(M6_FINAL_ROWS).sort_values('MAE').reset_index(drop=True)
M6_FINAL_CIS=pd.DataFrame(M6_PAIRED);M6_HORIZON=pd.DataFrame(M6_HORIZON);M6_HOUR=pd.DataFrame(M6_HOUR);M6_SESSIONS=pd.DataFrame(M6_SESSIONS)
for name,df in [('comparacion_completa',M6_FINAL_TABLE),('intervalos_pareados',M6_FINAL_CIS),('por_horizonte',M6_HORIZON),('por_hora_NY',M6_HOUR),('por_sesion_NY',M6_SESSIONS)]:
    final_csv('metricas/'+name+'.csv',df);final_json('metricas/'+name+'.json',df.to_dict(orient='records'))
seeds=M6_FINAL_TABLE[M6_FINAL_TABLE.model.str.startswith(('Head_','MC_'))].groupby('family').agg(seeds=('model','size'),MAE_mean=('MAE','mean'),MAE_std=('MAE','std'),MAE_min=('MAE','min'),MAE_max=('MAE','max')).reset_index()
final_csv('metricas/variacion_semillas.csv',seeds)

# Exact clipping rates from context-only normalization; no access to other years.
norms=[];clips=[]
for i in range(950):
    raw=m6_dev.values[int(m6_dev.starts[i]):int(m6_dev.starts[i])+60]
    mu=raw[:56].mean(0);sd=raw[:56].std(0)+np.float32(1e-5)
    norms.append((raw-mu)/sd);clips.append(abs((raw-mu)/sd)>5)
norms=np.stack(norms);clips=np.stack(clips)
cliprows=[{'field':field,'context_clip_pct':float(100*clips[:,:56,j].mean()),'future_clip_pct':float(100*clips[:,56:,j].mean())} for j,field in enumerate(M6_FIELDS)]
final_csv('metricas/clipping_por_canal.csv',pd.DataFrame(cliprows))
final_json('metricas/BSQ_latent_quantization.json',{'mean_MSE':float(M6_AUDIT['latent_quantization_mse'].mean()),
 'median_MSE':float(np.median(M6_AUDIT['latent_quantization_mse'])),'scope':'20 dimensional latent; all60 tokens; oracle diagnostic'})
attribution=pd.read_csv(M6_DIR/'metricas/atribucion_deterioro_MSE.csv')
final_csv('metricas/atribucion_deterioro_MSE.csv',attribution)
correlations=[]
for n in ['Kronos_base','Kronos_LoRA_M05']:
    ce=M6_CE[n].mean((1,2));ae=abs(M6_PRED[n]-truth).mean(1)
    correlations.append({'model':n,'Pearson_CE_MAE':float(np.corrcoef(ce,ae)[0,1]),'CE':float(ce.mean()),'MAE':float(ae.mean())})
final_csv('metricas/CE_precio.csv',pd.DataFrame(correlations))

# Compact figures include all candidates and fixed calendar examples.
fig,axes=plt.subplots(1,2,figsize=(18,10))
rank=M6_FINAL_TABLE.iloc[::-1]
axes[0].barh(rank.model,rank.MAE,color=['#222222' if n=='Persistencia' else '#377eb8' for n in rank.model])
axes[0].errorbar(rank.MAE,range(len(rank)),xerr=np.vstack([rank.MAE-rank.MAE_CI95_low,rank.MAE_CI95_high-rank.MAE]),fmt='none',ecolor='gray')
axes[0].set(title='MAE e IC95 por bloques de sesiones NY',xlabel='MAE precio XAUUSD')
paired=M6_FINAL_CIS[M6_FINAL_CIS.reference.eq('Persistencia')].set_index('candidate').loc[rank.model]
axes[1].errorbar(paired.delta_MAE,range(len(rank)),xerr=np.vstack([paired.delta_MAE-paired.CI95_low,paired.CI95_high-paired.delta_MAE]),fmt='o',capsize=3)
axes[1].axvline(0,color='black',ls='--');axes[1].set(yticks=range(len(rank)),yticklabels=rank.model,title='Delta MAE pareado frente a persistencia',xlabel='Negativo favorece candidato; selección no corregida')
final_fig('08_comparacion_final_todos.png')
for metric in ['MAE','RMSE']:
    matrix=M6_HORIZON.pivot(index='model',columns='horizon',values=metric).reindex(M6_FINAL_TABLE.model)
    fig,ax=plt.subplots(figsize=(9,11));im=ax.imshow(matrix,aspect='auto',cmap='YlOrRd')
    ax.set(yticks=range(len(matrix)),yticklabels=matrix.index,xticks=range(4),xticklabels=['H1','H2','H3','H4'],title=metric+' por horizonte: todos los candidatos')
    for i in range(len(matrix)):
        for j in range(4):ax.text(j,i,f'{matrix.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax);final_fig('09_'+metric+'_por_horizonte.png')
mat=M6_HOUR.pivot(index='model',columns='ny_hour',values='delta_MAE').reindex(M6_FINAL_TABLE.model)
fig,ax=plt.subplots(figsize=(10,11));v=max(abs(mat.to_numpy()).max(),.01);im=ax.imshow(mat,aspect='auto',cmap='RdBu_r',vmin=-v,vmax=v)
ax.set(yticks=range(len(mat)),yticklabels=mat.index,xticks=range(4),xticklabels=mat.columns,title='Delta MAE frente a persistencia según hora NY')
for i in range(len(mat)):
    for j in range(4):ax.text(j,i,f'{mat.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8)
fig.colorbar(im,ax=ax);final_fig('10_error_hora_NY.png')
groups=[['Persistencia','Kronos_base','Kronos_LoRA_M05'],[n for n in names if n.startswith('MC_') and 'Kronos_base' in n],
 [n for n in names if n.startswith('MC_') and 'Kronos_LoRA' in n],[n for n in names if n.startswith('Head_Huber')],
 [n for n in names if n.startswith('Head_Laplace')],[n for n in names if n.startswith('Ridge_')],
 [n for n in names if n.startswith('Pilot_') and not n.endswith('continuous')],[n for n in names if n.endswith('continuous')]]
fig,axes=plt.subplots(4,2,figsize=(16,17))
for ax,group in zip(axes.ravel(),groups):
    for n in group:
        e=(M6_PRED[n]-truth).ravel();ax.hist(e,bins=80,histtype='step',density=True,label=n+f' sesgo={e.mean():.3f}')
    ax.axvline(0,color='gray',ls='--');ax.set(xlabel='Predicción menos cierre real',ylabel='Densidad');ax.legend(fontsize=7)
final_fig('11_distribuciones_sesgo.png')
example_ids=[]
for quarter in range(1,5):
    candidates=np.flatnonzero((m6_dev_ix.decision_utc.dt.quarter==quarter)&m6_dev_ix.ny_hour.eq('08:30'))
    example_ids.append(int(candidates[0]))
final_json('configuraciones/ejemplos_fechas_fijas.json',{'rule':'first available08:30 NY window of each calendar quarter2020, chosen by date only','windows':example_ids,'dates':m6_dev_ix.iloc[example_ids].decision_ny.tolist()})
fig,axes=plt.subplots(2,2,figsize=(16,9))
for ax,i in zip(axes.ravel(),example_ids):
    s=int(m6_dev.starts[i]);raw=m6_dev.values[s:s+60,3]
    ax.plot(np.arange(-56,0),raw[:56],color='#555555',label='Contexto56')
    ax.plot(range(4),raw[56:],'-ok',label='Futuro real4');ax.axvline(-.5,color='gray',ls=':')
    for n in ['Persistencia','Kronos_base','Kronos_LoRA_M05','Head_Huber_s271828']:
        ax.plot(range(4),M6_PRED[n][i],'-o',label=n,ms=3)
    ax.set(title=m6_dev_ix.iloc[i].decision_ny,xlabel='Velas M15 respecto al inicio objetivo',ylabel='Precio XAUUSD');ax.legend(fontsize=7)
final_fig('12_contextos_56_futuros_4.png')
for i in example_ids:
    fig,axes=plt.subplots(4,2,figsize=(16,17))
    for ax,group in zip(axes.ravel(),groups):
        ax.plot(range(1,5),truth[i],'-ok',lw=2,label='Real');ax.axhline(last[i],color='gray',ls=':',label='Último cierre')
        for n in group:ax.plot(range(1,5),M6_PRED[n][i],'-o',label=n,ms=3)
        ax.set(title=m6_dev_ix.iloc[i].decision_ny,xlabel='Horizonte M15',ylabel='Cierre XAUUSD');ax.legend(fontsize=7)
    final_fig('13_todos_modelos_'+str(m6_dev_ix.iloc[i].ny_date)+'.png')
fig,axes=plt.subplots(2,2,figsize=(15,10))
for ax,i in zip(axes.ravel(),example_ids):
    ax.plot(range(1,5),truth[i],'-ok',label='Real')
    for n in ['clipped','continuous_bypass','oracle_tokens','oracle_s1_only']:
        ax.plot(range(1,5),M6_AUDIT[n][i,:,3],'-o',label=n)
    ax.set(title=str(m6_dev_ix.iloc[i].decision_ny)+' · ORÁCULO',xlabel='Horizonte',ylabel='Cierre');ax.legend(fontsize=8)
final_fig('14_reconstruccion_BSQ_fechas_reales.png')
fig,axes=plt.subplots(1,2,figsize=(14,5))
axes[0].hist(M6_AUDIT['latent_quantization_mse'],bins=60);axes[0].set(title='Distorsión de cuantización latente',xlabel='MSE (20 dimensiones)')
axes[1].hist(norms[:,56:,3].ravel(),bins=80);axes[1].axvline(-5,color='red');axes[1].axvline(5,color='red');axes[1].set(title='Cierre futuro normalizado solo con contexto',xlabel='Z-score antes de clipping')
final_fig('15_cuantizacion_y_clipping.png')
fig,axes=plt.subplots(1,2,figsize=(14,5))
for n in ['Kronos_base','Kronos_LoRA_M05']:
    axes[0].scatter(M6_CE[n].mean((1,2)),abs(M6_PRED[n]-truth).mean(1),s=8,alpha=.3,label=n)
axes[0].set(xlabel='CE teacher forced',ylabel='MAE rollout',title='Por ventana: objetivos diferentes');axes[0].legend()
ce_rows=M6_FINAL_TABLE.dropna(subset=['CE'])
for row in ce_rows.itertuples():
    if str(row.model).endswith('continuous'):continue
    axes[1].scatter(row.CE,row.MAE);axes[1].annotate(row.model,(row.CE,row.MAE),fontsize=6)
axes[1].set(xlabel='CE media en2020',ylabel='MAE de precio',title='Referencias y pilotos a paso fijo')
final_fig('16_CE_frente_precio.png')

# Explicit evidence gate: point improvement is not sufficient for promotion.
persist=float(M6_FINAL_TABLE.set_index('model').loc['Persistencia','MAE'])
base=float(M6_FINAL_TABLE.set_index('model').loc['Kronos_base','MAE'])
point=M6_FINAL_TABLE[(M6_FINAL_TABLE.MAE<persist)&(M6_FINAL_TABLE.MAE<base)].model.tolist()
ci_pass=[]
for n in point:
    cis=M6_FINAL_CIS[M6_FINAL_CIS.candidate.eq(n)]
    if (cis.CI95_high<0).all():ci_pass.append(n)
robust=[]
for n in ci_pass:
    if M6_HORIZON[M6_HORIZON.model.eq(n)].delta_MAE_persistence.lt(0).all() and M6_HOUR[M6_HOUR.model.eq(n)].CI95_high.lt(0).all():robust.append(n)
decision={'point_better_than_both':point,'paired_CI_better_than_both':ci_pass,'all_horizons_and_hours_pass':robust,
 'promotion':False,'reason':'2020 reused for development; independent authorized evaluation required regardless of exploratory intervals',
 'reserved2021_2022_executed':False,'test2023_2024_accessed':False,'orders_enabled':False}
final_json('informes/decision_cientifica.json',decision)
if point:
    proposal='''# Propuesta M07 — pendiente de autorización, NO ejecutada

Hay mejoras puntuales frente a ambas referencias en2020, pero esto no demuestra generalización.
Congelar antes de evaluar: tokenizer/base/hashes, features causales, normalizaciónL56, H4,
cabezas Huber de las tres semillas predefinidas y sus checkpoints ya elegidos, sin nuevas búsquedas.
Comparar cada semilla con persistencia y Kronos base greedy; publicar MAE/RMSE/dirección/sesgo,
intervalos pareados por bloques de sesionesNY, estabilidad por horizonte/hora/año y magnitud.
Mantener primario MAE acumuladoH1-H4 y reportar todos los candidatos prefijados; sin escoger
el mejor tras ver2021–2022. Exigir mejora frente a ambas referencias, consistencia entre semillas,
ausencia de deterioro material por horizonte y evaluación de selección múltiple.
2021–2022 ya participaron en evaluaciones históricasM05: no presentarlos como datos nunca vistos
por la investigación. Un resultado favorable requerirá posterior evidencia verdaderamente independiente.
No cargar2021–2022 hasta autorización explícita. Test2023–2024 seguirá completamente reservado.
Sin backtest con costes no se concluye ventaja económica; órdenes reales deshabilitadas.
'''
    final_bytes('informes/PROPUESTA_M07_NO_EJECUTADA.md',proposal.encode())

# Versioned executable sources and hashes belong to this run, alongside data provenance.
source_manifest=[]
for p in [Path('/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION')/n for n in [
 'm06_recovery_stable_v2_39a2cbcf.py','m06_stability_report_v3_183ffe4d.py','m06_mc_guarded_baaf376e.py',
 'm06_heads_atomic_v1_92885403.py','m06_pilots_3ee88503.py','m06_completed_recovery_5557a47b.py']]:
    final_bytes('codigo/'+p.name,p.read_bytes());source_manifest.append({'name':p.name,'sha256':m6_stable.sha(p),'bytes':p.stat().st_size})
final_json('configuraciones/codigo_y_entorno.json',{'code':source_manifest,'python':platform.python_version(),'torch':torch.__version__,
 'gpu':torch.cuda.get_device_name(0),'gpu_peak_GiB':torch.cuda.max_memory_allocated()/2**30,'base':M6_BASE_HASH,'tokenizer':M6_TOK_HASH,
 'M05_checkpoint':str(M6_CHECKPOINT),'M05_sha256':m6_sha(M6_CHECKPOINT),'data':M6_SOURCE_MANIFEST,'seeds':[271828,271829,271830]})
assert m6_hash(m6_model,True)==M6_BASE_HASH and m6_hash(m6_tok)==M6_TOK_HASH
assert all(torch.equal(p.detach().cpu(),M6_ADAPTER_M05[n]) for n,p in M6_ADAPTER_PARAMS.items())
assert not any(p.requires_grad for p in m6_model.parameters())
assert m6_stable.inventory()==json.loads((m6_stable.OPS/'artifact_manifest_before_recovery.json').read_text())
client=_message.blocking_request('get_ipynb',timeout_sec=30)['ipynb']
original=json.loads((M6_DIR/'auditoria/notebook_al_inicio.ipynb').read_text())
comparisons=[]
for i,(a,b) in enumerate(zip(original['cells'][:40],client['cells'][:40])):
    comparisons.append({'index':i,'id_equal':a.get('metadata',{}).get('id')==b.get('metadata',{}).get('id'),
      'source_equal':''.join(a.get('source',[]))==''.join(b.get('source',[])),
      'outputs_equal':a.get('outputs',[])==b.get('outputs',[]),'metadata_equal':a.get('metadata',{})==b.get('metadata',{}),
      'execution_count_equal':a.get('execution_count')==b.get('execution_count')})
assert len(comparisons)==40 and all(r['id_equal'] and r['source_equal'] and r['outputs_equal'] for r in comparisons)
final_json('auditoria/historicas40.json',comparisons)
m6_stable.atomic_json(M6_DIR/'auditoria/estabilizacion_v2/NOTEBOOK_M06_FINAL_ANTES_SALIDA.ipynb',client)

close=M6_RECON_TABLE[M6_RECON_TABLE.variable.eq('close')]
factor=M6_FACTORIAL[M6_FACTORIAL.metric.eq('MSE')]
best=M6_FINAL_TABLE.iloc[0]
report=f'''# AXXEL CAPITAL — Misión06: informe técnico ejecutado

Fecha UTC: {datetime.now(timezone.utc).isoformat()}. RESEARCH ONLY — ORDERS DISABLED.

## Resultado y alcance

Se ejecutaron auditoríaBSQ, inferenciaMonteCarlo, regresión continua congelada y cinco pilotosLoRA.
Comparación homogénea:8428 ventanas train2010–2019,950 ventanas/243 sesionesNY de desarrollo2020,
L56/H4 M15, normalización exclusivamente contextual, semillas y versiones fijadas.
Persistencia MAE={persist:.6f}; Kronos base greedy={base:.6f}. Mejor punto exploratorio:
{best['model']} MAE={best['MAE']:.6f}. Mejorar puntualmente no equivale a superar la incertidumbre.
Candidatos con ambos IC95 pareados por debajo de cero: {ci_pass}.
Modelos promovidos: ninguno. No se ejecutó evaluación2021–2022 ni se accedió al test2023–2024.
Las métricasM05 originales incluyen otro alcance temporal; no confundirlas con estas950 ventanas2020.

## Comparación completa — desarrollo2020

{md_table(M6_FINAL_TABLE[['model','MAE','RMSE','direction_pct','bias','magnitude_MAE','MAE_CI95_low','MAE_CI95_high']])}

## Auditoría real del tokenizador y atribución

El código inspeccionado del commit67b630e67f6a18c9e9be918d9b4337c960db1e9a proyecta OHLCV+amount0
al encoder causal y a20 coordenadas; normalizaL2 y cuantiza el signo a ±1/sqrt(20).
s1 codifica las primeras10 coordenadas y s2 las últimas10, mediante potencias de2 ascendentes.
Cada vocabulario tiene1024 IDs. No son categorías financieras de tendencia o ruido.
El decoder concatena ambos subcódigos, aplica proyección aprendida, bloques causales y cabezaOHLCV.
Existe una proyección/decoder parcial entrenado para s1; esto no demuestra semántica exclusiva.

{md_table(close[['intervention','MAE','RMSE']])}

El oráculo usa precios/tokens futuros verdaderos y está excluido de cualquier ranking predictivo.
La reconstrucción con tokens reales es mucho más precisa que la predicción libre. El error de
reconstrucción es relevante y no nulo, pero no explica por sí solo el gran error predictivo observado.
El bypass continuo empeora frente a los códigos discretos: eliminar la cuantización sin reentrenar
el decoder no mejora automáticamente. Ese bypass cambia la distribución latente; no identifica
una contribución causal pura e independiente de la cuantización ni un límite irreducible del MAE.

{md_table(pd.DataFrame(cliprows))}

Intervenciones factoriales: mantener tokens de contexto reales; combinar futuros{s1 if False else ''}
s1/s2 reales y predichos, sin volver a generar el otro subcódigo. Esto mide sustitución controlada
en el decoder, no el efecto de entrenar cada componente de forma aislada. La descomposiciónShapley
es una convención explícita para repartir interacción; no prueba independencia causal.

{md_table(factor[['model','E00_real_real','E10_pred_real','E01_real_pred','E11_pred_pred','interaction','shapley_s1','shapley_s2']])}

Para el deterioroLoRA−base se usa además e=e_clipping+e_decoder_continuo+e_quant_intervencion+
e_prediccion. Se expande MSE=sum E[e_i²]+2sum E[e_i e_j]. Las contribuciones firmadas y porcentajes
están en atribucion_deterioro_MSE.csv; pueden ser negativas o mayores que100 por interacción.
El tokenizer y decoder permanecieron idénticos en ambas ramas; sus términos propios no cambian.
La diferencia aparece en la predicción y sus interacciones con la reconstrucción. No procede asignar
porcentajes causales únicos a componentes dependientes.

{md_table(attribution)}

## Monte Carlo y presupuesto

Cinco configuraciones prefijadas por modelo:T0.6/T0.8/T1.0, top-k50 aT0.8 y top-p0.9 aT0.8.
16/32 trayectorias anidadas, media/mediana en espacio de precios. Exploración común en32 sesiones;
una configuración por modelo, replicada en950 ventanas con tres semillas. Se conservaron cachés
y tiempos medidos; los16 caminos reutilizan la primera mitad de32, no una medida de latencia16 separada.
La mediaMonteCarlo reduce el MAE frente al greedy en estos datos, pero ninguna réplica supera
persistencia. Las bandas de dispersión describen trayectorias del modelo, no intervalos calibrados.
Resultados, latencias y dispersión: metricas/MC_exploracion.csv, MC_replicas_2020.csv,
MC_intervalos_pareados.csv, MC_tiempos.csv del directorio principalM06.

## Regresión continua

Representación512 del último contexto en decode_s1; prueba de invariancia al modificar los futuros.
Kronos, tokenizer y adaptadoresM05 congelados. Scalers ajustados solo en train. Cabeza512→64GELU→4
para Huber y→8 para localización/escalaLaplace; retornos acumuladosH1–H4 enbps.
Seis entrenamientos de300 pasos, batch64, Adam0.001, tres semillas por pérdida. Selección de
checkpoint cada50 pasos usando2020; paso0 equivale a persistencia. BaselinesRidge con alpha100
fijo sobre12 variables causales y sobre512 representaciones. No se buscó alpha en validación.
Huber muestra ganancias pequeñas en las tres semillas; sus intervalos pareados incluyen cero
frente a persistencia. Laplace puede seleccionar paso0. Las coberturas nominales80/90 deben
compararse con las observadas; no declarar incertidumbre calibrada.

{md_table(seeds)}

## Pilotos LoRA

Cinco pilotos nuevos desde base; originales intactos. Q/Vr8; FFNr4; Q/Vr4+FFNr2;
Q/V con CE_s2 ponderada2:1; Q/V con CE+0.1Huber vía cabeza congelada de primera semilla.
Todos147456 parámetros,100 pasos,LR1e-4,lote efectivo64/micro16×4,dropout0.1 y escala2.
El camino auxiliar produce gradiente no nulo medido, sin argmax/decoder discreto en esa ruta.
El checkpoint es el paso100 fijo. Las curvas intermedias usan124 ventanas; el punto100 usa950:
no atribuir ese cambio de población al aprendizaje. Cada piloto tiene una semilla; no implica
que toda la familiaLoRA esté descartada. No se escaló hardware ni se efectuó búsqueda masiva.

{md_table(M6_IV_TABLE[['model','MAE','RMSE','CE','trainable_parameters','aux_gradient_norm']])}

## Evaluación, sesgos y límites

IC95:2000 remuestreos pareados de bloques circulares de5 sesionesNY, preservando las ventanas
intrasesión. CSV por horizonte, horaNY y sesión completos; variación entre tres semillas paraMC
y cabezas. Los intervalos son exploratorios: no corrigen selección múltiple ni reutilización2020.
2020 participa en exploración/selección; no existe evaluación final independiente de generalización.
No demostrar mejora no prueba ausencia de señal en todoKronos. El volumen disponible y amount0,
la muestra de anclasNY, la escala/régimen de2020 y el presupuesto limitado condicionan conclusiones.
No se ha medido rentabilidad, slippage, comisiones ni ejecución; ninguna ventaja económica declarada.

## Incidencias y estabilización

Conflicto de versiones por editores/escrituras concurrentes: ambas versiones respaldadas y
comparadas porID,código,metadatos,salidas. Principal modificado solo medianteColab; API sin
reemplazos del notebook. Se preservaron40 celdas históricas yM05. Un contador de ejecución
histórico fue cambiado porColab antes de estabilizar; código y salidas permanecieron idénticos.
SerializaciónNaN: fallo separado del conflicto; valores no finitos se rechazan en artefactos
críticos y los campos no aplicables se convierten explícitamente anull con registro. No se rellenan
métricas inexistentes con cero. Archivos independientes con temporales,fsync,hash,size,lectura,
locks y control de versión esperada. DriveFUSE no garantiza transacciones distribuidas absolutas.
Las pruebasA–F pasaron; estado M06_ENVIRONMENT_STABLE derivado de evidencias, no habilitado a mano.
La recuperación fue probada dos veces desde espacios vacíos; más tarde se añadió restauración
deIII–IV con hashes y checkpoints, también probada dos veces sin repetir entrenamiento.
FalloMonteCarlo row.aggregate: conflicto con método deSeries; corregido con row['aggregate'],
reutilizando trayectorias válidas. Desconexiones espontáneas: recuperación desde archivos.
No se garantiza que un escritor externo desconocido no reaparezca; detectar, respaldar y reconciliar
antes de sobrescribir. Consultar auditoria/estabilizacion_v2/INFORME_ESTABILIZACION.md.

## Recomendación científica para Misión07

No promover porCE ni por una reducción puntual deMAE. La alineación continua evita parte del
deterioro del greedy, pero todavía debe demostrar ventaja reproducible frente a persistencia.
Priorizar réplica independiente con protocolo congelado y comprobar magnitud/calibración,
causalidad y distribución por sesión antes de ampliar entrenamiento. La propuesta2021–2022
queda escrita y NO ejecutada; esos años ya fueron vistos en evaluaciones históricasM05.
Mantener completamente reservado2023–2024. No conectar ejecución real.

## Trazabilidad y recuperación

Los resultadosI–II originales permanecen en metricas/graficas/checkpoints. III yIV tienen carpetas
independientes fase_III/fase_IV conchecksums.json, optimizadores,semillas,RNG y predicciones.
Esta síntesis contiene métricasCSV/JSON,PNG,código y manifiesto. Ejecutar solo celda de recuperación,
prueba de guardado e informe de estabilidad; después usar loadersidempotentes para fases ya
terminadas. Nunca ejecutar todas ni las recuperacionesM05 que cargan otros años.
'''
final_bytes('informes/INFORME_TECNICO_FINAL_M06.md',report.encode())
final_verify()
final_json('phase_status.json',{'complete':True,'candidates':len(names),'historical40_code_outputs_preserved':True,
 'phaseIII_IV_restore_idempotent':True,'M06_ENVIRONMENT_STABLE':True,'promotion':False,'evaluation_scope':'development2020 only'})
display(Markdown('## M06 ejecutada — desarrollo2020, sin promoción'))
display(M6_FINAL_TABLE[['model','MAE','RMSE','direction_pct','bias']]);display(M6_FINAL_CIS)
display(Markdown('**Informe técnico:** '+str(M6_FINAL_DIR/'informes/INFORME_TECNICO_FINAL_M06.md')))
print('M06_EXPERIMENTOS_COMPLETOS',len(names),'candidatos; validación independiente NO ejecutada')
print('M06_ENVIRONMENT_STABLE = TRUE; originales y cachés verificados; resultados persistidos')
