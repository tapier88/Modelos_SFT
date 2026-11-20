#@title MISIÓN 06 · 4. Fase II: Monte Carlo, selección acotada y tres semillas
assert globals().get('M6_PHASE_I_DONE') is True, 'Recuperar M06 en este runtime antes de continuar'
M6_MC_BATCH=8
M6_CANDIDATE_META=globals().get('M6_CANDIDATE_META',{})
M6_MC_RUNS=[]
M6_MC_FULL_STATS=[]
M6_MC_PILOT_ROWS=[]
M6_MC_REPLICA_ROWS=[]

def m6_mc_cached(label,config,ids,seed,scope):
    ids=np.asarray(ids,dtype=int)
    identity={'label':label,'config':config,'ids':ids.tolist(),'seed':int(seed),'paths':32,
              'batch':M6_MC_BATCH,'base_hash':M6_BASE_HASH,'tokenizer_hash':M6_TOK_HASH,
              'adapter_checkpoint_sha256':M6_PROVENANCE[-1]['sha256'],
              'dataset_hashes':M6_SOURCE_MANIFEST,'seed_rule':'seed + first_window_id_of_batch * 1009'}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest()[:16]
    folder=M6_DIR/'checkpoints/fase_II'/digest; folder.mkdir(parents=True,exist_ok=True)
    m6_save_json(str(folder.relative_to(M6_DIR)/'identity.json'),identity)
    outputs=[]; seconds=0.
    m6_mode(label)
    for offset in range(0,len(ids),64):
        shard=folder/f'shard_{offset:04d}.npz'
        subset=ids[offset:offset+64]
        if shard.exists():
            with np.load(shard,allow_pickle=False) as z:
                assert np.array_equal(z['ids'],subset)
                result=z['paths']; duration=float(z['seconds'])
        else:
            parts=[]; duration=0.
            for j in range(0,len(subset),M6_MC_BATCH):
                batch_ids=subset[j:j+M6_MC_BATCH]
                batch=m6_batch(m6_dev,batch_ids)
                m6_sync(); t0=time.perf_counter()
                p,_=m6_rollout(batch,paths=32,seed=int(seed)+int(batch_ids[0])*1009,
                               temperature=config['temperature'],top_k=config['top_k'],top_p=config['top_p'])
                m6_sync(); duration+=time.perf_counter()-t0
                parts.append(p[:,:,:,3].cpu().numpy())
            result=np.concatenate(parts)
            assert np.isfinite(result).all()
            np.savez_compressed(shard,ids=subset,paths=result,seconds=duration)
        outputs.append(result); seconds+=duration
        if scope=='desarrollo_completo' and (offset==0 or offset%256==0):
            print('MC',scope,label,config['id'],'seed',seed,offset+len(subset),'/',len(ids),flush=True)
    M6_MC_RUNS.append({'model':label,'configuration':config['id'],'seed':seed,'scope':scope,'windows':len(ids),
                       'paths':32,'seconds_inference':seconds,'seconds_per_window':seconds/len(ids),'cache':digest})
    return np.concatenate(outputs),seconds

# Cinco configuraciones prefijadas, mismo subconjunto de 32 sesiones para todos.
for label in M6_LABELS:
    for cfg in M6_PLAN['mc_configs']:
        paths,seconds=m6_mc_cached(label,cfg,M6_MC_IDS,271828,'exploracion_32_sesiones')
        for count in [16,32]:
            for agg in ['mean','median']:
                pred=getattr(np,agg)(paths[:,:count],axis=1)
                row={'model':label,'configuration':cfg['id'],'paths':count,'aggregate':agg,
                     'scope':'exploracion_32_sesiones','seed':271828,
                     **m6_metrics(pred,M6_Y[M6_MC_IDS],M6_LAST[M6_MC_IDS]),
                     'trajectory_std':float(paths[:,:count].std(axis=1).mean()),
                     'seconds_per_window_32_paths':seconds/len(M6_MC_IDS)}
                M6_MC_PILOT_ROWS.append(row)
                pd.DataFrame(M6_MC_PILOT_ROWS).to_csv(M6_DIR/'metricas/MC_exploracion.csv',index=False)
        print('MC_EXPLORACION_LISTA',label,cfg['id'],flush=True)

M6_MC_PILOT=pd.DataFrame(M6_MC_PILOT_ROWS)
M6_MC_SELECTED=[]
for label in M6_LABELS:
    # Desempate determinista; no ampliar la búsqueda tras ver estos resultados.
    row=M6_MC_PILOT[M6_MC_PILOT.model.eq(label)].sort_values(['MAE','configuration','paths','aggregate']).iloc[0]
    choice={'model':label,'configuration':str(row.configuration),'paths':int(row.paths),'aggregate':str(row.aggregate),
            'selection_MAE':float(row.MAE),'selection_scope':'32 sesiones de desarrollo 2020; no evaluacion independiente'}
    M6_MC_SELECTED.append(choice)
m6_save_json('configuraciones/MC_seleccion_fijada.json',M6_MC_SELECTED)
print('MC_SELECCION_FIJADA',M6_MC_SELECTED,flush=True)

for choice in M6_MC_SELECTED:
    label=choice['model']; cfg=next(c for c in M6_PLAN['mc_configs'] if c['id']==choice['configuration'])
    for seed in M6_PLAN['replicate_seeds']:
        paths,seconds=m6_mc_cached(label,cfg,np.arange(len(m6_dev)),seed,'desarrollo_completo')
        pred=getattr(np,choice['aggregate'])(paths[:,:choice['paths']],axis=1).astype(np.float64)
        name=f"MC_{label}_{cfg['id']}_{choice['paths']}_{choice['aggregate']}_s{seed}"
        M6_PRED[name]=pred; M6_TIMES[name]=seconds/len(m6_dev)
        M6_CANDIDATE_META[name]={'family':'MC_'+label,'seed':seed,'eligible':True,'selection':'MC_seleccion_fijada.json'}
        stats={'model':name,'base_model':label,'seed':seed,**m6_metrics(pred),
               'trajectory_std':float(paths[:,:choice['paths']].std(axis=1).mean()),
               'seconds_per_window_32_paths':seconds/len(m6_dev)}
        M6_MC_REPLICA_ROWS.append(stats)
        m6_prediction_frame(name,pred).to_parquet(M6_DIR/'metricas'/f'{name}.parquet',index=False)
        for ref in ['Persistencia','Kronos_base']:
            M6_MC_FULL_STATS.append({'candidate':name,'reference':ref,**m6_paired_ci(pred,M6_PRED[ref])})
        if seed==271828:
            # Comparación de agregaciones para visualizar; no sustituye la elección fijada.
            for count in [16,32]:
                for agg in ['mean','median']:
                    p=getattr(np,agg)(paths[:,:count],axis=1).astype(np.float64)
                    diag=f'MC_DIAG_{label}_{count}_{agg}'
                    M6_PRED[diag]=p
                    M6_CANDIDATE_META[diag]={'family':'MC_diagnostic','seed':seed,'eligible':False}
        pd.DataFrame(M6_MC_REPLICA_ROWS).to_csv(M6_DIR/'metricas/MC_replicas_2020.csv',index=False)
        pd.DataFrame(M6_MC_FULL_STATS).to_csv(M6_DIR/'metricas/MC_intervalos_pareados.csv',index=False)
        pd.DataFrame(M6_MC_RUNS).to_csv(M6_DIR/'metricas/MC_tiempos.csv',index=False)
        m6_save_json('configuraciones/candidatos.json',M6_CANDIDATE_META)
        print('MC_REPLICA_FINALIZADA',name,'MAE',stats['MAE'],flush=True)

M6_MC_REPLICAS=pd.DataFrame(M6_MC_REPLICA_ROWS)
fig,axes=plt.subplots(1,2,figsize=(14,5))
for axis,label in zip(axes,M6_LABELS):
    sub=M6_MC_PILOT[M6_MC_PILOT.model.eq(label)]
    for (count,agg),g in sub.groupby(['paths','aggregate']):
        g=g.set_index('configuration').reindex([c['id'] for c in M6_PLAN['mc_configs']])
        axis.plot(g.index,g.MAE,marker='o',label=f'{count} {agg}')
    for ref in ['Persistencia','Kronos_base']:
        mae=m6_metrics(M6_PRED[ref][M6_MC_IDS],M6_Y[M6_MC_IDS],M6_LAST[M6_MC_IDS])['MAE']
        axis.axhline(mae,linestyle='--',label=ref)
    axis.set(title=label+' · 32 sesiones exploratorias',ylabel='MAE de precio'); axis.legend(fontsize=8)
m6_figsave('04_MC_busqueda_acotada.png')
fig,axes=plt.subplots(1,2,figsize=(14,5))
for axis,label in zip(axes,M6_LABELS):
    names=[label,'Persistencia']+[f'MC_DIAG_{label}_{n}_{agg}' for n in [16,32] for agg in ['mean','median']]
    for name in names: axis.plot(range(1,5),np.abs(M6_PRED[name]-M6_Y).mean(0),marker='o',label=name.replace('MC_DIAG_'+label+'_',''))
    axis.set(title=label+' · 950 ventanas 2020',xlabel='Horizonte M15',ylabel='MAE'); axis.legend(fontsize=7)
m6_figsave('05_MC_media_mediana_horizonte.png')
display(M6_MC_REPLICAS)
display(pd.DataFrame(M6_MC_FULL_STATS))
assert m6_hash(m6_model,True)==M6_BASE_HASH and m6_hash(m6_tok)==M6_TOK_HASH
M6_PHASE_II_DONE=True
print('FASE_II_COMPLETA; 2020 es desarrollo reutilizado; no promover sin evaluación posterior autorizada')
