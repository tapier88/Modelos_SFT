#@title MISIÓN 06 · 3. Fase I: BSQ, clipping, intervenciones y referencias 2020
assert globals().get('M6_READY') is True
M6_LABELS=['Kronos_base','Kronos_LoRA_M05']
M6_FIELDS=['open','high','low','close','volume','amount']
M6_PRED={}; M6_CE={}; M6_TIMES={}
M6_AUDIT_PARTS=[]
audit_dir=M6_DIR/'checkpoints/fase_I'; audit_dir.mkdir(exist_ok=True)

@torch.no_grad()
def m6_decode_latent(latent, pre=False):
    z=m6_tok.post_quant_embed_pre(latent) if pre else m6_tok.post_quant_embed(latent)
    for layer in m6_tok.decoder: z=layer(z)
    return m6_tok.head(z)

for start in range(0,len(m6_dev),16):
    cache=audit_dir/f'batch_{start:04d}.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as cached: part={k:cached[k] for k in cached.files}
        M6_AUDIT_PARTS.append(part); continue
    batch=m6_batch(m6_dev,range(start,min(start+16,len(m6_dev))))
    with torch.no_grad():
        true_a,true_b=m6_tok.encode(batch['x'],half=True)
        context_a,context_b=m6_tok.encode(batch['x'][:,:56],half=True)
        assert torch.equal(true_a[:,:56],context_a) and torch.equal(true_b[:,:56],context_b)
        z=m6_tok.embed(batch['x'])
        for layer in m6_tok.encoder: z=layer(z)
        continuous=F.normalize(m6_tok.quant_embed(z),dim=-1)
        quantized=m6_tok.indices_to_bits([true_a,true_b],half=True)
        def to_price(norm): return norm[:,-4:]*batch['scale'][:,None]+batch['mean'][:,None]
        clip=to_price(batch['x'])
        cont=to_price(m6_decode_latent(continuous))
        oracle=to_price(m6_tok.decode([true_a,true_b],half=True))
        pre=to_price(m6_decode_latent(quantized[:,:,:10],pre=True))
        part={'truth':batch['truth'].cpu().numpy(),'last_close':batch['last_close'].cpu().numpy(),
              'clipped':clip.cpu().numpy(),'continuous_bypass':cont.cpu().numpy(),
              'oracle_tokens':oracle.cpu().numpy(),'oracle_s1_only':pre.cpu().numpy(),
              'latent_quantization_mse':(quantized-continuous).square().mean((1,2)).cpu().numpy()}
        for label in M6_LABELS:
            m6_mode(label); m6_sync(); t0=time.perf_counter()
            pred,(pa,pb)=m6_rollout(batch)
            m6_sync(); elapsed=time.perf_counter()-t0
            part[label]=pred[:,0].cpu().numpy()
            part[label+'_seconds']=np.array([elapsed])
            part[label+'_CE']=m6_ce(batch).cpu().numpy()
            aa,bb=pa[:,0].clone(),pb[:,0].clone()
            aa[:,56:]=true_a[:,56:]
            part[label+'_real_s1_pred_s2']=to_price(m6_tok.decode([aa,bb],half=True)).cpu().numpy()
            aa,bb=pa[:,0].clone(),pb[:,0].clone()
            bb[:,56:]=true_b[:,56:]
            part[label+'_pred_s1_real_s2']=to_price(m6_tok.decode([aa,bb],half=True)).cpu().numpy()
            part[label+'_token_accuracy']=np.stack([(pa[:,0,56:]==true_a[:,56:]).cpu().numpy(),(pb[:,0,56:]==true_b[:,56:]).cpu().numpy()],-1)
        assert all(np.isfinite(v).all() for v in part.values())
    np.savez_compressed(cache,**part)
    M6_AUDIT_PARTS.append(part)
    if start%160==0: print('FASE_I',start,'/',len(m6_dev),flush=True)

M6_AUDIT={k:np.concatenate([p[k] for p in M6_AUDIT_PARTS],axis=0) for k in M6_AUDIT_PARTS[0]}
M6_Y=M6_AUDIT['truth'][:,:,3].astype(np.float64)
M6_LAST=M6_AUDIT['last_close'].astype(np.float64)
M6_PRED['Persistencia']=np.repeat(M6_LAST[:,None],4,axis=1)
for label in M6_LABELS:
    M6_PRED[label]=M6_AUDIT[label][:,:,3].astype(np.float64)
    M6_CE[label]=M6_AUDIT[label+'_CE']
    M6_TIMES[label]=float(M6_AUDIT[label+'_seconds'].sum()/len(m6_dev))

def m6_metrics(pred,truth=M6_Y,last=M6_LAST):
    error=np.asarray(pred,dtype=np.float64)-truth
    change=pred-last[:,None]; actual=truth-last[:,None]
    nonflat=actual!=0; signals=change!=0
    return {'MAE':float(np.abs(error).mean()),'RMSE':float(np.sqrt(np.mean(error**2))),
            'bias':float(error.mean()),'direction_pct':float(100*np.mean(np.sign(change)[nonflat]==np.sign(actual)[nonflat])) if signals.any() and nonflat.any() else None,
            'direction_coverage_pct':float(100*signals.mean()),'actual_nonzero':int(nonflat.sum()),
            'magnitude_MAE':float(np.abs(np.abs(change)-np.abs(actual)).mean()),'windows':len(truth)}

def m6_prediction_frame(label,pred,ids=None):
    ids=np.arange(len(m6_dev)) if ids is None else np.asarray(ids)
    ix=m6_dev_ix.iloc[ids]
    return pd.DataFrame({'model':label,'window':np.repeat(ids,4),'horizon':np.tile(np.arange(1,5),len(ids)),
                         'ny_date':np.repeat(ix.ny_date.to_numpy(),4),'ny_hour':np.repeat(ix.ny_hour.to_numpy(),4),
                         'decision_utc':np.repeat(ix.decision_utc.astype(str).to_numpy(),4),
                         'truth':M6_Y[ids].ravel(),'last_close':np.repeat(M6_LAST[ids],4),'prediction':pred.ravel()})

def m6_paired_ci(pred,reference,ids=None):
    ids=np.arange(len(M6_Y)) if ids is None else np.asarray(ids)
    delta=np.abs(pred-M6_Y[ids])-np.abs(reference-M6_Y[ids])
    group=pd.DataFrame({'session':m6_dev_ix.iloc[ids].ny_date.to_numpy(),'delta_sum':delta.sum(1),'n':4}).groupby('session',sort=True).sum()
    values=group.delta_sum.to_numpy(); counts=group.n.to_numpy(); n=len(group)
    rng=np.random.default_rng(271828); draws=[]
    for _ in range(2000):
        starts=rng.integers(0,n,size=math.ceil(n/5))
        take=((starts[:,None]+np.arange(5))%n).ravel()[:n]
        draws.append(values[take].sum()/counts[take].sum())
    lo,hi=np.quantile(draws,[.025,.975])
    return {'delta_MAE':float(delta.mean()),'CI95_low':float(lo),'CI95_high':float(hi),'sessions':n,'bootstrap_replicates':2000,'block_sessions':5,'interpretation':'exploratorio; no corrige seleccion multiple'}

M6_BASELINE_TABLE=pd.DataFrame([{'model':label,**m6_metrics(p),'seconds_per_window':M6_TIMES.get(label)} for label,p in M6_PRED.items()])
M6_BASELINE_TABLE.to_csv(M6_DIR/'metricas/referencias_2020.csv',index=False)
pd.concat([m6_prediction_frame(label,p) for label,p in M6_PRED.items()]).to_parquet(M6_DIR/'metricas/referencias_predicciones_2020.parquet',index=False)
reconstruction=[]
for name in ['clipped','continuous_bypass','oracle_tokens','oracle_s1_only']:
    for j,field in enumerate(M6_FIELDS):
        e=M6_AUDIT[name][:,:,j].astype(np.float64)-M6_AUDIT['truth'][:,:,j]
        reconstruction.append({'intervention':name,'variable':field,'MAE':float(abs(e).mean()),'RMSE':float(np.sqrt((e**2).mean())),'scope':'ORACULO_DIAGNOSTICO_NO_PREDICTOR'})
M6_RECON_TABLE=pd.DataFrame(reconstruction)
M6_RECON_TABLE.to_csv(M6_DIR/'metricas/reconstruccion_OHLCV.csv',index=False)
factorial=[]; decomposition=[]
for label in M6_LABELS:
    e00=M6_AUDIT['oracle_tokens'][:,:,3]-M6_Y
    e10=M6_AUDIT[label+'_pred_s1_real_s2'][:,:,3]-M6_Y
    e01=M6_AUDIT[label+'_real_s1_pred_s2'][:,:,3]-M6_Y
    e11=M6_PRED[label]-M6_Y
    for metric,fun in [('MAE',lambda e:np.abs(e).mean()),('MSE',lambda e:np.square(e).mean())]:
        E00,E10,E01,E11=map(fun,[e00,e10,e01,e11])
        interaction=E11-E10-E01+E00
        phi1=.5*((E10-E00)+(E11-E01)); phi2=.5*((E01-E00)+(E11-E10))
        assert np.isclose(E00+phi1+phi2,E11)
        factorial.append({'model':label,'metric':metric,'E00_real_real':E00,'E10_pred_real':E10,'E01_real_pred':E01,'E11_pred_pred':E11,'interaction':interaction,'shapley_s1':phi1,'shapley_s2':phi2})
    stages={'clipping':M6_AUDIT['clipped'][:,:,3]-M6_Y,
            'decoder_continuous':M6_AUDIT['continuous_bypass'][:,:,3]-M6_AUDIT['clipped'][:,:,3],
            'quantization_intervention':M6_AUDIT['oracle_tokens'][:,:,3]-M6_AUDIT['continuous_bypass'][:,:,3],
            'token_prediction':M6_PRED[label]-M6_AUDIT['oracle_tokens'][:,:,3]}
    assert np.allclose(sum(stages.values()),e11)
    terms=[]
    for name,e in stages.items(): terms.append((name,float(np.mean(e**2))))
    names=list(stages)
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            terms.append((names[i]+' x '+names[j],float(2*np.mean(stages[names[i]]*stages[names[j]]))))
    assert np.isclose(sum(v for _,v in terms),np.mean(e11**2),rtol=1e-5,atol=1e-5)
    for name,value in terms: decomposition.append({'model':label,'term':name,'MSE_contribution':value})
M6_FACTORIAL=pd.DataFrame(factorial); M6_DECOMP=pd.DataFrame(decomposition)
M6_FACTORIAL.to_csv(M6_DIR/'metricas/subcodigos_factorial.csv',index=False)
M6_DECOMP.to_csv(M6_DIR/'metricas/descomposicion_MSE_interacciones.csv',index=False)
degradation=M6_DECOMP.pivot(index='term',columns='model',values='MSE_contribution')
degradation['delta_MSE']=degradation.Kronos_LoRA_M05-degradation.Kronos_base
total_delta=float(degradation.delta_MSE.sum())
degradation['signed_percent_of_delta']=100*degradation.delta_MSE/total_delta if abs(total_delta)>1e-10 else np.nan
degradation.to_csv(M6_DIR/'metricas/atribucion_deterioro_MSE.csv')

def m6_figsave(name):
    plt.tight_layout(); plt.savefig(M6_DIR/'graficas'/name,dpi=150,bbox_inches='tight'); plt.show(); plt.close()

fig,axes=plt.subplots(1,2,figsize=(13,4))
for label,p in M6_PRED.items():
    axes[0].plot(range(1,5),np.abs(p-M6_Y).mean(0),marker='o',label=label)
    axes[1].plot(range(1,5),np.sqrt(((p-M6_Y)**2).mean(0)),marker='o',label=label)
axes[0].set(title='MAE por horizonte · desarrollo 2020',xlabel='Velas M15',ylabel='Precio XAUUSD')
axes[1].set(title='RMSE por horizonte',xlabel='Velas M15',ylabel='Precio XAUUSD'); axes[0].legend()
m6_figsave('01_referencias_por_horizonte.png')
fig,axes=plt.subplots(1,2,figsize=(13,4))
for name in ['clipped','continuous_bypass','oracle_tokens']:
    e=M6_AUDIT[name][:,:,3]-M6_Y
    axes[0].hist(e.ravel(),bins=60,histtype='step',label=name)
    axes[1].plot(range(1,5),abs(e).mean(0),marker='o',label=name)
axes[0].set(title='Reconstrucción con oráculo: errores',xlabel='Error de cierre'); axes[0].legend()
axes[1].set(title='MAE de reconstrucción por horizonte',xlabel='Velas M15'); axes[1].legend()
m6_figsave('02_auditoria_BSQ_clipping.png')
fig,axes=plt.subplots(1,2,figsize=(13,4))
for label in M6_LABELS:
    axes[0].scatter(M6_CE[label].mean((1,2)),abs(M6_PRED[label]-M6_Y).mean(1),s=7,alpha=.3,label=label)
    axes[1].hist((M6_PRED[label]-M6_Y).ravel(),bins=70,histtype='step',label=label)
axes[0].set(title='CE teacher-forced frente a MAE de rollout',xlabel='CE media',ylabel='MAE por ventana'); axes[0].legend()
axes[1].set(title='Distribución del error de precio',xlabel='Predicción menos precio real'); axes[1].legend()
m6_figsave('03_CE_error_y_sesgo.png')
display(M6_BASELINE_TABLE)
display(M6_RECON_TABLE[M6_RECON_TABLE.variable.eq('close')])
display(M6_FACTORIAL)
print('DELTA_MSE_LORA_MENOS_BASE',total_delta)
print('FASE_I_COMPLETA_2020',len(M6_Y),'ventanas; intervenciones con oráculo excluidas de ranking predictivo')
M6_PHASE_I_DONE=True
