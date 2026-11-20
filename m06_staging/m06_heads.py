#@title MISIÓN 06 · 5. Fase III: representaciones congeladas y regresión continua
assert globals().get('M6_PHASE_II_DONE') is True
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
m6_mode('Kronos_base')
assert not any(p.requires_grad for p in m6_model.parameters())
# La recuperación anterior terminó sus comprobaciones; se corrige únicamente la
# serialización de NaN de la tabla (persistencia sin dirección/tiempo medido).
if 'M6_RECOVERY' in globals():
    M6_RECOVERY['recovered_baselines']=json.loads(M6_BASELINE_TABLE.to_json(orient='records'))
    m6_save_json('auditoria/recuperacion_runtime.json',M6_RECOVERY)
M6_HEAD_SPEC={'hidden':64,'activation':'GELU','output':'retornos acumulados en bps H1-H4',
 'feature':'ultimo estado normalizado de decode_s1; solo contexto 56; Kronos base',
 'scalers':'media/std features y std retornos ajustados exclusivamente en train',
 'losses':M6_PLAN['head_losses'],'steps':300,'batch':64,'lr':.001,'weight_decay':0.,
 'seeds':M6_PLAN['head_seeds'],'checkpoint_selection':'min MAE 2020 entre pasos 0,50,...,300',
 'zero_initialized_location':True,'ridge_alpha':100.,'baseline_features':'retornos rezagados, volatilidad, rango y cuerpo; solo contexto',
 'interpretation':'exploracion; seleccion reutiliza 2020; ninguna evaluacion independiente'}
m6_save_json('configuraciones/cabezas_continuas.json',M6_HEAD_SPEC)
M6_FEATURE_ID=hashlib.sha256(json.dumps({'base':M6_BASE_HASH,'tokenizer':M6_TOK_HASH,
 'data':M6_SOURCE_MANIFEST,'spec':M6_HEAD_SPEC},sort_keys=True).encode()).hexdigest()[:16]
M6_FDIR=M6_DIR/'checkpoints/fase_III'/M6_FEATURE_ID; M6_FDIR.mkdir(parents=True,exist_ok=True)

def m6_raw_features(ds,ids):
    raw=np.stack([ds.values[int(ds.starts[i]):int(ds.starts[i])+56,:4] for i in ids]).astype(np.float64)
    close=raw[:,:,3]; last=close[:,-1]
    lag=np.column_stack([1e4*(last/close[:,-1-k]-1) for k in [1,2,4,8,16,32,55]])
    changes=1e4*(close[:,1:]/close[:,:-1]-1)
    vol=np.column_stack([changes[:,-k:].std(1) for k in [4,16,55]])
    ranges=1e4*np.column_stack([(raw[:,:,1]-raw[:,:,2]).mean(1)/last,(raw[:,-1,3]-raw[:,-1,0])/last])
    return np.column_stack([lag,vol,ranges]).astype(np.float32)

def m6_cache_features(ds,split):
    pieces=[]
    for start in range(0,len(ds),256):
        path=M6_FDIR/f'features_{split}_{start:05d}.npz'
        ids=np.arange(start,min(start+256,len(ds)))
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                part={k:z[k] for k in z.files}; assert np.array_equal(part['ids'],ids)
        else:
            hs=[]; ys=[]; lasts=[]; secs=0.
            for offset in range(0,len(ids),64):
                b=m6_batch(ds,ids[offset:offset+64]); m6_sync(); t=time.perf_counter()
                h=m6_features(b); m6_sync(); secs+=time.perf_counter()-t
                hs.append(h.cpu().numpy()); ys.append(b['truth'][:,:,3].cpu().numpy()); lasts.append(b['last_close'].cpu().numpy())
            part={'ids':ids,'hidden':np.concatenate(hs),'truth':np.concatenate(ys),
                  'last':np.concatenate(lasts),'raw':m6_raw_features(ds,ids),'seconds':np.array([secs])}
            assert all(np.isfinite(v).all() for v in part.values())
            np.savez_compressed(path,**part)
        pieces.append(part)
        if start%2048==0: print('REPRESENTACIONES_CAUSALES',split,start+len(ids),'/',len(ds),flush=True)
    return {k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}

M6_FTRAIN=m6_cache_features(m6_train,'train_2010_2019')
M6_FDEV=m6_cache_features(m6_dev,'development_2020')
assert np.array_equal(M6_FDEV['truth'].astype(np.float64),M6_Y)
M6_TRETURN=1e4*(M6_FTRAIN['truth'].astype(np.float64)/M6_FTRAIN['last'][:,None]-1)
M6_DRETURN=1e4*(M6_Y/M6_LAST[:,None]-1)
M6_HSCALER=StandardScaler().fit(M6_FTRAIN['hidden'])
M6_RSCALER=StandardScaler().fit(M6_FTRAIN['raw'])
M6_Y_SCALE=np.maximum(M6_TRETURN.std(0),1e-3).astype(np.float32)
np.savez_compressed(M6_FDIR/'train_only_scalers.npz',hidden_mean=M6_HSCALER.mean_,hidden_scale=M6_HSCALER.scale_,
 raw_mean=M6_RSCALER.mean_,raw_scale=M6_RSCALER.scale_,return_scale=M6_Y_SCALE)
M6_HEAD_ROWS=[]; M6_HEAD_CIS=[]; M6_HEAD_CURVES=[]; M6_HEAD_MODELS={}

for tag,scaler,key in [('Ridge_contexto',M6_RSCALER,'raw'),('Ridge_Kronos_congelado',M6_HSCALER,'hidden')]:
    x=scaler.transform(M6_FTRAIN[key]); xd=scaler.transform(M6_FDEV[key])
    ridge=Ridge(alpha=100.).fit(x,M6_TRETURN/M6_Y_SCALE)
    m6_sync(); t=time.perf_counter(); ret=ridge.predict(xd)*M6_Y_SCALE; seconds=time.perf_counter()-t
    pred=M6_LAST[:,None]*(1+ret/1e4); M6_PRED[tag]=pred
    M6_TIMES[tag]=seconds/len(m6_dev)+(float(M6_FDEV['seconds'].sum())/len(m6_dev) if key=='hidden' else 0.)
    M6_CANDIDATE_META[tag]={'family':tag,'seed':None,'eligible':True,'selection':'alpha fijo 100, sin busqueda'}
    np.savez_compressed(M6_FDIR/(tag+'.npz'),coef=ridge.coef_,intercept=ridge.intercept_,alpha=100.)
    M6_HEAD_ROWS.append({'model':tag,'seed':None,'best_step':None,**m6_metrics(pred)})
    m6_prediction_frame(tag,pred).to_parquet(M6_DIR/'metricas'/(tag+'.parquet'),index=False)

class M6ReturnHead(nn.Module):
    def __init__(self,probabilistic=False):
        super().__init__(); self.probabilistic=probabilistic
        self.hidden=nn.Linear(512,64); self.output=nn.Linear(64,8 if probabilistic else 4)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
    def forward(self,x):
        out=self.output(F.gelu(self.hidden(x)))
        if self.probabilistic: return out[:,:4],F.softplus(out[:,4:])+1e-4
        return out,None

hx=torch.tensor(M6_HSCALER.transform(M6_FTRAIN['hidden']),dtype=torch.float32,device=M6_DEVICE)
hd=torch.tensor(M6_HSCALER.transform(M6_FDEV['hidden']),dtype=torch.float32,device=M6_DEVICE)
hy=torch.tensor(M6_TRETURN/M6_Y_SCALE,dtype=torch.float32,device=M6_DEVICE)
hydev=torch.tensor(M6_DRETURN/M6_Y_SCALE,dtype=torch.float32,device=M6_DEVICE)
fixed_train=np.linspace(0,len(m6_train)-1,512,dtype=int)

@torch.no_grad()
def m6_head_predict(net,x,last):
    net.eval(); location,scale=net(x)
    ret=location.cpu().numpy().astype(np.float64)*M6_Y_SCALE
    price=last[:,None]*(1+ret/1e4)
    sigma=None if scale is None else scale.cpu().numpy().astype(np.float64)*M6_Y_SCALE*last[:,None]/1e4
    return price,sigma

def m6_head_loss(net,x,y):
    mu,b=net(x)
    return F.smooth_l1_loss(mu,y) if b is None else (torch.abs(y-mu)/b+torch.log(2*b)).mean()

for loss in M6_PLAN['head_losses']:
    for seed in M6_PLAN['head_seeds']:
        name=f'Head_{loss}_s{seed}'; m6_seed(seed)
        head=M6ReturnHead(loss=='Laplace_NLL').to(M6_DEVICE)
        optimizer=torch.optim.Adam(head.parameters(),lr=.001)
        rng=np.random.default_rng(seed); checkpoint=M6_FDIR/(name+'_resume.pt')
        best=float('inf'); best_step=0; best_state=None; curves=[]; begin=0
        if checkpoint.exists():
            state=torch.load(checkpoint,map_location=M6_DEVICE,weights_only=False)
            assert state['feature_id']==M6_FEATURE_ID
            head.load_state_dict(state['head']); optimizer.load_state_dict(state['optimizer'])
            rng.bit_generator.state=state['numpy_rng']; begin=state['step']; best=state['best_MAE']
            best_step=state['best_step']; best_state=state['best_state']; curves=state['curves']
        for step in range(begin,301):
            if step>begin:
                head.train(); idx=rng.integers(0,len(hx),64)
                optimizer.zero_grad(set_to_none=True); objective=m6_head_loss(head,hx[idx],hy[idx])
                assert torch.isfinite(objective); objective.backward()
                nn.utils.clip_grad_norm_(head.parameters(),1.); optimizer.step()
            if step%50!=0 or (step==begin and curves): continue
            p,scale=m6_head_predict(head,hd,M6_LAST)
            pt,_=m6_head_predict(head,hx[fixed_train],M6_FTRAIN['last'][fixed_train])
            val=m6_metrics(p)['MAE']; train_mae=float(abs(pt-M6_FTRAIN['truth'][fixed_train]).mean())
            with torch.no_grad():
                train_obj=float(m6_head_loss(head,hx[fixed_train],hy[fixed_train])); dev_obj=float(m6_head_loss(head,hd,hydev))
            curves.append({'model':name,'loss':loss,'seed':seed,'step':step,'train_MAE':train_mae,'dev_MAE':val,
                           'train_objective':train_obj,'dev_objective':dev_obj})
            if val<best:
                best=val; best_step=step; best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
            torch.save({'step':step,'feature_id':M6_FEATURE_ID,'head':head.state_dict(),'optimizer':optimizer.state_dict(),
             'numpy_rng':rng.bit_generator.state,'best_MAE':best,'best_step':best_step,'best_state':best_state,'curves':curves},checkpoint)
        head.load_state_dict(best_state); m6_sync(); t=time.perf_counter()
        pred,scale=m6_head_predict(head,hd,M6_LAST); m6_sync(); secs=time.perf_counter()-t
        M6_PRED[name]=pred; M6_HEAD_MODELS[name]=head
        M6_TIMES[name]=secs/len(m6_dev)+float(M6_FDEV['seconds'].sum())/len(m6_dev)
        M6_CANDIDATE_META[name]={'family':'Head_'+loss,'seed':seed,'eligible':True,'best_step':best_step}
        stats={'model':name,'seed':seed,'best_step':best_step,'trainable_parameters':sum(p.numel() for p in head.parameters()),**m6_metrics(pred)}
        if scale is not None:
            stats['coverage80']=float(np.mean(abs(pred-M6_Y)<=-np.log(.2)*scale))
            stats['coverage90']=float(np.mean(abs(pred-M6_Y)<=-np.log(.1)*scale))
            stats['interval90_width']=float((-2*np.log(.1)*scale).mean())
            stats['price_Laplace_NLL']=float((abs(pred-M6_Y)/scale+np.log(2*scale)).mean())
            expected_abs=abs(pred-M6_LAST[:,None])+scale*np.exp(-abs(pred-M6_LAST[:,None])/scale)
            stats['expected_magnitude_MAE']=float(abs(expected_abs-abs(M6_Y-M6_LAST[:,None])).mean())
            np.savez_compressed(M6_FDIR/(name+'_uncertainty.npz'),location=pred,laplace_scale=scale)
        M6_HEAD_ROWS.append(stats); M6_HEAD_CURVES.extend(curves)
        torch.save({'head':best_state,'spec':M6_HEAD_SPEC,'loss':loss,'seed':seed,'best_step':best_step,
                    'feature_id':M6_FEATURE_ID,'base_hash':M6_BASE_HASH,'tokenizer_hash':M6_TOK_HASH},M6_FDIR/(name+'_best.pt'))
        m6_prediction_frame(name,pred).to_parquet(M6_DIR/'metricas'/(name+'.parquet'),index=False)
        pd.DataFrame(M6_HEAD_ROWS).to_csv(M6_DIR/'metricas/cabezas_resultados_2020.csv',index=False)
        pd.DataFrame(M6_HEAD_CURVES).to_csv(M6_DIR/'metricas/cabezas_curvas.csv',index=False)
        print('CABEZA_FINALIZADA',name,'MAE',stats['MAE'],'paso',best_step,flush=True)
for row in M6_HEAD_ROWS:
    for ref in ['Persistencia','Kronos_base']:
        M6_HEAD_CIS.append({'candidate':row['model'],'reference':ref,**m6_paired_ci(M6_PRED[row['model']],M6_PRED[ref])})
M6_HEAD_TABLE=pd.DataFrame(M6_HEAD_ROWS)
pd.DataFrame(M6_HEAD_CIS).to_csv(M6_DIR/'metricas/cabezas_intervalos.csv',index=False)
m6_save_json('configuraciones/candidatos.json',M6_CANDIDATE_META)
fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,loss in zip(axes,M6_PLAN['head_losses']):
    for name,g in pd.DataFrame(M6_HEAD_CURVES).groupby('model'):
        if loss not in name: continue
        ax.plot(g.step,g.dev_MAE,label=name.replace('Head_',''))
        ax.plot(g.step,g.train_MAE,linestyle=':',alpha=.6)
    ax.axhline(m6_metrics(M6_PRED['Persistencia'])['MAE'],color='black',linestyle='--',label='Persistencia dev')
    ax.set(title=loss+' · sólido dev / punteado train',xlabel='Pasos',ylabel='MAE precio'); ax.legend(fontsize=7)
m6_figsave('06_cabezas_entrenamiento_validacion.png')
display(M6_HEAD_TABLE); display(pd.DataFrame(M6_HEAD_CIS))
assert m6_hash(m6_model,True)==M6_BASE_HASH and m6_hash(m6_tok)==M6_TOK_HASH
assert not any(p.requires_grad or p.grad is not None for p in m6_model.parameters())
M6_PHASE_III_DONE=True
print('FASE_III_COMPLETA: Kronos y adaptadores congelados; 2020 usado para desarrollo, no holdout')
