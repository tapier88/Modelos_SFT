# M06 Phase IV: fixed-budget causal pilots. Executed only inside Colab.
import io, copy, gc
assert M06_ENVIRONMENT_STABLE and M6_PHASE_III_DONE
M6_IV_DIR=M6_DIR/'fase_IV'
M6_IV_DIR.mkdir(exist_ok=True)
M6_IV_MANIFEST_PATH=M6_IV_DIR/'checksums.json'
M6_IV_MANIFEST=json.loads(M6_IV_MANIFEST_PATH.read_text()) if M6_IV_MANIFEST_PATH.exists() else {}
def iv_verify():
    for rel,v in M6_IV_MANIFEST.items():
        p=M6_IV_DIR/rel
        assert p.is_file() and p.stat().st_size==v['bytes'] and m6_stable.sha(p)==v['sha256'], rel
iv_verify()
def iv_bytes(rel,data):
    p=M6_IV_DIR/rel; old=M6_IV_MANIFEST.get(rel,{}).get('sha256')
    assert not p.exists() or old is not None, 'Untracked artifact: '+rel
    mh=m6_stable.sha(M6_IV_MANIFEST_PATH) if M6_IV_MANIFEST_PATH.exists() else None
    info=m6_stable.atomic_bytes(p,data,expected_sha=old)
    M6_IV_MANIFEST[rel]={'bytes':info['bytes'],'sha256':info['sha256']}
    m6_stable.atomic_json(M6_IV_MANIFEST_PATH,M6_IV_MANIFEST,expected_sha=mh)
def iv_json(rel,obj):
    iv_bytes(rel,json.dumps(m6_stable.json_value(obj),sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode())
def iv_torch(rel,obj):
    b=io.BytesIO();torch.save(obj,b);iv_bytes(rel,b.getvalue())
def iv_frame(rel,df):
    if rel.endswith('.csv'): iv_bytes(rel,df.to_csv(index=False).encode())
    else:
        b=io.BytesIO();df.to_parquet(b,index=False)
        assert len(pd.read_parquet(io.BytesIO(b.getvalue())))==len(df)
        iv_bytes(rel,b.getvalue())
def iv_npz(rel,**kw):
    b=io.BytesIO();np.savez_compressed(b,**kw)
    with np.load(io.BytesIO(b.getvalue()),allow_pickle=False) as z:
        assert all(np.array_equal(z[k],v) for k,v in kw.items())
    iv_bytes(rel,b.getvalue())
def iv_fig(name):
    b=io.BytesIO();plt.tight_layout();plt.savefig(b,format='png',dpi=150,bbox_inches='tight')
    iv_bytes('graficas/'+name,b.getvalue());plt.show();plt.close()

# Fixed comparison: all fresh adapters, identical training windows/order and optimizer.
# Feed-forward ranks are set from actual shapes before any pilot metrics are observed.
M6_IV_BASE_FULL_HASH=m6_hash(m6_model)
_plain=copy.deepcopy(m6_model).requires_grad_(False).eval()
for name in M6_TARGETS:
    parent,leaf=name.rsplit('.',1);setattr(_plain.get_submodule(parent),leaf,_plain.get_submodule(name).base)
assert m6_hash(_plain)==M6_BASE_HASH
iv_attn=list(M6_TARGETS)
iv_ff=[n+'.'+leaf for n,m in _plain.named_modules() if type(m).__name__=='FeedForward' for leaf in ['w1','w2','w3']]
assert len(iv_ff)==24 and len(iv_attn)==18
def ranks_for(names,budget):
    costs=[_plain.get_submodule(n).in_features+_plain.get_submodule(n).out_features for n in names]
    assert len(set(costs))==1
    total=round(budget/costs[0]);q,r=divmod(total,len(names));assert q>=1
    return {n:q+(i<r) for i,n in enumerate(names)}
qv={n:8 for n in iv_attn}
ff=ranks_for(iv_ff,147456)
combo={**{n:4 for n in iv_attn},**ranks_for(iv_ff,73728)}
IV_VARIANTS=[{'id':'QV_CE','ranks':qv,'weights':[1.,1.],'aux':0.},
 {'id':'FF_CE','ranks':ff,'weights':[1.,1.],'aux':0.},
 {'id':'QV_FF_CE','ranks':combo,'weights':[1.,1.],'aux':0.},
 {'id':'QV_CE_s2x2','ranks':qv,'weights':[1.,2.],'aux':0.},
 {'id':'QV_CE_aux','ranks':qv,'weights':[1.,1.],'aux':.1}]
M6_IV_SPEC={'reason':'I: oracle MAE far below predictive error; II: MC does not beat persistence; III: small Huber gains with CI crossing zero. Test limited placement/objective changes.',
 'steps':100,'lr':1e-4,'effective_batch':64,'microbatch':16,'accumulation':4,'seed':271828,
 'dropout':.1,'scale':2.,'optimizer':'AdamW weight_decay 0','clip_grad':1.,
 'validation_monitor':'fixed 124 development windows at steps 0,25,50,75; final all 950 at step100',
 'selection':'fixed final step100; no checkpoint selection by training or reserved evaluation',
 'aux_head':'Head_Huber_s271828 fixed first seed, learned in PhaseIII; frozen; auxiliary Huber normalized returns weight0.1',
 'variants':IV_VARIANTS,'fresh_from_base':True,'original_M05_unchanged':True,
 'independent_evaluation':False,'reserved_data_read':False}
for v in IV_VARIANTS:
    v['trainable_parameters']=sum(r*(_plain.get_submodule(n).in_features+_plain.get_submodule(n).out_features) for n,r in v['ranks'].items())
    assert abs(v['trainable_parameters']/147456-1)<.01
iv_json('configuraciones/protocolo_previo.json',M6_IV_SPEC)
print('PILOTOS_PROTOCOLO_FIJO',[(v['id'],v['trainable_parameters']) for v in IV_VARIANTS],flush=True)

class IVLoRA(nn.Module):
    def __init__(self,base,rank):
        super().__init__();self.base=base.requires_grad_(False)
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5));self.dropout=nn.Dropout(.1)
    def forward(self,x):
        return self.base(x)+2.*F.linear(F.linear(self.dropout(x),self.lora_A),self.lora_B)

iv_head=M6_HEAD_MODELS['Head_Huber_s271828'].requires_grad_(False).eval()
assert M6_CANDIDATE_META['Head_Huber_s271828']['best_step']>0
iv_hmean=torch.tensor(M6_HSCALER.mean_,device=M6_DEVICE,dtype=torch.float32)
iv_hscale=torch.tensor(M6_HSCALER.scale_,device=M6_DEVICE,dtype=torch.float32)
iv_yscale=torch.tensor(M6_Y_SCALE,device=M6_DEVICE)

def iv_adapter(net):
    return {n:p.detach().cpu().clone() for n,p in net.named_parameters() if 'lora_' in n}
def iv_load(net,state):
    dest=dict(net.named_parameters());assert set(state)=={n for n in dest if 'lora_' in n}
    with torch.no_grad():
        for n,x in state.items():dest[n].copy_(x.to(M6_DEVICE))
def iv_continuous(net,b):
    with torch.no_grad():a,c=m6_tok.encode(b['x'][:,:56],half=True)
    h=net.decode_s1(a,c,stamp=b['stamp'][:,:56])[1][:,-1]
    return iv_head((h-iv_hmean)/iv_hscale)[0]
def iv_ce(net,b):
    with torch.no_grad():a,c=m6_tok.encode(b['x'],half=True)
    l1,l2=net(a[:,:-1],c[:,:-1],stamp=b['stamp'][:,:-1],use_teacher_forcing=True,s1_targets=a[:,1:])
    ce1=F.cross_entropy(l1[:,55:].transpose(1,2),a[:,56:],reduction='none')
    ce2=F.cross_entropy(l2[:,55:].transpose(1,2),c[:,56:],reduction='none')
    return torch.stack([ce1,ce2],-1)
@torch.no_grad()
def iv_rollout(net,b):
    a,c=m6_tok.encode(b['x'][:,:56],half=True)
    for j in range(4):
        log,h=net.decode_s1(a,c,stamp=b['stamp'][:,:56+j]);na=log[:,-1].argmax(-1,keepdim=True)
        nc=net.decode_s2(h,torch.cat([a[:,1:],na],1))[:,-1].argmax(-1,keepdim=True)
        a=torch.cat([a,na],1);c=torch.cat([c,nc],1)
    return (m6_tok.decode([a,c],half=True)[:,-4:]*b['scale'][:,None]+b['mean'][:,None])[:,:,3]
@torch.no_grad()
def iv_eval(net,ids,aux=False):
    net.eval();pred=[];ces=[];continuous=[];m6_sync();t=time.perf_counter()
    for offset in range(0,len(ids),16):
        b=m6_batch(m6_dev,ids[offset:offset+16]);pred.append(iv_rollout(net,b).cpu().numpy())
        ces.append(iv_ce(net,b).cpu().numpy())
        if aux:
            ret=(iv_continuous(net,b)*iv_yscale).cpu().numpy().astype(np.float64)
            continuous.append(b['last_close'].cpu().numpy()[:,None]*(1+ret/1e4))
    m6_sync()
    return np.concatenate(pred).astype(np.float64),np.concatenate(ces),None if not aux else np.concatenate(continuous),time.perf_counter()-t

M6_IV_ROWS=[];M6_IV_CURVES=[];M6_IV_CIS=[];M6_IV_PRED={}
for v in IV_VARIANTS:
    name='Pilot_'+v['id']+'_s271828';folder='checkpoints/'+name
    done=M6_IV_DIR/(folder+'/complete.json')
    if done.exists():
        packet=json.loads(done.read_text());M6_IV_ROWS.extend(packet['rows']);M6_IV_CURVES.extend(packet['curves'])
        for row in packet['rows']:
            f=pd.read_parquet(M6_IV_DIR/'metricas'/(row['model']+'.parquet'))
            p=f.pivot(index='window',columns='horizon',values='prediction').sort_index().to_numpy()
            assert p.shape==(950,4) and abs(m6_metrics(p)['MAE']-row['MAE'])<1e-10
            M6_PRED[row['model']]=p;M6_IV_PRED[row['model']]=p
        print('PILOTO_RECUPERADO',name,flush=True);continue
    m6_seed(271828);net=copy.deepcopy(_plain)
    for n,r in v['ranks'].items():
        par,leaf=n.rsplit('.',1);setattr(net.get_submodule(par),leaf,IVLoRA(net.get_submodule(n),r))
    params=[p for p in net.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params)==v['trainable_parameters']
    assert m6_hash(net,True)==M6_BASE_HASH
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=0.)
    rng=np.random.default_rng(271828);begin=0;curves=[];aux_grad=None
    resume=M6_IV_DIR/(folder+'/resume.pt')
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        assert state['variant']==v and state['base_hash']==M6_BASE_HASH
        iv_load(net,state['adapter']);optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng']);begin=state['step'];curves=state['curves'];aux_grad=state['aux_grad']
    if begin==0:
        net.eval();probe=m6_batch(m6_dev,[0,1]);assert np.array_equal(iv_rollout(net,probe).cpu().numpy(),M6_PRED['Kronos_base'][:2])
        alternate={k:x.clone() for k,x in probe.items()};alternate['x'][:,56:]*=-3;alternate['truth'].zero_()
        assert torch.equal(iv_continuous(net,probe),iv_continuous(net,alternate))
        p,c,pa,secs=iv_eval(net,np.asarray(M6_MC_IDS),bool(v['aux']))
        curves.append({'model':name,'step':0,'dev_MAE':float(abs(p-M6_Y[M6_MC_IDS]).mean()),'dev_CE':float(c.mean()),'train_loss':None})
    tstart=time.perf_counter()
    for step in range(begin+1,101):
        net.train();optimizer.zero_grad(set_to_none=True);losses=[]
        ids=rng.integers(0,len(m6_train),64)
        for chunk in range(4):
            b=m6_batch(m6_train,ids[chunk*16:(chunk+1)*16]);ce=iv_ce(net,b)
            objective=(v['weights'][0]*ce[:,:,0].mean()+v['weights'][1]*ce[:,:,1].mean())/sum(v['weights'])
            if v['aux']:
                mu=iv_continuous(net,b);target=1e4*(b['truth'][:,:,3]/b['last_close'][:,None]-1)/iv_yscale
                aux=F.smooth_l1_loss(mu,target)
                if step==1 and chunk==0:
                    grad=torch.autograd.grad(aux,params,retain_graph=True,allow_unused=True)
                    aux_grad=float(sum(g.detach().square().sum() for g in grad if g is not None).sqrt())
                    assert math.isfinite(aux_grad) and aux_grad>0, 'Auxiliary path must have nonzero gradients'
                objective=objective+v['aux']*aux
            assert torch.isfinite(objective);(objective/4).backward();losses.append(float(objective.detach()))
        norm=float(nn.utils.clip_grad_norm_(params,1.));assert math.isfinite(norm);optimizer.step()
        if step%25==0:
            ids_eval=np.arange(950) if step==100 else np.asarray(M6_MC_IDS)
            p,c,pa,secs=iv_eval(net,ids_eval,bool(v['aux']))
            row={'model':name,'step':step,'train_loss':float(np.mean(losses)),'grad_norm':norm,
                 'dev_MAE':float(abs(p-M6_Y[ids_eval]).mean()),'dev_CE':float(c.mean()),'dev_windows':len(ids_eval)}
            curves.append(row)
            iv_torch(folder+'/resume.pt',{'step':step,'variant':v,'base_hash':M6_BASE_HASH,'adapter':iv_adapter(net),
                'optimizer':optimizer.state_dict(),'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all(),'curves':curves,'aux_grad':aux_grad})
            iv_frame('metricas/'+name+'_curves.csv',pd.DataFrame(curves))
            print('PILOTO',name,'paso',step,'CE',row['dev_CE'],'MAE',row['dev_MAE'],'n',len(ids_eval),flush=True)
    net.zero_grad(set_to_none=True)
    assert m6_hash(net,True)==M6_BASE_HASH and m6_hash(m6_model)==M6_IV_BASE_FULL_HASH and m6_hash(m6_tok)==M6_TOK_HASH
    result={'model':name,'seed':271828,'variant':v['id'],'steps':100,'trainable_parameters':v['trainable_parameters'],
            'CE':float(c.mean()),'CE_s1':float(c[:,:,0].mean()),'CE_s2':float(c[:,:,1].mean()),
            'seconds_this_call':time.perf_counter()-tstart,'evaluation_seconds_per_window':secs/950,
            'timing_scope':'greedy + diagnostic CE, plus continuous head if aux','aux_gradient_norm':aux_grad,**m6_metrics(p)}
    rows=[result];preds={name:p}
    if pa is not None:
        aname=name+'_continuous';preds[aname]=pa
        rows.append({**result,'model':aname,**m6_metrics(pa),'output':'frozen Huber head with adapted representations'})
    iv_npz(folder+'/evaluation.npz',greedy=p,CE=c,ids=np.arange(950),**({'continuous':pa} if pa is not None else {}))
    for label,pred in preds.items():
        M6_PRED[label]=pred;M6_IV_PRED[label]=pred
        iv_frame('metricas/'+label+'.parquet',m6_prediction_frame(label,pred))
    iv_torch(folder+'/adapter_final.pt',{'adapter':iv_adapter(net),'variant':v,'base_hash':M6_BASE_HASH,'tokenizer_hash':M6_TOK_HASH,'step':100,'seed':271828})
    iv_json(folder+'/complete.json',{'rows':rows,'curves':curves,'fixed_final_step':100,'base_unchanged':True,'aux_gradient_norm':aux_grad})
    M6_IV_ROWS.extend(rows);M6_IV_CURVES.extend(curves)
    iv_frame('metricas/pilotos_resultados_2020.csv',pd.DataFrame(M6_IV_ROWS))
    del net,optimizer,params;gc.collect();torch.cuda.empty_cache()
for row in M6_IV_ROWS:
    label=row['model'];M6_CANDIDATE_META[label]={'family':'Pilot_'+row['variant'],'seed':271828,'eligible':True,'selection':'fixed step100; single-seed pilot'}
    for ref in ['Persistencia','Kronos_base']:
        M6_IV_CIS.append({'candidate':label,'reference':ref,**m6_paired_ci(M6_PRED[label],M6_PRED[ref])})
M6_IV_TABLE=pd.DataFrame(M6_IV_ROWS)
iv_frame('metricas/pilotos_resultados_2020.csv',M6_IV_TABLE)
iv_frame('metricas/pilotos_intervalos.csv',pd.DataFrame(M6_IV_CIS))
iv_json('configuraciones/candidatos.json',M6_CANDIDATE_META)
fig,axes=plt.subplots(1,2,figsize=(14,5))
for name,g in pd.DataFrame(M6_IV_CURVES).groupby('model'):
    # step100 is all950, earlier monitors are124. Explicit separate markers.
    axes[0].plot(g.step,g.dev_CE,marker='o',label=name.replace('Pilot_',''))
    axes[1].plot(g.step,g.dev_MAE,marker='o',label=name.replace('Pilot_',''))
axes[0].set(title='CE dev: 124 ventanas; paso100: 950',xlabel='Paso',ylabel='CE')
axes[1].set(title='MAE dev: 124 ventanas; paso100: 950',xlabel='Paso',ylabel='MAE');axes[1].legend(fontsize=7)
iv_fig('07_pilotos_LoRA.png')
display(M6_IV_TABLE);display(pd.DataFrame(M6_IV_CIS))
iv_verify();assert m6_stable.inventory()==json.loads((m6_stable.OPS/'artifact_manifest_before_recovery.json').read_text())
iv_json('phase_status.json',{'complete':True,'pilots':5,'candidate_outputs':len(M6_IV_ROWS),'steps_each':100,
 'original_base_tokenizer_M05_unchanged':True,'test_read':False,'orders_enabled':False})
M6_PHASE_IV_DONE=True
del _plain;gc.collect();torch.cuda.empty_cache()
print('FASE_IV_COMPLETA',len(M6_IV_ROWS),'salidas; cinco pilotos controlados, ninguna evaluacion reservada')
