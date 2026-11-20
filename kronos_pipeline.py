"""Read-only source preparation and verified Kronos SFT. Python 3.10+."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, random, sys, time, types
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'vendor'))
sys.path.insert(0, str(ROOT/'deps'))
COMMIT = '67b630e67f6a18c9e9be918d9b4337c960db1e9a'
REVISIONS = {'Kronos-small':'901c26c1332695a2a8f243eb2f37243a37bea320',
             'Kronos-Tokenizer-base':'0e0117387f39004a9016484a186a908917e22426'}
CONFIG = {'M5':(5,128,12), 'M15':(15,64,4)}
EXPECTED = {'M5': {'rows':1046748,'train':308635,'validation':102155,'test':60354},
            'M15':{'rows':346565,'train':51606,'validation':18168,'test':10522}}
COLS=['open','high','low','close','volume','amount']
EPS, CLIP = 1e-5, 5.0

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def prepare(source, output, reference=None):
    source,output=Path(source).resolve(),Path(output).resolve()
    assert source!=output and source not in output.parents, 'Output must not be inside source.'
    output.mkdir(parents=True,exist_ok=True)
    manifest=[]; summary={}; frames={}; indices={}
    refs={x['file']:x['sha256'] for x in (reference or {}).get('files',[])}
    for tf,(step,L,H) in CONFIG.items():
        paths=sorted(source.glob(f'XAUUSD_{tf}_BID_*.parquet'))
        assert len(paths)==15, f'{tf}: expected 15 files, found {len(paths)}'
        parts=[]
        for path in paths:
            digest=sha(path)
            if refs: assert digest==refs[path.name], f'Changed source: {path.name}'
            d=pd.read_parquet(path)
            required=['timestamp_utc','open','high','low','close','volume','m1_candles']
            assert list(d.columns)==required, f'Unexpected schema: {path.name}'
            t=pd.to_datetime(d.timestamp_utc,utc=True)
            a=d[COLS[:5]].to_numpy()
            assert not d.isna().any().any() and np.isfinite(a).all()
            assert t.is_monotonic_increasing and not t.duplicated().any()
            assert ((t.astype('int64')%(step*60*10**9))==0).all()
            assert (d.m1_candles==step).all() and (d.volume>=0).all()
            assert ((d.low>0)&(d.low<=d[['open','close']].min(axis=1))&
                    (d.high>=d[['open','close']].max(axis=1))&(d.high>=d.low)).all()
            d=d.copy(); d['timestamp_utc']=t; d['source_file']=path.name
            parts.append(d)
            manifest.append({'file':path.name,'sha256':digest,'bytes':path.stat().st_size,
                             'rows':len(d),'timeframe':tf})
        df=pd.concat(parts,ignore_index=True); t=df.timestamp_utc
        assert t.is_monotonic_increasing and not t.duplicated().any()
        assert (t.dt.year>=2010).all() and (t<=pd.Timestamp('2024-10-18 20:59Z')).all()
        split=np.select([t.dt.year<=2019,t.dt.year<=2022],['train','validation'],default='test')
        boundary=t.diff().ne(pd.Timedelta(minutes=step)) | pd.Series(split).ne(pd.Series(split).shift())
        segment=boundary.cumsum().to_numpy()
        df['split']=split; df['segment_id']=segment
        chunks=[]
        for seg, group in df.groupby('segment_id',sort=False):
            first,last=int(group.index[0]),int(group.index[-1])
            starts=np.arange(first,last-L-H+2,dtype=np.int64)
            if not len(starts): continue
            chunks.append(pd.DataFrame({'start_row':starts,'segment_id':int(seg),'split':split[first]}))
        assert chunks, f'No valid windows for {tf}'
        ix=pd.concat(chunks,ignore_index=True); starts=ix.start_row.to_numpy()
        ix['timeframe']=tf; ix['context_length']=L; ix['prediction_length']=H
        ix['context_end_row']=starts+L-1; ix['target_start_row']=starts+L
        ix['target_end_row']=starts+L+H-1
        ix['context_start']=t.iloc[starts].to_numpy()
        ix['decision_time']=t.iloc[starts+L].to_numpy() # preceding candle closes here
        ix['target_end_exclusive']=(t.iloc[starts+L+H-1]+pd.Timedelta(minutes=step)).to_numpy()
        ny=pd.to_datetime(ix.decision_time,utc=True).dt.tz_convert('America/New_York')
        minute=ny.dt.hour*60+ny.dt.minute
        ix['session_date_ny']=ny.dt.strftime('%Y-%m-%d')
        ix['ny_full_horizon']=(minute>=510)&(minute+H*step<=780)
        ix['source_start']=df.source_file.iloc[starts].to_numpy()
        ix['source_end']=df.source_file.iloc[starts+L+H-1].to_numpy()
        ix['quality_reason']='complete_contiguous'; ix['amount_available']=False
        assert np.array_equal(segment[starts],segment[starts+L+H-1])
        assert np.array_equal(split[starts],split[starts+L+H-1])
        counts={s:int((ix.split==s).sum()) for s in ['train','validation','test']}
        assert len(df)==EXPECTED[tf]['rows'] and all(counts[s]==EXPECTED[tf][s] for s in counts), (tf,counts)
        ix.to_parquet(output/f'windows_{tf}.parquet',index=False)
        summary[tf]={'rows':len(df),'windows':counts,'segments':int(segment[-1]),
                     'ny_windows':ix.groupby('split').ny_full_horizon.sum().astype(int).to_dict(),
                     'context':L,'horizon':H,'last_timestamp':str(t.iloc[-1])}
        frames[tf]=df; indices[tf]=ix
    for record in manifest: assert sha(source/record['file'])==record['sha256']
    write_json(output/'source_manifest.json',manifest)
    write_json(output/'preparation.json',summary)
    return frames,indices,summary,manifest

def normalize(x, L):
    """Same float32 context statistics used during training and forecasting."""
    x=np.asarray(x,dtype=np.float32)
    mean=x[:L].mean(axis=0); scale=x[:L].std(axis=0)+EPS
    return np.clip((x-mean)/scale,-CLIP,CLIP),mean,scale

class WindowDataset(Dataset):
    def __init__(self, df, ix, tf, split, ny_only=False):
        self.step,self.L,self.H=CONFIG[tf]
        rows=ix[ix.split==split]
        if ny_only: rows=rows[rows.ny_full_horizon]
        self.starts=rows.start_row.to_numpy(dtype=np.int64)
        if not len(self.starts): raise ValueError(f'Empty dataset: {tf}/{split}')
        self.values=np.column_stack([df[COLS[:5]].to_numpy(dtype=np.float32),np.zeros(len(df),np.float32)])
        t=df.timestamp_utc.dt
        self.stamp=np.column_stack([t.minute,t.hour,t.weekday,t.day,t.month]).astype(np.float32)
    def __len__(self): return len(self.starts)
    def __getitem__(self,i):
        s=int(self.starts[i]); x=self.values[s:s+self.L+self.H].copy()
        norm,mean,scale=normalize(x,self.L)
        loss_mask=np.arange(self.L+self.H-1)>=self.L-1
        return {'x':torch.from_numpy(norm),'stamp':torch.from_numpy(self.stamp[s:s+self.L+self.H].copy()),
                'loss_mask':torch.from_numpy(loss_mask),'mean':torch.from_numpy(mean),
                'scale':torch.from_numpy(scale),'truth':torch.from_numpy(x[self.L:].copy()),
                'last_close':torch.tensor(x[self.L-1,3])}

def causal_cross(self,query,key,value,key_padding_mask=None):
    """Equal-length, complete sequences; causal in train AND eval. Weights unchanged."""
    if key_padding_mask is not None: raise ValueError('Padding unsupported in fixed-window adapter')
    B,T,_=query.shape
    if key.shape[:2]!=(B,T) or value.shape[:2]!=(B,T):
        raise ValueError('Use full-length aligned s1 targets for conditional decoding')
    q=self.q_proj(query).view(B,T,self.n_heads,self.head_dim).transpose(1,2)
    k=self.k_proj(key).view(B,T,self.n_heads,self.head_dim).transpose(1,2)
    v=self.v_proj(value).view(B,T,self.n_heads,self.head_dim).transpose(1,2)
    q,k=self.rotary(q,k)
    y=F.scaled_dot_product_attention(q,k,v,dropout_p=self.attn_dropout_p if self.training else 0.,is_causal=True)
    y=y.transpose(1,2).contiguous().view(B,T,self.d_model)
    return self.resid_dropout(self.out_proj(y))

def load_models(weights,device):
    from model import Kronos,KronosTokenizer
    model=Kronos.from_pretrained(str(Path(weights)/'Kronos-small')).to(device)
    tokenizer=KronosTokenizer.from_pretrained(str(Path(weights)/'Kronos-Tokenizer-base')).to(device)
    model.dep_layer.cross_attn.forward=types.MethodType(causal_cross,model.dep_layer.cross_attn)
    tokenizer.eval().requires_grad_(False)
    return model,tokenizer

def to_device(batch,device): return {k:v.to(device) for k,v in batch.items()}

def loss_for_batch(model,tokenizer,batch):
    with torch.no_grad(): a,b=tokenizer.encode(batch['x'],half=True)
    l1,l2=model(a[:,:-1],b[:,:-1],stamp=batch['stamp'][:,:-1],
                use_teacher_forcing=True,s1_targets=a[:,1:])
    mask=batch['loss_mask']; assert mask.any() and mask.shape==a[:,1:].shape
    return .5*(F.cross_entropy(l1[mask],a[:,1:][mask])+F.cross_entropy(l2[mask],b[:,1:][mask]))

@torch.no_grad()
def rollout(model,tokenizer,batch,L,H):
    """Deterministic greedy baseline. No target values are read here."""
    model.eval(); tokenizer.eval()
    a,b=tokenizer.encode(batch['x'][:,:L],half=True)
    for j in range(H):
        logits,hidden=model.decode_s1(a,b,stamp=batch['stamp'][:,:L+j])
        next_a=logits[:,-1].argmax(-1,keepdim=True)
        conditioned=torch.cat([a[:,1:],next_a],dim=1)
        next_b=model.decode_s2(hidden,conditioned)[:,-1].argmax(-1,keepdim=True)
        a=torch.cat([a,next_a],dim=1); b=torch.cat([b,next_b],dim=1)
    return tokenizer.decode([a,b],half=True)[:,-H:]*batch['scale'][:,None]+batch['mean'][:,None]

def validate(model,tokenizer,dataset,device,batch_size=2,max_batches=8):
    model.eval(); total_loss=0.; n=0; absolute=0.; baseline=0.; points=0
    # evenly spaced validation examples; this is a fixed subset when max_batches is finite
    ids=np.linspace(0,len(dataset)-1,min(len(dataset),max_batches*batch_size),dtype=int)
    loader=DataLoader(torch.utils.data.Subset(dataset,ids),batch_size=batch_size,shuffle=False,
                      generator=torch.Generator().manual_seed(314))
    with torch.no_grad():
        for batch in loader:
            batch=to_device(batch,device); loss=loss_for_batch(model,tokenizer,batch)
            pred=rollout(model,tokenizer,batch,dataset.L,dataset.H)
            y=batch['truth'][:,:,3]; B=len(y)
            assert torch.isfinite(pred).all() and torch.isfinite(loss)
            total_loss+=float(loss)*B; n+=B
            absolute+=float((pred[:,:,3]-y).abs().sum())
            baseline+=float((batch['last_close'][:,None]-y).abs().sum()); points+=y.numel()
    return {'teacher_forced_loss':total_loss/n,'rollout_close_mae':absolute/points,
            'persistence_close_mae':baseline/points,'examples':n,'split':'validation',
            'selection':'evenly_spaced_fixed_subset','decoding':'greedy'}

def contract_tests(model,tokenizer,ds,device):
    batch=to_device(next(iter(DataLoader(ds,batch_size=1,generator=torch.Generator().manual_seed(314)))),device)
    L,H=ds.L,ds.H; results={}
    assert batch['loss_mask'].sum().item()==H
    active=torch.where(batch['loss_mask'][0])[0].tolist()
    assert active==list(range(L-1,L+H-1)); results['loss_active_indices']=active
    raw=ds.values[int(ds.starts[0]):int(ds.starts[0])+L+H].copy()
    changed=raw.copy(); changed[L:,:4]*=1.3
    x,m,s=normalize(raw,L); z,m2,s2=normalize(changed,L)
    assert np.array_equal(m,m2) and np.array_equal(s,s2) and np.array_equal(x[:L],z[:L])
    results['normalization_future_invariant']=True
    model.eval(); tokenizer.eval()
    with torch.no_grad():
        aa,bb=tokenizer.encode(batch['x'],half=True)
        changed_x=batch['x'].clone(); changed_x[:,L:]*=-1
        ac,bc=tokenizer.encode(changed_x,half=True)
        assert torch.equal(aa[:,:L],ac[:,:L]) and torch.equal(bb[:,:L],bc[:,:L])
        a0,b0=tokenizer.encode(batch['x'][:,:L],half=True)
        assert torch.equal(aa[:,:L],a0) and torch.equal(bb[:,:L],b0)
        results['tokenizer_future_and_prefix_invariant']=True
        cut=L//2; a=aa[:,:-1]; b=bb[:,:-1]; target=aa[:,1:]
        am=a.clone(); bm=b.clone(); tm=target.clone()
        am[:,cut+1:]=(am[:,cut+1:]+17)%1024; bm[:,cut+1:]=(bm[:,cut+1:]+29)%1024
        tm[:,cut+1:]=(tm[:,cut+1:]+11)%1024
        for training in [False,True]:
            model.train(training); seed_all(123)
            original=model(a,b,stamp=batch['stamp'][:,:-1],use_teacher_forcing=True,s1_targets=target)
            seed_all(123)
            modified=model(am,bm,stamp=batch['stamp'][:,:-1],use_teacher_forcing=True,s1_targets=tm)
            errors=[float((u[:,:cut+1]-v[:,:cut+1]).abs().max()) for u,v in zip(original,modified)]
            assert max(errors)<1e-5, errors
            results['predictor_causal_'+('train' if training else 'eval')]=errors
        model.eval(); p=rollout(model,tokenizer,batch,L,H)
        alternative={k:v.clone() for k,v in batch.items()}
        alternative['x'][:,L:]*=-1; alternative['truth'].zero_()
        p2=rollout(model,tokenizer,alternative,L,H)
        assert torch.equal(p,p2); results['rollout_future_invariant']=True
    results['tokenizer_frozen']=not any(p.requires_grad for p in tokenizer.parameters())
    return results

def train(model,tokenizer,train_ds,val_ds,device,out,steps=3,batch_size=2,epochs=1,resume=None,full=False):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); seed_all()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
    gen=torch.Generator().manual_seed(42)
    start_epoch=0; start_batch=0; history=[]; global_step=0; best=float('inf')
    if resume:
        state=torch.load(resume,map_location=device,weights_only=False)
        assert (state['L'],state['H'],state['batch_size'])==(train_ds.L,train_ds.H,batch_size), 'Resume configuration mismatch'
        assert state['commit']==COMMIT and state['revisions']==REVISIONS, 'Resume model revision mismatch'
        manifest_path=out.parent/'source_manifest.json'
        if manifest_path.exists() and state.get('source_manifest_sha256'):
            assert sha(manifest_path)==state['source_manifest_sha256'], 'Resume source manifest mismatch'
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start_epoch=state['epoch']; start_batch=state['next_batch']; global_step=state['step']
        history=state['history']; best=state['best']; random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng'].cpu())
        if device.type=='cuda' and state.get('cuda_rng'): torch.cuda.set_rng_state_all(state['cuda_rng'])
    else: state=None
    tok_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in tokenizer.parameters())).hexdigest()
    probe=next(model.parameters()).detach().clone(); started=time.time()
    for epoch in range(start_epoch,epochs):
        gen.manual_seed(42+epoch)
        order=torch.randperm(len(train_ds),generator=gen).tolist()
        offset=start_batch if epoch==start_epoch else 0
        loader=DataLoader(torch.utils.data.Subset(train_ds,order[offset*batch_size:]),batch_size=batch_size,shuffle=False,
                          generator=torch.Generator().manual_seed(314+epoch))
        for number,batch in enumerate(loader,start=offset):
            model.train(); batch=to_device(batch,device); optimizer.zero_grad(set_to_none=True)
            loss=loss_for_batch(model,tokenizer,batch); assert torch.isfinite(loss)
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); global_step+=1
            history.append({'step':global_step,'epoch':epoch,'batch':number,'loss':float(loss.detach()),
                            'gradient_norm':float(grad_norm),'seconds':time.time()-started})
            print(json.dumps(history[-1]),flush=True)
            end=(not full and global_step>=steps) or number+1==int(np.ceil(len(train_ds)/batch_size))
            if end or global_step%100==0:
                metrics=validate(model,tokenizer,val_ds,device,batch_size=1,max_batches=2 if not full else 8)
                improved=metrics['rollout_close_mae']<best
                best=min(best,metrics['rollout_close_mae'])
                saved={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                       'next_batch':number+1,'step':global_step,'history':history,'best':best,
                       'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                       'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                       'commit':COMMIT,'revisions':REVISIONS,'L':train_ds.L,'H':train_ds.H,
                       'source_manifest_sha256':sha(out.parent/'source_manifest.json') if (out.parent/'source_manifest.json').exists() else None,
                       'batch_size':batch_size,'normalization':{'epsilon':EPS,'clip':CLIP,'context_only':True},
                       'amount_policy':'explicit_zero','cross_attention':'always_causal_aligned_full_sequence',
                       'validation':metrics,'full_training':full}
                torch.save(saved,out/'last.pt')
                if improved: torch.save(saved,out/'best.pt')
                write_json(out/'history.json',history); write_json(out/'validation.json',metrics)
            if not full and global_step>=steps: break
        if not full and global_step>=steps: break
    assert not torch.equal(probe,next(model.parameters()).detach()), 'Weights did not change'
    assert tok_hash==hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in tokenizer.parameters())).hexdigest()
    # Real checkpoint round-trip, including optimizer state, without modifying source weights.
    batch=to_device(next(iter(DataLoader(val_ds,batch_size=1,generator=torch.Generator().manual_seed(314)))),device)
    before=rollout(model,tokenizer,batch,train_ds.L,train_ds.H)
    ck=torch.load(out/'last.pt',map_location=device,weights_only=False)
    model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
    after=rollout(model,tokenizer,batch,train_ds.L,train_ds.H)
    assert torch.equal(before,after)
    return {'steps':global_step,'weights_changed':True,'tokenizer_unchanged':True,
            'checkpoint_reload_identical':True,'validation':metrics,'seconds':time.time()-started}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--weights',default=str(ROOT/'weights')); parser.add_argument('--reference')
    parser.add_argument('--mode',choices=['prepare','smoke','full'],default='smoke')
    parser.add_argument('--timeframe',choices=['M5','M15','both'],default='both')
    parser.add_argument('--steps',type=int,default=3); parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--epochs',type=int,default=1); parser.add_argument('--resume')
    args=parser.parse_args(); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    if args.resume and args.timeframe=='both': parser.error('Resume one timeframe at a time')
    torch.set_num_threads(min(4,os.cpu_count() or 1)); seed_all()
    ref=json.loads(Path(args.reference).read_text(encoding='utf-8')) if args.reference else None
    frames,indices,prep,manifest=prepare(args.source,output,ref)
    status={'data_verified':True,'windows_prepared':True,'model_loaded':False,'smoke_passed':False,
            'full_training_done':False,'test_evaluated':False,'runtime':platform.platform(),
            'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
            'preparation':prep,'unresolved':['Ticks not integrated; 30 source blocks unresolved',
                'Tick ZIP equivalence, price divisor and volume units unverified',
                'Pretraining cutoff may overlap historical test; not a pristine holdout',
                'No broker execution or trading edge demonstrated']}
    write_json(output/'status.json',status)
    if args.mode=='prepare': return
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    status['device']=str(device); status['precision']='float32'
    selected=list(CONFIG) if args.timeframe=='both' else [args.timeframe]
    status['experiments']={}
    for tf in selected:
        model,tokenizer=load_models(args.weights,device); status['model_loaded']=True
        tr=WindowDataset(frames[tf],indices[tf],tf,'train'); va=WindowDataset(frames[tf],indices[tf],tf,'validation')
        checks=contract_tests(model,tokenizer,tr,device)
        write_json(output/f'tests_{tf}.json',checks)
        result=train(model,tokenizer,tr,va,device,output/tf,args.steps,args.batch_size,args.epochs,
                     args.resume,full=args.mode=='full')
        status['experiments'][tf]={'tests':checks,'training':result}
        write_json(output/'status.json',status)
        del model,tokenizer
    for item in manifest: assert sha(Path(args.source)/item['file'])==item['sha256']
    status['sources_unchanged']=True; status['smoke_passed']=True
    status['full_training_done']=args.mode=='full'
    write_json(output/'status.json',status)
    print(json.dumps(status,indent=2),flush=True)

if __name__=='__main__': main()
