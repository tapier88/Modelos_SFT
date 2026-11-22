# M10_PHASE_V_JOINT_RETURN_V1
"""Explicitly authorized six-run experiment; never invoked by daily start.
Configuration, audited data source, activity check and source hashes are embedded by builder.
"""
def m10_execute(session):
    import copy, datetime, gc, hashlib, io, json, math, os, random, time, traceback
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import default_collate

    P=M10_PROTOCOL
    assert M10_AUTHORIZED is True and P['experiment']=='M06_FV_JOINT_RETURN_V1'
    assert P['budget']['additional_steps_each']==300 and P['budget']['microbatch']==16 and P['budget']['accumulation']==4
    assert P['seeds']==[271828,271829,271830] and P['optimizer']['weight_decay']==0
    root=Path(P['canonical_run'])
    result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'RUNNING','runs':[],
            'test_read':False,'orders_enabled':False,'training_authorized':True}
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for b in iter(lambda:f.read(1048576),b''):h.update(b)
        return h.hexdigest()
    def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    def net_hash(net,exclude_lora=False):
        h=hashlib.sha256()
        for name,t in sorted(net.state_dict().items()):
            if exclude_lora and 'lora_' in name:continue
            name=name.replace('.base.', '.') if exclude_lora else name
            h.update(name.encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def put(path,data):
        path=Path(path)
        if path.exists():
            assert path.read_bytes()==data,'Conflicting existing artifact: '+str(path)
            return
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        assert digest(path)==hashlib.sha256(data).hexdigest()
    def put_json(path,data):put(path,canonical(data))
    def put_npz(path,**arrays):
        b=io.BytesIO();np.savez_compressed(b,**arrays);put(path,b.getvalue())
    def load_json(path):return json.loads(Path(path).read_text())
    def progress(**values):print('M10_PROGRESS '+json.dumps(values),flush=True)

    # Repeat only volatile readiness checks; historical experiments are never invoked.
    activity_ns={'__name__':'m10_activity'}
    exec(compile(M10_ACTIVITY_SOURCE,'m10_activity','exec'),activity_ns)
    activity=activity_ns['m09_activity_gate']({'AXXEL_SESSION':session})
    assert activity['status']=='PASS_CURRENT_ACTIVITY',activity['blockers']
    assert os.path.ismount('/content/drive') and torch.cuda.is_available()
    base,tok=session['model'],session['tokenizer']
    assert net_hash(base)==P['base_hash'] and net_hash(tok)==P['tokenizer_hash']
    assert all(not p.requires_grad and p.grad is None for net in (base,tok) for p in net.parameters())
    assert digest(root/'informes/PROTOCOLO_FASE_V_JOINT_PROPUESTO_20261120.json')==M10_PROTOCOL_SHA
    manifest3=load_json(root/'fase_III/checksums.json')
    assert digest(root/'fase_III/checksums.json')=='731d1c5412d4836e16961ff2e3747e6fd4da124db81eaf20265a638b6c90d7a1'
    def verified3(relative):
        p=root/'fase_III'/relative
        assert digest(p)==manifest3[relative]['sha256'],str(p)
        return p
    for seed in P['seeds']:
        init=P['head_initialization'][str(seed)]
        assert digest(init['path'])==init['sha256']
        packet=torch.load(init['path'],map_location='cpu',weights_only=True)
        assert all(torch.equal(v.cpu(),packet['head'][k]) for k,v in session['heads'][seed].state_dict().items())

    science=copy.deepcopy(P)
    for k in ('created_date_Bogota','status','purpose','canonical_run','output_subdirectory_proposed','training_started'):science.pop(k,None)
    for head in science['head_initialization'].values():head.pop('path');head.pop('model')
    science['execution_source_sha256']=M10_SOURCE_SHA
    science['data_code_sha256']=hashlib.sha256(M10_DATA_SOURCE.encode()).hexdigest()
    fingerprint=hashlib.sha256(canonical(science)).hexdigest()
    registry=root/'fase_V_joint_v1'
    registry.mkdir(exist_ok=True)
    # Any prior registry entry for this protocol with changed code/config is a conflict, not a fresh run.
    for folder in registry.iterdir():
        assert folder.is_dir() and (folder/'config.json').is_file(),'Untracked Phase V artifact: '+str(folder)
        assert folder.name==fingerprint and load_json(folder/'config.json')==science,'Existing Phase V configuration conflict'
    out=registry/fingerprint
    out.mkdir(exist_ok=True)
    put_json(out/'config.json',science)
    put(out/'training_source.py',M10_BODY_SOURCE.encode())
    put(out/'data_source.py',M10_DATA_SOURCE.encode())
    put(out/'evaluation_source.py',M10_EVALUATION_SOURCE.encode())
    put(out/'activity_source.py',M10_ACTIVITY_SOURCE.encode())
    put_json(out/'protocol_original.json',P)
    result['artifact_directory']=str(out);result['scientific_fingerprint']=fingerprint
    attempt=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    lock=out/'active.lock'
    owner={'pid':os.getpid(),'attempt':attempt,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    if lock.exists():raise RuntimeError('Existing execution lock: inspect persisted state before resuming')
    with lock.open('x') as f:json.dump(owner,f)
    saved_rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    old_deterministic=torch.are_deterministic_algorithms_enabled()
    old_tf32=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        base.eval();tok.eval()
        put_json(out/f'attempts/{attempt}/activity.json',activity)
        put_json(out/f'attempts/{attempt}/environment.json',{'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'free_total_bytes':torch.cuda.mem_get_info()})
        ns={'__name__':'m10_authorized_data','M6_PREFLIGHT_OK':True,'M6_BASE':root.parents[1],'M6_DIR':root,
            'M6_POLICY':load_json(root/'configuraciones/politica.json'),'torch':torch,'m6_sha':digest}
        assert ns['M6_POLICY']['allowed_source_years']==list(range(2010,2021))
        exec(compile(M10_DATA_SOURCE,'m10_authorized_data','exec'),ns)
        assert digest(root/'auditoria/datasets_autorizados.json')==P['data']['source_manifest_sha256']
        assert ns['M6_SOURCE_MANIFEST']==load_json(root/'auditoria/datasets_autorizados.json')
        pd.testing.assert_frame_equal(ns['m6_ix'],pd.read_parquet(root/'auditoria/ventanas_m06.parquet'))
        train,dev=ns['m6_train'],ns['m6_dev'];dev_index=ns['m6_dev_ix']
        assert len(train)==8428 and len(dev)==950 and dev_index.ny_date.nunique()==243
        scalers=session['scalers']
        scaler_path=verified3('checkpoints/fase_III/7587400db64558e6/train_only_scalers.npz')
        assert digest(scaler_path)==P['shared_model']['scalers_sha256']
        with np.load(scaler_path,allow_pickle=False) as z:
            assert all(np.array_equal(scalers[k],z[k]) for k in z.files)
        mean=torch.as_tensor(scalers['hidden_mean'],device='cuda')
        scale=torch.as_tensor(scalers['hidden_scale'],device='cuda')
        ret_scale=torch.as_tensor(scalers['return_scale'],device='cuda')
        def normalize(h):return ((h.double()-mean).float().double()/scale).float()
        def encode_dataset(ds):
            packs={k:[] for k in ('a','b','stamp','features','truth','last','target')}
            with torch.no_grad():
                for i in range(0,len(ds),16):
                    b={k:v.cuda() for k,v in default_collate([ds[j] for j in range(i,min(i+16,len(ds)))]).items()}
                    a,c=tok.encode(b['x'][:,:56],half=True)
                    stamp=b['stamp'][:,:56]
                    h=base.decode_s1(a,c,stamp=stamp)[1][:,-1]
                    truth=b['truth'][:,:,3].double();last=b['last_close'].double()
                    target=(10000*(truth/last[:,None]-1)/ret_scale).float()
                    for key,value in zip(packs,(a,c,stamp,normalize(h),truth,last,target)):packs[key].append(value.detach())
            return {k:torch.cat(v) for k,v in packs.items()}
        progress(stage='encoding_context_only',train_windows=len(train),dev_windows=len(dev))
        caches={'train':encode_dataset(train),'dev':encode_dataset(dev)}
        truth=caches['dev']['truth'].cpu().numpy();last=caches['dev']['last'].cpu().numpy()
        put_npz(out/'development_index.npz',truth=truth,last=last,session=dev_index.ny_date.to_numpy(dtype=str),hour=dev_index.ny_hour.to_numpy(dtype=str),decision=dev_index.decision_utc.astype(str).to_numpy(dtype=str))

        class LoRA(nn.Module):
            def __init__(self,layer):
                super().__init__();self.base=layer.requires_grad_(False)
                self.lora_A=nn.Parameter(torch.empty(8,layer.in_features,device='cuda'))
                self.lora_B=nn.Parameter(torch.zeros(layer.out_features,8,device='cuda'))
                self.dropout=nn.Dropout(.1);nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
            def forward(self,x):return self.base(x)+2*F.linear(F.linear(self.dropout(x),self.lora_A),self.lora_B)
        def features(net,cache,ids,joint):
            if not joint:return cache['features'][ids]
            return normalize(net.decode_s1(cache['a'][ids],cache['b'][ids],stamp=cache['stamp'][ids])[1][:,-1])
        def mode(net,head,training):
            net.eval();head.train(training)
            for module in net.modules():
                if isinstance(module,LoRA):module.dropout.train(training)
        def evaluate(net,head,joint):
            mode(net,head,False);parts=[];losses=[];start=time.perf_counter()
            with torch.no_grad():
                for i in range(0,950,16):
                    ids=slice(i,min(i+16,950));pred=head(features(net,caches['dev'],ids,joint))
                    assert torch.isfinite(pred).all()
                    parts.append(pred.double());losses.append((float(F.smooth_l1_loss(pred,caches['dev']['target'][ids],beta=1)),len(pred)))
            loc=torch.cat(parts).cpu().numpy()
            prices=last[:,None]*(1+loc*scalers['return_scale']/10000)
            return prices,sum(v*n for v,n in losses)/950,(time.perf_counter()-start)/950
        references={'Persistencia':np.repeat(last[:,None],4,axis=1)}
        for seed in P['seeds']:
            frame=pd.read_parquet(verified3(f'metricas/Head_Huber_s{seed}.parquet'))
            assert np.array_equal(frame.pivot(index='window',columns='horizon',values='truth').sort_index().to_numpy(),truth)
            references[f'Head_Huber_s{seed}']=frame.pivot(index='window',columns='horizon',values='prediction').sort_index().to_numpy()

        def rng_state(generator):
            n=np.random.get_state()
            return {'python':random.getstate(),'numpy_global':[n[0],n[1].tolist(),n[2],n[3],n[4]],'order':generator.bit_generator.state,'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
        def restore_rng(state,generator):
            random.setstate(state['python']);n=state['numpy_global'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
            generator.bit_generator.state=state['order'];torch.set_rng_state(state['cpu'].cpu());torch.cuda.set_rng_state_all([t.cpu() for t in state['cuda']])
        def verify_commit(folder):
            commit=load_json(folder/'commit.json')
            assert commit['fingerprint']==fingerprint and commit['source_sha256']==M10_SOURCE_SHA
            assert digest(folder/'checkpoint.pt')==commit['checkpoint_sha256']
            return commit
        def save_checkpoint(folder,step,net,head,opt,generator,curves,order_trace,arm,seed):
            dest=folder/'checkpoints'/f'step_{step:06d}'
            dest.mkdir(parents=True,exist_ok=False)
            packet={'step':step,'arm':arm,'seed':seed,'fingerprint':fingerprint,'source_sha256':M10_SOURCE_SHA,
                    'head':{k:v.detach().cpu() for k,v in head.state_dict().items()},
                    'adapter':{k:v.detach().cpu() for k,v in net.state_dict().items() if 'lora_' in k},
                    'optimizer':opt.state_dict(),'rng':rng_state(generator),'curves':curves,'order_trace':order_trace,
                    'base_hash':P['base_hash'],'tokenizer_hash':P['tokenizer_hash']}
            b=io.BytesIO();torch.save(packet,b);put(dest/'checkpoint.pt',b.getvalue())
            loaded=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=True)
            assert loaded['step']==step and all(torch.equal(v,loaded['head'][k]) for k,v in packet['head'].items())
            put_json(dest/'commit.json',{'step':step,'fingerprint':fingerprint,'source_sha256':M10_SOURCE_SHA,'checkpoint_sha256':digest(dest/'checkpoint.pt')})
            progress(stage='checkpoint_committed',arm=arm,seed=seed,step=step,loss=curves[-1]['loss'] if curves else None)

        predictions={};curves_by_run={};timings={}
        for arm in ('control_head_only','joint_QV_head'):
            joint=arm=='joint_QV_head'
            for seed in P['seeds']:
                folder=out/arm/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
                run_name=f'{arm}_s{seed}'
                if (folder/'completed.json').exists():
                    done=load_json(folder/'completed.json')
                    for rel,sha in done['hashes'].items():assert digest(folder/rel)==sha
                    with np.load(folder/'predictions.npz',allow_pickle=False) as z:predictions[run_name]=z['prediction']
                    curves_by_run[run_name]=load_json(folder/'curves.json');timings[run_name]=done['seconds_per_window']
                    result['runs'].append(done);progress(stage='completed_run_reused',arm=arm,seed=seed);continue
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                generator=np.random.default_rng(seed)
                net=copy.deepcopy(base) if joint else base
                if joint:
                    for name in P['arms'][arm]['lora_targets']:
                        parent,leaf=name.rsplit('.',1);obj=net.get_submodule(parent);setattr(obj,leaf,LoRA(getattr(obj,leaf)))
                head=copy.deepcopy(session['heads'][seed]).requires_grad_(True)
                trainable=[p for p in net.parameters() if p.requires_grad]+list(head.parameters())
                assert sum(p.numel() for p in trainable)==P['arms'][arm]['trainable_parameters']
                groups=[{'params':head.parameters(),'lr':3e-4}]
                if joint:groups.append({'params':[p for p in net.parameters() if p.requires_grad],'lr':1e-4})
                opt=torch.optim.AdamW(groups,weight_decay=0)
                step=0;curves=[];order_trace=[]
                existing=sorted((folder/'checkpoints').glob('step_*')) if (folder/'checkpoints').exists() else []
                if existing:
                    for checkpoint in existing:verify_commit(checkpoint)
                    packet=torch.load(existing[-1]/'checkpoint.pt',map_location='cpu',weights_only=True)
                    assert packet['arm']==arm and packet['seed']==seed
                    head.load_state_dict(packet['head'],strict=True)
                    sd=net.state_dict();assert set(packet['adapter'])=={k for k in sd if 'lora_' in k};sd.update(packet['adapter']);net.load_state_dict(sd,strict=True)
                    opt.load_state_dict(packet['optimizer']);restore_rng(packet['rng'],generator)
                    step=packet['step'];curves=packet['curves'];order_trace=packet['order_trace']
                    assert 0<=step<=300
                    progress(stage='resume_verified',arm=arm,seed=seed,step=step)
                else:
                    pred0,loss0,_=evaluate(net,head,joint)
                    delta=np.abs(pred0-references[f'Head_Huber_s{seed}'])
                    assert delta.max()<=.001 and delta.mean()<=.00001,'Step-zero mismatch'
                    put_npz(folder/'step0.npz',prediction=pred0,truth=truth)
                    put_json(folder/'step0_metrics.json',{'MAE':float(abs(pred0-truth).mean()),'loss':loss0,'max_diff':float(delta.max()),'mean_diff':float(delta.mean())})
                    save_checkpoint(folder,0,net,head,opt,generator,curves,order_trace,arm,seed)
                mode(net,head,True)
                started=time.perf_counter()
                while step<300:
                    ids=generator.integers(0,8428,size=64);opt.zero_grad(set_to_none=True);total=0.
                    for j in range(4):
                        rows=ids[j*16:(j+1)*16]
                        pred=head(features(net,caches['train'],rows,joint))
                        loss=F.smooth_l1_loss(pred,caches['train']['target'][rows],beta=1)
                        assert torch.isfinite(loss),'Nonfinite loss'
                        (loss/4).backward();total+=float(loss.detach())/4
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable),'Invalid/missing gradient'
                    assert sum(float(p.grad.abs().sum()) for p in head.parameters())>0,'Zero head gradients'
                    if joint:
                        assert all(p.grad.abs().sum()>0 for n,p in net.named_parameters() if 'lora_B' in n),'Zero LoRA B gradient'
                        if step>0:assert any(p.grad.abs().sum()>0 for n,p in net.named_parameters() if 'lora_A' in n),'Zero LoRA A gradient after first update'
                    assert all(p.grad is None for n,p in net.named_parameters() if 'lora_' not in n)
                    assert all(p.grad is None for p in tok.parameters())
                    norm=nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
                    opt.step();step+=1
                    assert all(torch.isfinite(p).all() for p in trainable),'Nonfinite parameters'
                    curves.append({'step':step,'loss':total,'gradient_norm':float(norm),'elapsed_seconds_this_attempt':time.perf_counter()-started})
                    order_trace.append(hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest())
                    if step%25==0:save_checkpoint(folder,step,net,head,opt,generator,curves,order_trace,arm,seed)
                pred,dev_loss,latency=evaluate(net,head,joint)
                assert net_hash(net,exclude_lora=joint)==P['base_hash'],'Backbone changed'
                put_npz(folder/'predictions.npz',prediction=pred,truth=truth,last=last)
                put_json(folder/'curves.json',curves);put_json(folder/'training_order.json',order_trace)
                put_json(folder/'final_metrics.json',{'MAE':float(abs(pred-truth).mean()),'loss':dev_loss,'seconds_per_window':latency})
                files=[p for p in folder.rglob('*') if p.is_file()]
                done={'arm':arm,'seed':seed,'status':'COMPLETED','step':300,'MAE':float(abs(pred-truth).mean()),'seconds_per_window':latency,
                      'hashes':{str(p.relative_to(folder)):digest(p) for p in files}}
                put_json(folder/'completed.json',done)
                predictions[run_name]=pred;curves_by_run[run_name]=curves;timings[run_name]=latency;result['runs'].append(done)
                progress(stage='run_completed',arm=arm,seed=seed,MAE=done['MAE'])
                del head,opt,trainable,net;gc.collect();torch.cuda.empty_cache()
        for seed in P['seeds']:
            assert load_json(out/'control_head_only'/f'seed_{seed}'/'training_order.json')==load_json(out/'joint_QV_head'/f'seed_{seed}'/'training_order.json')
        # Evaluate and plot only persisted real predictions; no further fitting or selection.
        evaluation_ns={}
        exec(compile(M10_EVALUATION_SOURCE,'m10_evaluation','exec'),evaluation_ns)
        result['evaluation']=evaluation_ns['m10_evaluate'](out,root,P,predictions,references,truth,last,dev_index,curves_by_run,timings)
        result['status']='COMPLETED';result['runs']=[{k:v for k,v in r.items() if k!='hashes'} for r in result['runs']]
        assert net_hash(base)==P['base_hash'] and net_hash(tok)==P['tokenizer_hash']
    except Exception as exc:
        result['status']='BLOCKED';result['error']=type(exc).__name__+': '+str(exc)
        result['traceback']=traceback.format_exc()
    finally:
        random.setstate(saved_rng[0]);np.random.set_state(saved_rng[1]);torch.set_rng_state(saved_rng[2]);torch.cuda.set_rng_state_all(saved_rng[3])
        torch.use_deterministic_algorithms(old_deterministic);torch.backends.cuda.matmul.allow_tf32=old_tf32[0];torch.backends.cudnn.allow_tf32=old_tf32[1]
        result['production_weights_unchanged']=net_hash(base)==P['base_hash'] and net_hash(tok)==P['tokenizer_hash']
        put_json(out/f'attempts/{attempt}/result.json',result)
        if load_json(lock)==owner:lock.unlink()
        files=[p for p in out.rglob('*') if p.is_file()]
        put_json(out/f'attempts/{attempt}/manifest.json',{str(p.relative_to(out)):digest(p) for p in files})
        print('M10_PHASE_V_RESULT '+json.dumps(result),flush=True)
    return result

M10_RESULT=m10_execute(AXXEL_SESSION)
