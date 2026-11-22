# M09_SCIENTIFIC_PREFLIGHT_V1
# DATA_SOURCE and PROTOCOL are embedded by automation/m09_build_preflight.py.
def m09_validate(session):
    import copy, datetime, hashlib, io, json, os, random
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import default_collate

    root = Path(PROTOCOL['canonical_run'])
    out = root.parent / 'm09_preflight' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    report = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'phase_V_started': False, 'test_read': False, 'orders_enabled': False,
              'status': 'RUNNING', 'checks': {}, 'blockers': [], 'artifact_directory': str(out)}
    def sha(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda: f.read(1048576), b''): h.update(b)
        return h.hexdigest()
    def model_hash(net):
        h = hashlib.sha256()
        for name, x in sorted(net.state_dict().items()):
            h.update(name.encode()); h.update(str(tuple(x.shape)).encode()); h.update(x.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def record(key, value):
        report['checks'][key] = value
        print('M09_CHECK', key, json.dumps(value, default=str), flush=True)
    model, tok = session['model'], session['tokenizer']
    rng_before = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    deterministic = torch.are_deterministic_algorithms_enabled()
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    model_mode, tok_mode = model.training, tok.training
    base_before, tok_before = model_hash(model), model_hash(tok)
    try:
        assert base_before == PROTOCOL['base_hash'] and tok_before == PROTOCOL['tokenizer_hash']
        assert not any(p.requires_grad for p in model.parameters())
        assert not any(p.requires_grad for p in tok.parameters())
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        model.eval(); tok.eval()
        assert os.path.ismount('/content/drive') and torch.cuda.is_available()
        record('infrastructure', {'gpu': torch.cuda.get_device_name(0), 'free_bytes': torch.cuda.mem_get_info()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda, 'base_hash': base_before, 'tokenizer_hash': tok_before})
        ns = {'__name__': 'm09_authorized_data', 'M6_PREFLIGHT_OK': True, 'M6_BASE': root.parents[1],
              'M6_DIR': root, 'M6_POLICY': json.loads((root/'configuraciones/politica.json').read_text()), 'torch': torch, 'm6_sha': sha}
        assert ns['M6_POLICY']['allowed_source_years'] == list(range(2010,2021))
        exec(compile(DATA_SOURCE, 'm09_readonly_data', 'exec'), ns)
        assert sha(root/'auditoria/datasets_autorizados.json') == PROTOCOL['data']['source_manifest_sha256']
        assert ns['M6_SOURCE_MANIFEST'] == json.loads((root/'auditoria/datasets_autorizados.json').read_text())
        pd.testing.assert_frame_equal(ns['m6_ix'], pd.read_parquet(root/'auditoria/ventanas_m06.parquet'))
        train, dev = ns['m6_train'], ns['m6_dev']
        assert len(train)==8428 and len(dev)==950 and ns['m6_dev_ix'].ny_date.nunique()==243
        for ds in (train,dev):
            for i in (0,len(ds)//2,len(ds)-1):
                sample=ds[i]; raw=ds.values[int(ds.starts[i]):int(ds.starts[i])+60]
                assert np.array_equal(sample['truth'].numpy(),raw[56:60])
                assert float(sample['last_close'])==float(raw[55,3])
        record('data', dict(ns['M6_DATA_CHECKS'], target_H1_H4_aligned=True, indices_equal_audit=True, splits_disjoint=True))
        record('causality', session['pipeline'].contract_tests(model,tok,train,torch.device('cuda')))
        scalers=session['scalers']
        fdir=root/'fase_III/checkpoints/fase_III/7587400db64558e6'
        assert sha(fdir/'train_only_scalers.npz')==PROTOCOL['shared_model']['scalers_sha256']
        with np.load(fdir/'train_only_scalers.npz',allow_pickle=False) as z:
            assert all(np.array_equal(scalers[k],z[k]) for k in z.files)
        from sklearn.preprocessing import StandardScaler
        train_parts=[]
        manifest=json.loads((root/'fase_III/checksums.json').read_text())
        assert sha(root/'fase_III/checksums.json')=='731d1c5412d4836e16961ff2e3747e6fd4da124db81eaf20265a638b6c90d7a1'
        for relative, meta in manifest.items():
            if 'features_train_2010_2019_' in relative and relative.endswith('.npz'):
                path=root/'fase_III'/relative
                assert sha(path)==meta['sha256']
                with np.load(path,allow_pickle=False) as z: train_parts.append({k:z[k] for k in ['ids','hidden','truth','last']})
        parts={k:np.concatenate([x[k] for x in train_parts]) for k in train_parts[0]}
        order=np.argsort(parts['ids']); parts={k:v[order] for k,v in parts.items()}
        assert np.array_equal(parts['ids'],np.arange(8428))
        scaler=StandardScaler().fit(parts['hidden'])
        assert np.array_equal(scaler.mean_,scalers['hidden_mean']) and np.array_equal(scaler.scale_,scalers['hidden_scale'])
        returns=10000*(parts['truth'].astype(np.float64)/parts['last'][:,None]-1)
        assert np.array_equal(np.maximum(returns.std(0),.001).astype(np.float32),scalers['return_scale'])
        del parts,train_parts
        record('scalers_train_only',True)
        def batch(ds,ids):return {k:v.cuda() for k,v in default_collate([ds[i] for i in ids]).items()}
        def hidden(net,b):
            with torch.no_grad():a,c=tok.encode(b['x'][:,:56],half=True)
            return net.decode_s1(a,c,stamp=b['stamp'][:,:56])[1][:,-1]
        hs=[];last=[];truth=[]
        with torch.no_grad():
            for i in range(0,950,16):
                b=batch(dev,range(i,min(i+16,950)))
                hs.append(hidden(model,b).cpu().numpy());last.append(b['last_close'].cpu().numpy());truth.append(b['truth'][:,:,3].cpu().numpy())
        h=np.concatenate(hs);last=np.concatenate(last).astype(np.float64);truth=np.concatenate(truth).astype(np.float64)
        features=scaler.transform(h).astype(np.float32)
        equivalence=[]
        for seed,head in session['heads'].items():
            assert sha(Path(PROTOCOL['head_initialization'][str(seed)]['path']))==PROTOCOL['head_initialization'][str(seed)]['sha256']
            packet=torch.load(Path(PROTOCOL['head_initialization'][str(seed)]['path']),map_location='cpu',weights_only=True)
            assert all(torch.equal(v.detach().cpu(),packet['head'][k]) for k,v in head.state_dict().items())
            with torch.no_grad():loc=head(torch.as_tensor(features,device='cuda')).cpu().numpy().astype(np.float64)
            pred=last[:,None]*(1+loc*scalers['return_scale']/10000)
            ref=pd.read_parquet(root/f'fase_III/metricas/Head_Huber_s{seed}.parquet')
            saved=ref.pivot(index='window',columns='horizon',values='prediction').sort_index().to_numpy()
            assert np.array_equal(ref.pivot(index='window',columns='horizon',values='truth').sort_index().to_numpy(),truth)
            diff=abs(pred-saved)
            assert diff.max()<=.001 and diff.mean()<=.00001
            equivalence.append({'seed':seed,'max_abs_price_diff':float(diff.max()),'mean_abs_price_diff':float(diff.mean()),'MAE':float(abs(pred-truth).mean())})
            np.savez_compressed(out/f'step0_{seed}.npz',prediction=pred,truth=truth)
        record('full_price_step0',equivalence)
        b=batch(train,[0,1]); changed={k:v.clone() for k,v in b.items()}
        changed['x'][:,56:]=1234; changed['truth'].zero_()
        with torch.no_grad(): assert torch.equal(hidden(model,b),hidden(model,changed))
        record('regression_future_invariance',True)

        class LoRA(nn.Module):
            def __init__(self,base):
                super().__init__();self.base=base.requires_grad_(False)
                self.lora_A=nn.Parameter(torch.empty(8,base.in_features,device='cuda'))
                self.lora_B=nn.Parameter(torch.zeros(base.out_features,8,device='cuda'))
                self.dropout=nn.Dropout(.1);nn.init.kaiming_uniform_(self.lora_A,a=5**.5)
            def forward(self,x):return self.base(x)+2*F.linear(F.linear(self.dropout(x),self.lora_A),self.lora_B)
        joint=copy.deepcopy(model).eval()
        torch.manual_seed(271828)
        for name in PROTOCOL['arms']['joint_QV_head']['lora_targets']:
            parent,leaf=name.rsplit('.',1);obj=joint.get_submodule(parent);setattr(obj,leaf,LoRA(getattr(obj,leaf)))
        head=copy.deepcopy(session['heads'][271828]).requires_grad_(True).train()
        joint.eval()
        with torch.no_grad(): assert torch.equal(hidden(joint,b),hidden(model,b))
        for mod in joint.modules():
            if isinstance(mod,LoRA):mod.dropout.train()
        assert sum(p.numel() for p in head.parameters())==33092
        assert sum(p.numel() for p in joint.parameters() if p.requires_grad)==131072
        target=(10000*(b['truth'][:,:,3].double()/b['last_close'][:,None].double()-1)/torch.as_tensor(scalers['return_scale'],device='cuda')).float()
        def normalize_feature(x):
            return ((x.double()-torch.as_tensor(scalers['hidden_mean'],device='cuda')).float().double()/torch.as_tensor(scalers['hidden_scale'],device='cuda')).float()
        F.smooth_l1_loss(head(normalize_feature(hidden(joint,b))),target,beta=1).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for n,p in joint.named_parameters() if 'lora_B' in n)
        assert all(p.grad is None for n,p in joint.named_parameters() if 'lora_' not in n)
        assert all(p.grad is None for p in tok.parameters())
        record('architecture_gradients',{'control_parameters':33092,'joint_parameters':164164,'zero_step_identical':True,'head_and_B_finite':True,'base_tokenizer_gradients_absent':True})

        # Disposable synthetic fixture: no market labels and no production optimizer.
        synthetic={k:v.clone() for k,v in b.items()};synthetic['x'].zero_()
        synth_target=torch.ones(2,4,device='cuda')
        opt=torch.optim.AdamW([{'params':head.parameters(),'lr':3e-4},{'params':[p for p in joint.parameters() if p.requires_grad],'lr':1e-4}],weight_decay=0)
        gen=np.random.default_rng(271828)
        def fixture_step(net,hd,optimizer,generator):
            optimizer.zero_grad(set_to_none=True)
            ids=generator.integers(0,2,size=2)
            inp={k:v[ids] for k,v in synthetic.items()}
            loss=F.smooth_l1_loss(hd(normalize_feature(hidden(net,inp))),synth_target)
            loss.backward()
            a_gradient=any(p.grad is not None and p.grad.abs().sum()>0 for n,p in net.named_parameters() if 'lora_A' in n)
            nn.utils.clip_grad_norm_([p for p in list(net.parameters())+list(hd.parameters()) if p.requires_grad],1.)
            optimizer.step()
            return float(loss.detach()),a_gradient,ids.tolist()
        fixture_step(joint,head,opt,gen)
        snapshot={'model':joint.state_dict(),'head':head.state_dict(),'optimizer':opt.state_dict(),'numpy':gen.bit_generator.state,
                  'python':random.getstate(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
        buffer=io.BytesIO();torch.save(snapshot,buffer);raw=buffer.getvalue()
        fixture_path=out/'synthetic_resume_fixture.pt'
        with fixture_path.open('xb') as f:f.write(raw)
        assert sha(fixture_path)==hashlib.sha256(raw).hexdigest()
        uninterrupted=fixture_step(joint,head,opt,gen)
        assert uninterrupted[1], 'LoRA_A gradient missing after first synthetic update'
        packet=torch.load(io.BytesIO(raw),map_location='cuda',weights_only=False)
        restored=copy.deepcopy(joint);restored.load_state_dict(packet['model'],strict=True)
        rh=copy.deepcopy(head);rh.load_state_dict(packet['head'],strict=True)
        ro=torch.optim.AdamW([{'params':rh.parameters(),'lr':3e-4},{'params':[p for p in restored.parameters() if p.requires_grad],'lr':1e-4}],weight_decay=0)
        ro.load_state_dict(packet['optimizer']);rg=np.random.default_rng();rg.bit_generator.state=packet['numpy']
        random.setstate(packet['python']);torch.set_rng_state(packet['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in packet['cuda']])
        resumed=fixture_step(restored,rh,ro,rg)
        assert uninterrupted==resumed
        assert all(torch.equal(x,restored.state_dict()[k]) for k,x in joint.state_dict().items())
        assert all(torch.equal(x,rh.state_dict()[k]) for k,x in head.state_dict().items())
        record('synthetic_resume',{'exact_loss_indices_parameters':True,'lora_A_second_gradient':True,'save_readback_sha256':sha(fixture_path),'disposable_optimizer_steps':3,'phase_V_steps':0})
        del joint,head,restored,rh,opt,ro,snapshot,packet

        # Inventory only experiment metadata, never dataset directories or reserved-year files.
        candidates=[]; ambiguous=[]
        for path in root.parent.rglob('*.json'):
            rel=str(path.relative_to(root.parent))
            if any(x in rel.lower() for x in ['2023','2024','datasets','m09_preflight','m07_recovery']):continue
            if path.stat().st_size>2000000:continue
            if not any(x in path.name.lower() for x in ['config','protocol','protocolo','manifest','complete','status','registry']):continue
            try: data=json.loads(path.read_text())
            except Exception: ambiguous.append(rel);continue
            text=json.dumps(data,sort_keys=True)
            if 'M06_FV_JOINT_RETURN_V1' in text or 'fase_V_joint' in rel:
                candidates.append({'path':rel,'sha256':sha(path),'status':data.get('status') if isinstance(data,dict) else None})
                if not isinstance(data,dict) or data.get('status')!='PROPOSED_NOT_EXECUTED':ambiguous.append(rel)
        record('deduplication_inventory',{'scope':str(root.parent),'candidates':candidates,'unreconciled':ambiguous})
        if ambiguous:report['blockers'].append('Unreconciled potential Phase V metadata')
        # Do not promote incomplete acceptance coverage to a complete scientific preflight.
        report['blockers'].append('Full namespace restore of III-IV twice and complete scientific fingerprint registry not yet certified by M09')
        report['status']='NO_APTO' if report['blockers'] else 'APTO'
    except Exception as exc:
        report['status']='NO_APTO';report['blockers'].append(type(exc).__name__+': '+str(exc))
    finally:
        random.setstate(rng_before[0]);np.random.set_state(rng_before[1]);torch.set_rng_state(rng_before[2]);torch.cuda.set_rng_state_all(rng_before[3])
        model.train(model_mode);tok.train(tok_mode)
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cuda.matmul.allow_tf32=tf32[0];torch.backends.cudnn.allow_tf32=tf32[1]
        report['production_weights_unchanged']=model_hash(model)==base_before and model_hash(tok)==tok_before
        report['production_rng_restored']=True
        if not report['production_weights_unchanged']:report['status']='NO_APTO';report['blockers'].append('Production weights changed')
        with (out/'report.json').open('x') as f:json.dump(report,f,indent=2)
        checks={p.name:sha(p) for p in out.iterdir() if p.is_file()}
        with (out/'checksums.json').open('x') as f:json.dump(checks,f,indent=2)
        report['manifest_sha256']=sha(out/'checksums.json')
        print('M09_PREFLIGHT_RESULT',json.dumps(report,default=str),flush=True)
    return report

M09_RESULT=m09_validate(AXXEL_SESSION)
