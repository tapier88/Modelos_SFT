# Continuation fragment; assembled with the reviewed read-only data source.
M7_DATA_NS={'__name__':'m07_authorized_data','M6_PREFLIGHT_OK':True,'M6_BASE':M7_ROOT.parents[1], 'M6_DIR':M7_ROOT,
            'M6_POLICY':json.loads((M7_ROOT/'configuraciones/politica.json').read_text()),'torch':torch,'m6_sha':m7_sha}
assert M7_DATA_NS['M6_POLICY']['allowed_source_years']==list(range(2010,2021))
M7_STEP0={'status':'running','RESEARCH_ONLY':True,'test_read':False,'orders_enabled':False,'training_steps':0,
          'tolerance':{'price_max_abs':0.001,'price_mean_abs':0.00001,'rationale':'FP32 CPU/GPU roundoff; fixed before comparison. Absolute bound 0.001 price units plus mean bound 1e-5; no relative tolerance.'}}
try:
    exec(compile(M7_DATA_SOURCE,'m07_authorized_data_readonly','exec'),M7_DATA_NS)
    assert M7_DATA_NS['M6_SOURCE_MANIFEST']==json.loads((M7_ROOT/'auditoria/datasets_autorizados.json').read_text())
    assert m7_sha(M7_ROOT/'auditoria/datasets_autorizados.json')=='a4f37d6518acd433712e404ba7a35a0800d5dacda1c87dbebba9fa9a234d1b7b'
    pd.testing.assert_frame_equal(M7_DATA_NS['m6_ix'],pd.read_parquet(M7_ROOT/'auditoria/ventanas_m06.parquet'))
    M7_TRAIN=M7_DATA_NS['m6_train'];M7_DEV=M7_DATA_NS['m6_dev']
    assert len(M7_TRAIN)==8428 and len(M7_DEV)==950
    assert M7_DATA_NS['m6_dev_ix'].ny_date.nunique()==243
    M7_STEP0['data_checks']=M7_DATA_NS['M6_DATA_CHECKS']
    print('M07_DATA_VALIDATED',json.dumps(M7_STEP0['data_checks']),flush=True)
    # Refit scalers using only already-verified train caches; never recompute/train old heads.
    from sklearn.preprocessing import StandardScaler
    train_parts=[]
    for path in sorted(M7_FDIR.glob('features_train_2010_2019_*.npz')):
        with np.load(path,allow_pickle=False) as z:train_parts.append({k:z[k] for k in ['ids','hidden','truth','last']})
    tc={k:np.concatenate([p[k] for p in train_parts]) for k in train_parts[0]}
    assert np.array_equal(tc['ids'],np.arange(8428))
    scaler=StandardScaler().fit(tc['hidden'])
    assert np.array_equal(scaler.mean_,M7_SCALERS['hidden_mean']) and np.array_equal(scaler.scale_,M7_SCALERS['hidden_scale'])
    returns=10000*(tc['truth'].astype(np.float64)/tc['last'][:,None]-1)
    assert np.array_equal(np.maximum(returns.std(0),.001).astype(np.float32),M7_SCALERS['return_scale'])
    M7_STEP0['scalers_reproduced_train_only']=True
    del tc,train_parts
    M7_STEP0['causality']=M7_KP.contract_tests(M7_MODEL,M7_TOK,M7_TRAIN,M7_DEVICE)
    def m7_batch(ds,ids):return {k:v.to(M7_DEVICE) for k,v in default_collate([ds[int(i)] for i in ids]).items()}
    def m7_hidden(model,batch):
        with torch.no_grad():a,b=M7_TOK.encode(batch['x'][:,:56],half=True)
        return model.decode_s1(a,b,stamp=batch['stamp'][:,:56])[1][:,-1]
    # Full feature inference is new verification, not a repetition of head training.
    hs=[];truth=[];last=[]
    M7_MODEL.eval();M7_TOK.eval()
    with torch.no_grad():
        for start in range(0,950,16):
            batch=m7_batch(M7_DEV,range(start,min(start+16,950)))
            hs.append(m7_hidden(M7_MODEL,batch).cpu().numpy())
            truth.append(batch['truth'][:,:,3].cpu().numpy());last.append(batch['last_close'].cpu().numpy())
            if start%160==0:print('M07_STEP0_WINDOWS',start,flush=True)
    M7_HIDDEN=np.concatenate(hs);M7_TRUTH=np.concatenate(truth).astype(np.float64);M7_LAST=np.concatenate(last).astype(np.float64)
    feature=scaler.transform(M7_HIDDEN).astype(np.float32)
    M7_HD=torch.tensor(feature,device=M7_DEVICE)
    M7_PRED0={};results=[]
    for seed,head in M7_HEADS.items():
        with torch.no_grad():loc=head(M7_HD).cpu().numpy().astype(np.float64)
        pred=M7_LAST[:,None]*(1+loc*M7_SCALERS['return_scale']/10000)
        ref=pd.read_parquet(M7_ROOT/f'fase_III/metricas/Head_Huber_s{seed}.parquet')
        saved=ref.pivot(index='window',columns='horizon',values='prediction').sort_index().to_numpy()
        assert np.array_equal(ref.pivot(index='window',columns='horizon',values='truth').sort_index().to_numpy(),M7_TRUTH)
        delta=abs(pred-saved)
        result={'seed':seed,'max_abs_price_diff':float(delta.max()),'mean_abs_price_diff':float(delta.mean()),'MAE':float(abs(pred-M7_TRUTH).mean())}
        results.append(result);print('M07_STEP0_COMPARISON',json.dumps(result),flush=True)
        assert delta.max()<=.001 and delta.mean()<=.00001, 'Step0 numeric tolerance exceeded'
        M7_PRED0[seed]=pred
    M7_STEP0['step0']=results
    # Check labels and future-invariance on an independent train probe, no optimizer updates.
    probe=m7_batch(M7_TRAIN,[0,1,2,3])
    modified={k:v.clone() for k,v in probe.items()}
    modified['x'][:,56:]=1234;modified['truth'].zero_()
    with torch.no_grad():assert torch.equal(m7_hidden(M7_MODEL,probe),m7_hidden(M7_MODEL,modified))
    M7_STEP0['future_invariance_hidden']=True
    joint=copy.deepcopy(M7_MODEL)
    torch.manual_seed(271828)
    for block in joint.transformer:
        for leaf in ['q_proj','v_proj']:
            base=getattr(block.self_attn,leaf);layer=M7LoRA(base,8)
            nn.init.kaiming_uniform_(layer.lora_A,a=5**.5)
            setattr(block.self_attn,leaf,layer)
    joint.eval();head=copy.deepcopy(M7_HEADS[271828]);head.requires_grad_(True)
    assert sum(p.numel() for p in joint.parameters() if p.requires_grad)+sum(p.numel() for p in head.parameters())==164164
    with torch.no_grad():assert torch.equal(m7_hidden(joint,probe),m7_hidden(M7_MODEL,probe))
    # Use sklearn-equivalent float32 in-place rounding for comparison; differentiable normalization here.
    mean=torch.as_tensor(M7_SCALERS['hidden_mean'],device=M7_DEVICE,dtype=torch.float64)
    scale=torch.as_tensor(M7_SCALERS['hidden_scale'],device=M7_DEVICE,dtype=torch.float64)
    h=m7_hidden(joint,probe)
    h=((h.double()-mean).float().double()/scale).float()
    target=(10000*(probe['truth'][:,:,3].double()/probe['last_close'][:,None].double()-1)/torch.as_tensor(M7_SCALERS['return_scale'],device=M7_DEVICE)).float()
    objective=F.smooth_l1_loss(head(h),target,beta=1)
    objective.backward()
    head_grad=all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()) and any(p.grad.abs().sum()>0 for p in head.parameters())
    b_grad=all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for n,p in joint.named_parameters() if 'lora_B' in n)
    base_grad_absent=all(p.grad is None for n,p in joint.named_parameters() if 'lora_' not in n)
    assert head_grad and b_grad and base_grad_absent and all(p.grad is None for p in M7_TOK.parameters())
    M7_STEP0['joint_zero_and_gradients']={'step0_identical':True,'head_finite_nonzero':bool(head_grad),'lora_B_finite_nonzero':bool(b_grad),'base_gradients_absent':base_grad_absent,'optimizer_steps':0,'lora_A_later_step_check':'pending; no optimization authorized'}
    assert m7_hash(M7_MODEL)==model_info['base_hash'] and m7_hash(M7_TOK)==model_info['tokenizer_hash']
    del joint,head
    M7_STEP0['status']='passed_checks_executed_training_still_disabled'
except Exception as exc:
    M7_STEP0['status']='blocked';M7_STEP0['error']=repr(exc)
    raise
finally:
    with (M7_OUT/'step0_preflight.json').open('x') as f:json.dump(M7_STEP0,f,indent=2,default=str)
    print('M07_STEP0_STATUS',json.dumps(M7_STEP0,default=str),flush=True)
