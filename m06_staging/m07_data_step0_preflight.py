# M07_DATA_STEP0_PREFLIGHT_V1
assert M7_CHECKS["status"] == "recovered_pending_data_and_step0"
M7_DATA_SOURCE="#@title MISIÓN 06 · 1. Datos autorizados, índice nuevo y normalización causal\nassert globals().get('M6_PREFLIGHT_OK') is True, 'Ejecutar primero M06 · 0'\nimport numpy as np\nimport pandas as pd\n\nM6_SOURCE = M6_BASE / 'datasets/processed/kronos_ohlcv_v1'\nM6_SOURCE_MANIFEST, m6_parts = [], []\n# Lista cerrada; no se enumeran directorios de datos, archivos ZIP ni índices históricos.\ndef m6_authorized_source(year):\n    if type(year) is not int or year not in range(2010, 2021):\n        raise ValueError('M06 solo permite archivos fuente 2010–2020')\n    return M6_SOURCE / f'XAUUSD_M15_BID_{year}.parquet'\n\nfor m6_year in range(2010, 2021):\n    m6_path = m6_authorized_source(m6_year)\n    m6_digest = m6_sha(m6_path)\n    m6_part = pd.read_parquet(m6_path)\n    m6_part['timestamp_utc'] = pd.to_datetime(m6_part.timestamp_utc, utc=True)\n    assert len(m6_part) and m6_part.timestamp_utc.dt.year.eq(m6_year).all()\n    assert m6_part.timestamp_utc.is_monotonic_increasing and m6_part.timestamp_utc.is_unique\n    assert m6_part.m1_candles.eq(15).all()\n    m6_prices = m6_part[['open', 'high', 'low', 'close', 'volume']]\n    assert np.isfinite(m6_prices.to_numpy()).all()\n    assert (m6_part.low > 0).all() and (m6_part.volume >= 0).all()\n    assert (m6_part.low <= m6_part[['open', 'close']].min(axis=1)).all()\n    assert (m6_part.high >= m6_part[['open', 'close']].max(axis=1)).all()\n    assert (m6_part.low <= m6_part.high).all()\n    assert m6_sha(m6_path) == m6_digest, 'Fuente cambió durante lectura'\n    M6_SOURCE_MANIFEST.append({'year': m6_year, 'file': m6_path.name, 'rows': len(m6_part), 'sha256': m6_digest})\n    m6_parts.append(m6_part[['timestamp_utc', 'open', 'high', 'low', 'close', 'volume']].copy())\nm6_df = pd.concat(m6_parts, ignore_index=True)\ndel m6_parts, m6_part, m6_prices\nassert m6_df.timestamp_utc.is_unique and m6_df.timestamp_utc.is_monotonic_increasing\nm6_ts = m6_df.timestamp_utc.astype('int64').to_numpy()\nM6_STEP = 15 * 60 * 10**9\nassert np.all(m6_ts % M6_STEP == 0)\nm6_df['split'] = np.where(m6_df.timestamp_utc.dt.year <= 2019, 'train', 'development')\nm6_df['segment'] = (m6_df.timestamp_utc.diff().ne(pd.Timedelta(minutes=15)) |\n                    m6_df.split.ne(m6_df.split.shift())).cumsum()\nm6_windows = []\nfor m6_segment, m6_group in m6_df.groupby('segment', sort=False):\n    m6_first, m6_last = int(m6_group.index[0]), int(m6_group.index[-1])\n    m6_starts = np.arange(m6_first, m6_last - 59 + 1, dtype=np.int64)\n    if not len(m6_starts):\n        continue\n    m6_decisions = m6_df.timestamp_utc.iloc[m6_starts + 56]\n    m6_ny = m6_decisions.dt.tz_convert('America/New_York')\n    for s, decision, ny in zip(m6_starts, m6_decisions, m6_ny):\n        if ny.strftime('%H:%M') not in M6_POLICY['ny_anchors']:\n            continue\n        m6_windows.append({'start_row': int(s), 'split': m6_group.split.iloc[0],\n                           'decision_utc': decision, 'decision_ny': ny.isoformat(),\n                           'ny_date': ny.strftime('%Y-%m-%d'), 'ny_hour': ny.strftime('%H:%M'),\n                           'context_start': m6_df.timestamp_utc.iloc[s],\n                           'target_end': m6_df.timestamp_utc.iloc[s+59] + pd.Timedelta(minutes=15)})\nm6_ix = pd.DataFrame(m6_windows).sort_values('decision_utc').reset_index(drop=True)\nassert len(m6_ix) and m6_ix.decision_utc.is_unique\nm6_train_ix = m6_ix[m6_ix.split.eq('train')].reset_index(drop=True)\nm6_dev_ix = m6_ix[m6_ix.split.eq('development')].reset_index(drop=True)\nassert len(m6_train_ix) and len(m6_dev_ix)\nassert m6_dev_ix.decision_utc.dt.year.eq(2020).all()\nassert m6_train_ix.target_end.max() < m6_dev_ix.context_start.min()\nassert (m6_ix.decision_utc.iloc[1:].to_numpy() >= m6_ix.target_end.iloc[:-1].to_numpy()).all()\nm6_starts = m6_ix.start_row.to_numpy(np.int64)\nassert np.all(m6_ts[m6_starts[:, None] + np.arange(60)] == m6_ts[m6_starts, None] + np.arange(60) * M6_STEP)\n\ndef m6_normalize(raw):\n    raw = np.asarray(raw, dtype=np.float32)\n    if raw.shape != (60, 6):\n        raise ValueError('Contrato M06: 60 filas, OHLCV + amount=0')\n    mean = raw[:56].mean(axis=0)\n    scale = raw[:56].std(axis=0) + np.float32(1e-5)\n    return np.clip((raw - mean) / scale, -5, 5), mean, scale\n\nclass M6Dataset(torch.utils.data.Dataset):\n    def __init__(self, frame, index):\n        self.L, self.H = 56, 4\n        self.index = index.reset_index(drop=True)\n        self.starts = self.index.start_row.to_numpy(np.int64)\n        self.values = np.column_stack([frame[['open','high','low','close','volume']].to_numpy(np.float32), np.zeros(len(frame), np.float32)])\n        t = frame.timestamp_utc.dt\n        self.stamp = np.column_stack([t.minute, t.hour, t.weekday, t.day, t.month]).astype(np.float32)\n    def __len__(self):\n        return len(self.starts)\n    def __getitem__(self, i):\n        s = int(self.starts[i])\n        raw = self.values[s:s+60].copy()\n        x, mean, scale = m6_normalize(raw)\n        return {'x': torch.from_numpy(x), 'stamp': torch.from_numpy(self.stamp[s:s+60].copy()),\n                'loss_mask': torch.from_numpy(np.arange(59) >= 55),\n                'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale),\n                'truth': torch.from_numpy(raw[56:].copy()), 'last_close': torch.tensor(raw[55,3])}\n\nm6_train = M6Dataset(m6_df, m6_train_ix)\nm6_dev = M6Dataset(m6_df, m6_dev_ix)\nm6_probe = m6_train.values[m6_train.starts[0]:m6_train.starts[0]+60].copy()\nm6_changed = m6_probe.copy()\nm6_changed[56:] += np.float32(10000)\nm6_before, m6_after = m6_normalize(m6_probe), m6_normalize(m6_changed)\nassert np.array_equal(m6_before[0][:56], m6_after[0][:56])\nassert np.array_equal(m6_before[1], m6_after[1]) and np.array_equal(m6_before[2], m6_after[2])\nM6_DATA_CHECKS = {'loaded_years': list(range(2010, 2021)), 'normalization_future_invariant': True,\n                  'train_windows': len(m6_train), 'development_windows': len(m6_dev),\n                  'development_ny_sessions': int(m6_dev_ix.ny_date.nunique()),\n                  'causal_model_checks': 'PENDING', 'predictive_experiments': 'NOT_EXECUTED'}\n"
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
