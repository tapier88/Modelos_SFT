# M09_SCIENTIFIC_PREFLIGHT_V1
DATA_SOURCE="#@title MISIÓN 06 · 1. Datos autorizados, índice nuevo y normalización causal\nassert globals().get('M6_PREFLIGHT_OK') is True, 'Ejecutar primero M06 · 0'\nimport numpy as np\nimport pandas as pd\n\nM6_SOURCE = M6_BASE / 'datasets/processed/kronos_ohlcv_v1'\nM6_SOURCE_MANIFEST, m6_parts = [], []\n# Lista cerrada; no se enumeran directorios de datos, archivos ZIP ni índices históricos.\ndef m6_authorized_source(year):\n    if type(year) is not int or year not in range(2010, 2021):\n        raise ValueError('M06 solo permite archivos fuente 2010–2020')\n    return M6_SOURCE / f'XAUUSD_M15_BID_{year}.parquet'\n\nfor m6_year in range(2010, 2021):\n    m6_path = m6_authorized_source(m6_year)\n    m6_digest = m6_sha(m6_path)\n    m6_part = pd.read_parquet(m6_path)\n    m6_part['timestamp_utc'] = pd.to_datetime(m6_part.timestamp_utc, utc=True)\n    assert len(m6_part) and m6_part.timestamp_utc.dt.year.eq(m6_year).all()\n    assert m6_part.timestamp_utc.is_monotonic_increasing and m6_part.timestamp_utc.is_unique\n    assert m6_part.m1_candles.eq(15).all()\n    m6_prices = m6_part[['open', 'high', 'low', 'close', 'volume']]\n    assert np.isfinite(m6_prices.to_numpy()).all()\n    assert (m6_part.low > 0).all() and (m6_part.volume >= 0).all()\n    assert (m6_part.low <= m6_part[['open', 'close']].min(axis=1)).all()\n    assert (m6_part.high >= m6_part[['open', 'close']].max(axis=1)).all()\n    assert (m6_part.low <= m6_part.high).all()\n    assert m6_sha(m6_path) == m6_digest, 'Fuente cambió durante lectura'\n    M6_SOURCE_MANIFEST.append({'year': m6_year, 'file': m6_path.name, 'rows': len(m6_part), 'sha256': m6_digest})\n    m6_parts.append(m6_part[['timestamp_utc', 'open', 'high', 'low', 'close', 'volume']].copy())\nm6_df = pd.concat(m6_parts, ignore_index=True)\ndel m6_parts, m6_part, m6_prices\nassert m6_df.timestamp_utc.is_unique and m6_df.timestamp_utc.is_monotonic_increasing\nm6_ts = m6_df.timestamp_utc.astype('int64').to_numpy()\nM6_STEP = 15 * 60 * 10**9\nassert np.all(m6_ts % M6_STEP == 0)\nm6_df['split'] = np.where(m6_df.timestamp_utc.dt.year <= 2019, 'train', 'development')\nm6_df['segment'] = (m6_df.timestamp_utc.diff().ne(pd.Timedelta(minutes=15)) |\n                    m6_df.split.ne(m6_df.split.shift())).cumsum()\nm6_windows = []\nfor m6_segment, m6_group in m6_df.groupby('segment', sort=False):\n    m6_first, m6_last = int(m6_group.index[0]), int(m6_group.index[-1])\n    m6_starts = np.arange(m6_first, m6_last - 59 + 1, dtype=np.int64)\n    if not len(m6_starts):\n        continue\n    m6_decisions = m6_df.timestamp_utc.iloc[m6_starts + 56]\n    m6_ny = m6_decisions.dt.tz_convert('America/New_York')\n    for s, decision, ny in zip(m6_starts, m6_decisions, m6_ny):\n        if ny.strftime('%H:%M') not in M6_POLICY['ny_anchors']:\n            continue\n        m6_windows.append({'start_row': int(s), 'split': m6_group.split.iloc[0],\n                           'decision_utc': decision, 'decision_ny': ny.isoformat(),\n                           'ny_date': ny.strftime('%Y-%m-%d'), 'ny_hour': ny.strftime('%H:%M'),\n                           'context_start': m6_df.timestamp_utc.iloc[s],\n                           'target_end': m6_df.timestamp_utc.iloc[s+59] + pd.Timedelta(minutes=15)})\nm6_ix = pd.DataFrame(m6_windows).sort_values('decision_utc').reset_index(drop=True)\nassert len(m6_ix) and m6_ix.decision_utc.is_unique\nm6_train_ix = m6_ix[m6_ix.split.eq('train')].reset_index(drop=True)\nm6_dev_ix = m6_ix[m6_ix.split.eq('development')].reset_index(drop=True)\nassert len(m6_train_ix) and len(m6_dev_ix)\nassert m6_dev_ix.decision_utc.dt.year.eq(2020).all()\nassert m6_train_ix.target_end.max() < m6_dev_ix.context_start.min()\nassert (m6_ix.decision_utc.iloc[1:].to_numpy() >= m6_ix.target_end.iloc[:-1].to_numpy()).all()\nm6_starts = m6_ix.start_row.to_numpy(np.int64)\nassert np.all(m6_ts[m6_starts[:, None] + np.arange(60)] == m6_ts[m6_starts, None] + np.arange(60) * M6_STEP)\n\ndef m6_normalize(raw):\n    raw = np.asarray(raw, dtype=np.float32)\n    if raw.shape != (60, 6):\n        raise ValueError('Contrato M06: 60 filas, OHLCV + amount=0')\n    mean = raw[:56].mean(axis=0)\n    scale = raw[:56].std(axis=0) + np.float32(1e-5)\n    return np.clip((raw - mean) / scale, -5, 5), mean, scale\n\nclass M6Dataset(torch.utils.data.Dataset):\n    def __init__(self, frame, index):\n        self.L, self.H = 56, 4\n        self.index = index.reset_index(drop=True)\n        self.starts = self.index.start_row.to_numpy(np.int64)\n        self.values = np.column_stack([frame[['open','high','low','close','volume']].to_numpy(np.float32), np.zeros(len(frame), np.float32)])\n        t = frame.timestamp_utc.dt\n        self.stamp = np.column_stack([t.minute, t.hour, t.weekday, t.day, t.month]).astype(np.float32)\n    def __len__(self):\n        return len(self.starts)\n    def __getitem__(self, i):\n        s = int(self.starts[i])\n        raw = self.values[s:s+60].copy()\n        x, mean, scale = m6_normalize(raw)\n        return {'x': torch.from_numpy(x), 'stamp': torch.from_numpy(self.stamp[s:s+60].copy()),\n                'loss_mask': torch.from_numpy(np.arange(59) >= 55),\n                'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale),\n                'truth': torch.from_numpy(raw[56:].copy()), 'last_close': torch.tensor(raw[55,3])}\n\nm6_train = M6Dataset(m6_df, m6_train_ix)\nm6_dev = M6Dataset(m6_df, m6_dev_ix)\nm6_probe = m6_train.values[m6_train.starts[0]:m6_train.starts[0]+60].copy()\nm6_changed = m6_probe.copy()\nm6_changed[56:] += np.float32(10000)\nm6_before, m6_after = m6_normalize(m6_probe), m6_normalize(m6_changed)\nassert np.array_equal(m6_before[0][:56], m6_after[0][:56])\nassert np.array_equal(m6_before[1], m6_after[1]) and np.array_equal(m6_before[2], m6_after[2])\nM6_DATA_CHECKS = {'loaded_years': list(range(2010, 2021)), 'normalization_future_invariant': True,\n                  'train_windows': len(m6_train), 'development_windows': len(m6_dev),\n                  'development_ny_sessions': int(m6_dev_ix.ny_date.nunique()),\n                  'causal_model_checks': 'PENDING', 'predictive_experiments': 'NOT_EXECUTED'}\n"
PROTOCOL={'experiment': 'M06_FV_JOINT_RETURN_V1', 'status': 'PROPOSED_NOT_EXECUTED', 'created_date_Bogota': '2026-11-20', 'purpose': 'paired head-only continuation versus joint LoRA+head, without rerunning phases I-IV', 'canonical_run': '/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571', 'output_subdirectory_proposed': 'fase_V_joint_v1/<scientific_fingerprint>/<arm>/seed_<seed>', 'seeds': [271828, 271829, 271830], 'data': {'train_years': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019], 'development_years': [2020], 'train_windows': 8428, 'development_windows': 950, 'development_NY_sessions': 243, 'source_manifest_sha256': 'a4f37d6518acd433712e404ba7a35a0800d5dacda1c87dbebba9fa9a234d1b7b', 'L': 56, 'H': 4, 'timeframe': 'M15', 'anchors_NY': ['08:30', '09:30', '10:30', '11:30'], 'normalization': 'float32 context56 only; std(ddof=0)+1e-5; clip[-5,5]', 'test2023_2024_access': False, 'evaluation2021_2022_access': False}, 'base_hash': 'a3c79ed2ca1f5b35c921fce7cf673db515dd7593e5552c643a977f8dc96ae6a9', 'tokenizer_hash': 'e7885d9d7874cde53113b1cc43cbdf8f05941fb8d95bce0c38d03183665e4740', 'pipeline_sha256': '1424fb451942c7e1e1823808405f428283f9b1aacf7b34d08df7ca9edb503016', 'head_initialization': {'271828': {'model': 'Head_Huber_s271828', 'sha256': 'e2491f96d6a45db2cc8ea4dfed063c24044e56e3af0228065f5ecf51af81b254', 'best_step': 250, 'path': '/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571/fase_III/checkpoints/fase_III/7587400db64558e6/Head_Huber_s271828_best.pt'}, '271829': {'model': 'Head_Huber_s271829', 'sha256': '91648fafacf98c6f666e5692ab8b742a8f11da61b7321d3b2f3f6c4f9f2171aa', 'best_step': 200, 'path': '/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571/fase_III/checkpoints/fase_III/7587400db64558e6/Head_Huber_s271829_best.pt'}, '271830': {'model': 'Head_Huber_s271830', 'sha256': '8b20807ec220bbb888a0c63650f41c69a03f49278645f90e7052acf8ad18346e', 'best_step': 150, 'path': '/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571/fase_III/checkpoints/fase_III/7587400db64558e6/Head_Huber_s271830_best.pt'}}, 'reuse_M05_adapter': False, 'reuse_IV_adapter_for_training': False, 'shared_model': {'base_frozen': True, 'tokenizer_frozen': True, 'feature': 'decode_s1 last context token,512 dimensions,56 past candles only', 'head': 'Linear512x64+GELU+Linear64x4', 'head_parameters': 33092, 'feature_scalers': 'fixed train-only phaseIII scalers; verify SHA and feature identity', 'scalers_sha256': 'd39212a67cafe026976b9b36b51c712229ce1cd0613a3eb24f49f3a614ac579e', 'return_target': 'cumulative basis-point returns H1,H2,H3,H4', 'price_prediction': 'last_close*(1+predicted_bps/10000)'}, 'arms': {'control_head_only': {'head_trainable': True, 'base_and_adapters_frozen': True, 'trainable_parameters': 33092}, 'joint_QV_head': {'head_trainable': True, 'lora_targets': ['transformer.0.self_attn.q_proj', 'transformer.0.self_attn.v_proj', 'transformer.1.self_attn.q_proj', 'transformer.1.self_attn.v_proj', 'transformer.2.self_attn.q_proj', 'transformer.2.self_attn.v_proj', 'transformer.3.self_attn.q_proj', 'transformer.3.self_attn.v_proj', 'transformer.4.self_attn.q_proj', 'transformer.4.self_attn.v_proj', 'transformer.5.self_attn.q_proj', 'transformer.5.self_attn.v_proj', 'transformer.6.self_attn.q_proj', 'transformer.6.self_attn.v_proj', 'transformer.7.self_attn.q_proj', 'transformer.7.self_attn.v_proj'], 'rank': 8, 'alpha': 16, 'scale': 2, 'dropout': 0.1, 'initialization': 'fresh A Kaiming uniform, B zero; seed-matched phaseIII Huber head', 'lora_parameters': 131072, 'trainable_parameters': 164164, 'dependency_cross_attention_adapted': False}}, 'loss': {'type': 'SmoothL1_Huber', 'beta': 1.0, 'reduction': 'equal mean over batch and4 horizons', 'target_scaling': 'return_scale per horizon from train-only phaseIII scalers, fixed', 'CE_weight': 0, 'reason': 'retain phaseIII objective to isolate trainability; price-space objective is a later separately registered ablation'}, 'optimizer': {'name': 'AdamW', 'head_lr': 0.0003, 'lora_lr': 0.0001, 'weight_decay': 0.0, 'gradient_clip_norm': 1.0, 'schedule': 'constant'}, 'budget': {'arms': 2, 'seeds': 3, 'runs': 6, 'additional_steps_each': 300, 'maximum_total_steps': 1800, 'effective_batch': 64, 'microbatch': 16, 'accumulation': 4, 'checkpoint_every_steps': 25, 'development_evaluations': 'step0 for reload equivalence; fixed final step300 only', 'select_best_intermediate_2020': False, 'stop_on': 'nonfinite,hash mismatch,causality violation,missing gradients,version conflict; no metric-driven extension'}, 'mode': {'base': 'eval but autograd enabled for joint hidden path', 'tokenizer': 'eval no_grad', 'head': 'train', 'lora_dropout': 'train in joint arm only'}, 'randomness': {'order': 'same NumPy PCG64 training indices per seed in both arms', 'deterministic_algorithms': True, 'TF32': False, 'resume': 'optimizer,step,Python/NumPy/torchCPU/CUDA RNG,training order/scalers/config/source hashes'}, 'evaluation': {'primary': 'MAE of de-normalized close prices across950x4 forecasts', 'secondary': ['RMSE', 'direction accuracy excluding true flat moves', 'bias', 'magnitude MAE', 'NY hour/session', 'horizon', 'time per window'], 'references': ['Persistencia', 'Kronos_base greedy', 'each original phaseIII Huber head by seed', 'best observed phaseIII Huber_s271830 (selection-aware)', 'paired head-only continuation'], 'bootstrap': {'unit': 'NY session', 'circular_block_sessions': 5, 'replicates': 10000, 'seed': 271828, 'paired': True, 'confirmatory_development_contrasts': 9, 'contrasts': '3seeds x [persistence,best storedIII,matched new control]', 'simultaneous_intervals': 'Bonferroni percentile bounds alpha=.05/9, alongside exploratory95percent'}, 'candidate_followup_gate': 'all3seeds improve point MAE against persistence,bestIII and matched control; simultaneous intervals below0; no positive deltaMAE by horizon/hour; no promotion on2020 alone', 'independent_holdout': False, 'known_bias': '2020 already reused to select phaseIII checkpoints and earlier methods; cannot undo prior exposure'}, 'deduplication': {'fingerprint_fields': 'model/tokenizer/source hashes,initial head hashes,targets,ranks,loss,optimizer,budget,seed,arm', 'exclude_from_scientific_fingerprint': ['timestamps', 'paths', 'human labels'], 'existing_completed': 'verify manifest and load predictions; skip all training', 'existing_partial': 'verify and resume last committed step; never start from0', 'untracked_or_mismatched': 'fail closed, preserve files, reconcile without overwrite', 'same_method_different_folder': 'still considered duplicate'}, 'preflight_gates': ['backup both notebook states if conflict', 'one notebook writer:Colab UI', 'fresh runtime and GPU/CPU memory observed', 'base/tokenizer/current checkpoint hashes verified', 'strict shape/key loading', 'step0 reproduces phaseIII saved predictions with fixed documented numeric tolerance', 'future-invariance of context representation and regression outputs', 'head and LoRA_B have finite nonzero gradients,LoRA_A allowed zero at first step when B=0', 'base/tokenizer gradients absent', 'fresh empty namespace restore ofcompleted phasesIII-IV tested twice', 'save/readback and conflict detection pass', 'duplicate search across candidate manifests complete'], 'orders_enabled': False, 'training_started': False}
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
