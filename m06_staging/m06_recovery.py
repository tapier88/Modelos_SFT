#@title MISIÓN 06 · R. Recuperación verificada del runtime y cachés M06
import sys, os, json, hashlib, platform, importlib.metadata as m6_metadata
from pathlib import Path
from datetime import datetime, timezone
from IPython.display import display, Markdown
import torch
from google.colab import drive
M6_BASE=Path('/content/drive/MyDrive/AXXEL_CAPITAL')
M6_DIR=M6_BASE/'MISION06_ALINEACION/20261114T222455Z_32d5e571'
if not Path('/content/drive/MyDrive').exists(): drive.mount('/content/drive')
assert M6_DIR.is_dir(), 'Falta la ejecución M06 que se va a recuperar'
M6_DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert M6_DEVICE.type=='cuda', 'La recuperación requiere la GPU disponible para el presupuesto acordado'
def m6_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()
M6_POLICY=json.loads((M6_DIR/'configuraciones/politica.json').read_text())
M6_PROVENANCE=json.loads((M6_DIR/'auditoria/procedencia.json').read_text())
M6_HIST=Path(M6_PROVENANCE[0]['path']).parent
M6_CHECKPOINT=Path(M6_PROVENANCE[-1]['path'])
for item in M6_PROVENANCE: assert m6_sha(item['path'])==item['sha256']
M6_PREFLIGHT_OK=True
print('RECUPERANDO',str(M6_DIR),'GPU',torch.cuda.get_device_name(0),flush=True)
#@title MISIÓN 06 · 1. Datos autorizados, índice nuevo y normalización causal
assert globals().get('M6_PREFLIGHT_OK') is True, 'Ejecutar primero M06 · 0'
import numpy as np
import pandas as pd

M6_SOURCE = M6_BASE / 'datasets/processed/kronos_ohlcv_v1'
M6_SOURCE_MANIFEST, m6_parts = [], []
# Lista cerrada; no se enumeran directorios de datos, archivos ZIP ni índices históricos.
def m6_authorized_source(year):
    if type(year) is not int or year not in range(2010, 2021):
        raise ValueError('M06 solo permite archivos fuente 2010–2020')
    return M6_SOURCE / f'XAUUSD_M15_BID_{year}.parquet'

for m6_year in range(2010, 2021):
    m6_path = m6_authorized_source(m6_year)
    m6_digest = m6_sha(m6_path)
    m6_part = pd.read_parquet(m6_path)
    m6_part['timestamp_utc'] = pd.to_datetime(m6_part.timestamp_utc, utc=True)
    assert len(m6_part) and m6_part.timestamp_utc.dt.year.eq(m6_year).all()
    assert m6_part.timestamp_utc.is_monotonic_increasing and m6_part.timestamp_utc.is_unique
    assert m6_part.m1_candles.eq(15).all()
    m6_prices = m6_part[['open', 'high', 'low', 'close', 'volume']]
    assert np.isfinite(m6_prices.to_numpy()).all()
    assert (m6_part.low > 0).all() and (m6_part.volume >= 0).all()
    assert (m6_part.low <= m6_part[['open', 'close']].min(axis=1)).all()
    assert (m6_part.high >= m6_part[['open', 'close']].max(axis=1)).all()
    assert (m6_part.low <= m6_part.high).all()
    assert m6_sha(m6_path) == m6_digest, 'Fuente cambió durante lectura'
    M6_SOURCE_MANIFEST.append({'year': m6_year, 'file': m6_path.name, 'rows': len(m6_part), 'sha256': m6_digest})
    m6_parts.append(m6_part[['timestamp_utc', 'open', 'high', 'low', 'close', 'volume']].copy())
m6_df = pd.concat(m6_parts, ignore_index=True)
del m6_parts, m6_part, m6_prices
assert m6_df.timestamp_utc.is_unique and m6_df.timestamp_utc.is_monotonic_increasing
m6_ts = m6_df.timestamp_utc.astype('int64').to_numpy()
M6_STEP = 15 * 60 * 10**9
assert np.all(m6_ts % M6_STEP == 0)
m6_df['split'] = np.where(m6_df.timestamp_utc.dt.year <= 2019, 'train', 'development')
m6_df['segment'] = (m6_df.timestamp_utc.diff().ne(pd.Timedelta(minutes=15)) |
                    m6_df.split.ne(m6_df.split.shift())).cumsum()
m6_windows = []
for m6_segment, m6_group in m6_df.groupby('segment', sort=False):
    m6_first, m6_last = int(m6_group.index[0]), int(m6_group.index[-1])
    m6_starts = np.arange(m6_first, m6_last - 59 + 1, dtype=np.int64)
    if not len(m6_starts):
        continue
    m6_decisions = m6_df.timestamp_utc.iloc[m6_starts + 56]
    m6_ny = m6_decisions.dt.tz_convert('America/New_York')
    for s, decision, ny in zip(m6_starts, m6_decisions, m6_ny):
        if ny.strftime('%H:%M') not in M6_POLICY['ny_anchors']:
            continue
        m6_windows.append({'start_row': int(s), 'split': m6_group.split.iloc[0],
                           'decision_utc': decision, 'decision_ny': ny.isoformat(),
                           'ny_date': ny.strftime('%Y-%m-%d'), 'ny_hour': ny.strftime('%H:%M'),
                           'context_start': m6_df.timestamp_utc.iloc[s],
                           'target_end': m6_df.timestamp_utc.iloc[s+59] + pd.Timedelta(minutes=15)})
m6_ix = pd.DataFrame(m6_windows).sort_values('decision_utc').reset_index(drop=True)
assert len(m6_ix) and m6_ix.decision_utc.is_unique
m6_train_ix = m6_ix[m6_ix.split.eq('train')].reset_index(drop=True)
m6_dev_ix = m6_ix[m6_ix.split.eq('development')].reset_index(drop=True)
assert len(m6_train_ix) and len(m6_dev_ix)
assert m6_dev_ix.decision_utc.dt.year.eq(2020).all()
assert m6_train_ix.target_end.max() < m6_dev_ix.context_start.min()
assert (m6_ix.decision_utc.iloc[1:].to_numpy() >= m6_ix.target_end.iloc[:-1].to_numpy()).all()
m6_starts = m6_ix.start_row.to_numpy(np.int64)
assert np.all(m6_ts[m6_starts[:, None] + np.arange(60)] == m6_ts[m6_starts, None] + np.arange(60) * M6_STEP)

def m6_normalize(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (60, 6):
        raise ValueError('Contrato M06: 60 filas, OHLCV + amount=0')
    mean = raw[:56].mean(axis=0)
    scale = raw[:56].std(axis=0) + np.float32(1e-5)
    return np.clip((raw - mean) / scale, -5, 5), mean, scale

class M6Dataset(torch.utils.data.Dataset):
    def __init__(self, frame, index):
        self.L, self.H = 56, 4
        self.index = index.reset_index(drop=True)
        self.starts = self.index.start_row.to_numpy(np.int64)
        self.values = np.column_stack([frame[['open','high','low','close','volume']].to_numpy(np.float32), np.zeros(len(frame), np.float32)])
        t = frame.timestamp_utc.dt
        self.stamp = np.column_stack([t.minute, t.hour, t.weekday, t.day, t.month]).astype(np.float32)
    def __len__(self):
        return len(self.starts)
    def __getitem__(self, i):
        s = int(self.starts[i])
        raw = self.values[s:s+60].copy()
        x, mean, scale = m6_normalize(raw)
        return {'x': torch.from_numpy(x), 'stamp': torch.from_numpy(self.stamp[s:s+60].copy()),
                'loss_mask': torch.from_numpy(np.arange(59) >= 55),
                'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale),
                'truth': torch.from_numpy(raw[56:].copy()), 'last_close': torch.tensor(raw[55,3])}

m6_train = M6Dataset(m6_df, m6_train_ix)
m6_dev = M6Dataset(m6_df, m6_dev_ix)
m6_probe = m6_train.values[m6_train.starts[0]:m6_train.starts[0]+60].copy()
m6_changed = m6_probe.copy()
m6_changed[56:] += np.float32(10000)
m6_before, m6_after = m6_normalize(m6_probe), m6_normalize(m6_changed)
assert np.array_equal(m6_before[0][:56], m6_after[0][:56])
assert np.array_equal(m6_before[1], m6_after[1]) and np.array_equal(m6_before[2], m6_after[2])
M6_DATA_CHECKS = {'loaded_years': list(range(2010, 2021)), 'normalization_future_invariant': True,
                  'train_windows': len(m6_train), 'development_windows': len(m6_dev),
                  'development_ny_sessions': int(m6_dev_ix.ny_date.nunique()),
                  'causal_model_checks': 'PENDING', 'predictive_experiments': 'NOT_EXECUTED'}

assert M6_SOURCE_MANIFEST==json.loads((M6_DIR/'auditoria/datasets_autorizados.json').read_text())
pd.testing.assert_frame_equal(m6_ix,pd.read_parquet(M6_DIR/'auditoria/ventanas_m06.parquet'))
print('DATOS_RECUPERADOS_Y_HASHES_VERIFICADOS',len(m6_train),len(m6_dev),flush=True)
#@title MISIÓN 06 · 2. Modelos congelados, causalidad y protocolo experimental
assert globals().get('M6_PREFLIGHT_OK') and 'm6_dev' in globals()
import os, sys, time, math, random, copy, inspect, importlib.util
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import default_collate
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(min(2, os.cpu_count() or 1))

def m6_save_json(rel, obj):
    path = (M6_DIR / rel).resolve()
    assert M6_DIR.resolve() in path.parents
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)

def m6_seed(seed=271828):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def m6_hash(net, base=False):
    h = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        if base and ('lora_A' in name or 'lora_B' in name): continue
        name = name.replace('.base.', '.') if base else name
        h.update(name.encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

assert m6_sha(M6_HIST/'kronos_pipeline.py') == M6_PROVENANCE[0]['sha256']
spec = importlib.util.spec_from_file_location('m6_kp', M6_HIST/'kronos_pipeline.py')
m6_kp = importlib.util.module_from_spec(spec); spec.loader.exec_module(m6_kp)
assert m6_kp.COMMIT == '67b630e67f6a18c9e9be918d9b4337c960db1e9a'
M6_WEIGHTS = Path('/content/kronos_weights_pinned')
for repo, rev in m6_kp.REVISIONS.items():
    snapshot_download('NeoQuasar/'+repo, revision=rev, local_dir=str(M6_WEIGHTS/repo), allow_patterns=['config.json','model.safetensors'])
m6_model, m6_tok = m6_kp.load_models(M6_WEIGHTS, M6_DEVICE)
m6_model.requires_grad_(False).eval(); m6_tok.requires_grad_(False).eval()
assert len(m6_model.transformer) == 8 and m6_model.d_model == 512
assert (m6_tok.s1_bits, m6_tok.s2_bits) == (10, 10)
M6_BASE_HASH, M6_TOK_HASH = m6_hash(m6_model), m6_hash(m6_tok)
assert m6_sha(M6_CHECKPOINT) == M6_PROVENANCE[-1]['sha256']
m6_packet = torch.load(M6_CHECKPOINT, map_location='cpu', weights_only=False)
assert m6_packet['base_hash'] == M6_BASE_HASH and m6_packet['tokenizer_hash'] == M6_TOK_HASH
assert m6_packet['step'] == 1000
M6_ADAPTER_M05 = {k:v.clone() for k,v in m6_packet['adapter'].items()}
M6_M05_CONFIG = m6_packet['config']
del m6_packet

class M6LoRA(nn.Module):
    def __init__(self, base):
        super().__init__(); self.base = base.requires_grad_(False); self.enabled = True
        self.lora_A = nn.Parameter(torch.zeros(8, base.in_features, device=base.weight.device), requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, 8, device=base.weight.device), requires_grad=False)
        self.dropout = nn.Dropout(.10); self.scale = 2.
    def forward(self, x):
        y = self.base(x)
        return y + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)*self.scale if self.enabled else y

M6_TARGETS = []
for name, module in list(m6_model.named_modules()):
    if type(module).__name__ in ['MultiHeadAttentionWithRoPE','MultiHeadCrossAttentionWithRoPE']:
        for leaf in ['q_proj','v_proj']:
            layer = getattr(module, leaf)
            M6_TARGETS.append(name+'.'+leaf); setattr(module, leaf, M6LoRA(layer))
assert len(M6_TARGETS) == 18
M6_ADAPTER_PARAMS = {n:p for n,p in m6_model.named_parameters() if 'lora_' in n}
assert set(M6_ADAPTER_PARAMS) == set(M6_ADAPTER_M05)
assert sum(p.numel() for p in M6_ADAPTER_PARAMS.values()) == 147456
with torch.no_grad():
    for name, param in M6_ADAPTER_PARAMS.items(): param.copy_(M6_ADAPTER_M05[name].to(param.device))

def m6_mode(label):
    assert label in ['Kronos_base','Kronos_LoRA_M05']
    for name in M6_TARGETS: m6_model.get_submodule(name).enabled = label == 'Kronos_LoRA_M05'
    m6_model.eval(); m6_tok.eval()

def m6_batch(ds, ids):
    return {k:v.to(M6_DEVICE) for k,v in default_collate([ds[int(i)] for i in ids]).items()}

def m6_sync():
    if M6_DEVICE.type == 'cuda': torch.cuda.synchronize()

M6_CAUSAL = {}
for label in ['Kronos_base','Kronos_LoRA_M05']:
    m6_mode(label)
    M6_CAUSAL[label] = m6_kp.contract_tests(m6_model, m6_tok, m6_train, M6_DEVICE)
assert m6_hash(m6_model, True) == M6_BASE_HASH and m6_hash(m6_tok) == M6_TOK_HASH
assert not any(p.requires_grad for p in m6_model.parameters())

@torch.no_grad()
def m6_rollout(batch, paths=1, seed=271828, temperature=None, top_k=0, top_p=1.):
    # El predictor y sus representaciones reciben solo contexto observado.
    m6_model.eval(); m6_tok.eval()
    a,b = m6_tok.encode(batch['x'][:,:56], half=True)
    a=a.repeat_interleave(paths,0); b=b.repeat_interleave(paths,0)
    stamp=batch['stamp'].repeat_interleave(paths,0)
    generator=torch.Generator(device=M6_DEVICE).manual_seed(int(seed))
    def choose(logits):
        if temperature is None: return logits.argmax(-1, keepdim=True)
        logits=logits/float(temperature)
        if top_k:
            logits=logits.masked_fill(logits < logits.topk(top_k).values[:,-1:], -float('inf'))
        if top_p < 1:
            values,order=logits.sort(descending=True)
            remove=values.softmax(-1).cumsum(-1)>top_p
            remove[:,1:]=remove[:,:-1].clone(); remove[:,0]=False
            logits=logits.masked_fill(torch.zeros_like(remove).scatter(1,order,remove), -float('inf'))
        return torch.multinomial(logits.softmax(-1),1,generator=generator)
    for j in range(4):
        logits, hidden=m6_model.decode_s1(a,b,stamp=stamp[:,:56+j])
        na=choose(logits[:,-1])
        nb=choose(m6_model.decode_s2(hidden,torch.cat([a[:,1:],na],1))[:,-1])
        a=torch.cat([a,na],1); b=torch.cat([b,nb],1)
    prediction=m6_tok.decode([a,b],half=True)[:,-4:]
    prediction=prediction*batch['scale'].repeat_interleave(paths,0)[:,None]+batch['mean'].repeat_interleave(paths,0)[:,None]
    return prediction.reshape(len(batch['x']),paths,4,6), (a.reshape(len(batch['x']),paths,60),b.reshape(len(batch['x']),paths,60))

@torch.no_grad()
def m6_ce(batch):
    a,b=m6_tok.encode(batch['x'],half=True)
    l1,l2=m6_model(a[:,:-1],b[:,:-1],stamp=batch['stamp'][:,:-1],use_teacher_forcing=True,s1_targets=a[:,1:])
    ce1=F.cross_entropy(l1.transpose(1,2),a[:,1:],reduction='none')[:,55:]
    ce2=F.cross_entropy(l2.transpose(1,2),b[:,1:],reduction='none')[:,55:]
    return torch.stack([ce1,ce2],-1)

@torch.no_grad()
def m6_features(batch):
    a,b=m6_tok.encode(batch['x'][:,:56],half=True)
    return m6_model.decode_s1(a,b,stamp=batch['stamp'][:,:56])[1][:,-1]

m6_mode('Kronos_base'); probe=m6_batch(m6_train,[0,1])
pred,_=m6_rollout(probe)
assert torch.equal(pred[:,0],m6_kp.rollout(m6_model,m6_tok,probe,56,4))
alternate={k:v.clone() for k,v in probe.items()}; alternate['x'][:,56:]*=-1; alternate['truth'].zero_()
assert torch.equal(m6_features(probe),m6_features(alternate))
p1,_=m6_rollout(probe,paths=32,temperature=.8,seed=271828)
p2,_=m6_rollout(alternate,paths=32,temperature=.8,seed=271828)
assert torch.equal(p1,p2)
M6_PROFILE=[]
for paths in [1,32]:
    m6_sync(); start=time.perf_counter()
    _=m6_rollout(m6_batch(m6_train,range(4)),paths=paths,temperature=None if paths==1 else .8)
    m6_sync(); M6_PROFILE.append({'paths':paths,'windows':4,'seconds':time.perf_counter()-start})

M6_PLAN={'seed':271828,'replicate_seeds':[271828,271829,271830], 'L':56,'H':4,
 'train_windows':len(m6_train),'development_windows':len(m6_dev),'development_year':2020,
 'mc_configs':[{'id':'T06','temperature':.6,'top_k':0,'top_p':1.}, {'id':'T08','temperature':.8,'top_k':0,'top_p':1.}, {'id':'T10','temperature':1.,'top_k':0,'top_p':1.}, {'id':'K50_T08','temperature':.8,'top_k':50,'top_p':1.}, {'id':'P90_T08','temperature':.8,'top_k':0,'top_p':.9}],
 'mc_paths':[16,32], 'mc_aggregates':['mean','median'], 'mc_exploration_sessions':32,
 'mc_selection':'menor MAE exploratorio por modelo; una configuracion/metodo por modelo; replicar con 3 seeds en 2020 completo',
 'head_losses':['Huber','Laplace_NLL'],'head_steps':300,'head_batch':64,'head_lr':.001,'head_seeds':[271828,271829,271830],
 'head_selection':'checkpoint menor MAE 2020 cada 50 pasos; 2020 es desarrollo, no holdout independiente',
 'bootstrap':{'replicates':2000,'session_block':5,'seed':271828},
 'phase_IV':'Condicionada a resultados I–III; presupuesto y decision se documentan antes de entrenar',
 'reserved_evaluation_executed':False,'research_only':True,'orders_enabled':False}
days=np.array(sorted(m6_dev_ix.ny_date.unique()))
M6_MC_DAYS=days[np.linspace(0,len(days)-1,32,dtype=int)].tolist()
M6_MC_IDS=np.flatnonzero(m6_dev_ix.ny_date.isin(M6_MC_DAYS)).tolist()
M6_PLAN['mc_exploration_days']=M6_MC_DAYS
m6_save_json('configuraciones/protocolo_experimentos.json',M6_PLAN)
m6_save_json('auditoria/causalidad_modelos.json',M6_CAUSAL)
m6_save_json('auditoria/perfil_inferencia.json',M6_PROFILE)
m6_save_json('auditoria/modelos.json',{'base_hash':M6_BASE_HASH,'tokenizer_hash':M6_TOK_HASH,'commit':m6_kp.COMMIT,'revisions':m6_kp.REVISIONS,'checkpoint':str(M6_CHECKPOINT),'checkpoint_sha256':m6_sha(M6_CHECKPOINT),'lora_targets':M6_TARGETS,'trainable_parameters':0,'greedy_matches_M05_implementation':True,'hidden_and_sampling_future_invariant':True})
for source in ['model/kronos.py','model/module.py']:
    dest=M6_DIR/'auditoria'/Path(source).name
    dest.write_bytes((M6_HIST/'vendor'/source).read_bytes())
(M6_DIR/'auditoria/kronos_pipeline.py').write_bytes((M6_HIST/'kronos_pipeline.py').read_bytes())
M6_READY=True
print('MODELOS_Y_CAUSALIDAD_VERIFICADOS',M6_CAUSAL)
display(pd.DataFrame(M6_PROFILE))
print('PROTOCOLO_FIJADO_ANTES_DE_METRICAS_2020',len(M6_MC_IDS),'ventanas MC exploratorias')

for start in range(0,len(m6_dev),16):
    assert (M6_DIR/'checkpoints/fase_I'/f'batch_{start:04d}.npz').is_file(), 'Falta caché; no recalcular silenciosamente'
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


M6_PHASE_I_DONE=True
M6_RECOVERY={'utc':datetime.now(timezone.utc).isoformat(),'device':torch.cuda.get_device_name(0),
 'cached_phase_I_windows':len(M6_Y),'original_files_verified':True,'reason':'Runtime perdió variables al reconectar; MC detenido por assert antes de cálculo',
 'recovered_baselines':M6_BASELINE_TABLE.to_dict(orient='records')}
m6_save_json('auditoria/recuperacion_runtime.json',M6_RECOVERY)
display(M6_BASELINE_TABLE)
print('RECUPERACION_COMPLETA; ejecutar únicamente M06 · 4 para continuar')
