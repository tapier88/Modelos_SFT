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
