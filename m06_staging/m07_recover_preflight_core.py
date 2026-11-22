# M07_RECOVERY_PREFLIGHT_V1
import os, json, hashlib, datetime, importlib.util, importlib.metadata, random, copy
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np, pandas as pd, torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import default_collate

M7_ROOT=Path('/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571')
assert globals().get('verified',{}).get('fase_IV',{}).get('count')==37, 'Remote inventory must finish first'
assert 'fase_III' in verified
M7_OUT=M7_ROOT.parent/'m07_recovery'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
M7_OUT.mkdir(parents=True,exist_ok=False)
M7_CHECKS={'RESEARCH_ONLY':True,'orders_enabled':False,'test_read':False,'phase_V_started':False,'status':'running','output':str(M7_OUT)}
def m7_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def m7_log(label,value):
    M7_CHECKS[label]=value
    print('M07_PREFLIGHT',label,json.dumps(value,default=str),flush=True)
    path=M7_OUT/(str(len(M7_CHECKS))+'_'+label+'.json')
    with path.open('x') as f:json.dump(value,f,default=str,indent=2)
def m7_hash(net):
    h=hashlib.sha256()
    for name,value in sorted(net.state_dict().items()):
        h.update(name.encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
try:
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2)
    random.seed(271828);np.random.seed(271828);torch.manual_seed(271828)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(271828)
    M7_DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    provenance=json.loads((M7_ROOT/'auditoria/procedencia.json').read_text())
    for item in provenance[:3]:
        assert Path(item['path']).stat().st_size==item['bytes'] and m7_sha(item['path'])==item['sha256']
    assert provenance[0]['sha256']=='1424fb451942c7e1e1823808405f428283f9b1aacf7b34d08df7ca9edb503016'
    spec=importlib.util.spec_from_file_location('m07_verified_pipeline',provenance[0]['path'])
    M7_KP=importlib.util.module_from_spec(spec);spec.loader.exec_module(M7_KP)
    model_info=json.loads((M7_ROOT/'auditoria/modelos.json').read_text())
    assert M7_KP.REVISIONS==model_info['revisions'] and M7_KP.COMMIT==model_info['commit']
    from huggingface_hub import snapshot_download
    weights=Path('/content/m07_pinned_weights')
    for repo,revision in M7_KP.REVISIONS.items():
        snapshot_download('NeoQuasar/'+repo,revision=revision,local_dir=str(weights/repo),allow_patterns=['config.json','model.safetensors'])
    M7_MODEL,M7_TOK=M7_KP.load_models(weights,M7_DEVICE)
    M7_MODEL.requires_grad_(False).eval();M7_TOK.requires_grad_(False).eval()
    assert m7_hash(M7_MODEL)==model_info['base_hash'] and m7_hash(M7_TOK)==model_info['tokenizer_hash']
    m7_log('models_restored',{'base_hash':m7_hash(M7_MODEL),'tokenizer_hash':m7_hash(M7_TOK),'revisions':M7_KP.REVISIONS,'torch':torch.__version__,'numpy':np.__version__,'device':str(M7_DEVICE),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    M7_FDIR=M7_ROOT/'fase_III/checkpoints/fase_III/7587400db64558e6'
    assert m7_sha(M7_FDIR/'train_only_scalers.npz')=='d39212a67cafe026976b9b36b51c712229ce1cd0613a3eb24f49f3a614ac579e'
    with np.load(M7_FDIR/'train_only_scalers.npz',allow_pickle=False) as z:M7_SCALERS={k:z[k] for k in z.files}
    class M7Head(nn.Module):
        def __init__(self):
            super().__init__();self.hidden=nn.Linear(512,64);self.output=nn.Linear(64,4)
        def forward(self,x):return self.output(F.gelu(self.hidden(x)))
    M7_HEADS={};head_evidence=[]
    for seed in [271828,271829,271830]:
        packet=torch.load(M7_FDIR/f'Head_Huber_s{seed}_best.pt',map_location='cpu',weights_only=True)
        assert packet['base_hash']==model_info['base_hash'] and packet['tokenizer_hash']==model_info['tokenizer_hash']
        head=M7Head().to(M7_DEVICE);head.load_state_dict(packet['head'],strict=True);head.eval()
        repeated=M7Head().to(M7_DEVICE);repeated.load_state_dict(packet['head'],strict=True)
        assert m7_hash(head)==m7_hash(repeated)
        assert sum(p.numel() for p in head.parameters())==33092
        resume=torch.load(M7_FDIR/f'Head_Huber_s{seed}_resume.pt',map_location='cpu',weights_only=True)
        assert all(k in resume for k in ['optimizer','numpy_rng','step'])
        head_evidence.append({'seed':seed,'best_step':packet['best_step'],'resume_step':resume['step'],'strict_restore_twice':True,'resume_keys':sorted(resume.keys())})
        M7_HEADS[seed]=head
    m7_log('heads_restored',head_evidence)
    # Restore all five IV architectures strictly into disposable frozen copies; no optimizer steps.
    class M7LoRA(nn.Module):
        def __init__(self,base,rank):
            super().__init__();self.base=base.requires_grad_(False)
            self.lora_A=nn.Parameter(torch.zeros(rank,base.in_features,device=base.weight.device))
            self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device))
            self.dropout=nn.Dropout(.1)
        def forward(self,x):return self.base(x)+2*F.linear(F.linear(self.dropout(x),self.lora_A),self.lora_B)
    iv_evidence=[]
    ivspec=json.loads((M7_ROOT/'fase_IV/configuraciones/protocolo_previo.json').read_text())
    for variant in ivspec['variants']:
        folder=M7_ROOT/'fase_IV/checkpoints'/('Pilot_'+variant['id']+'_s271828')
        pkt=torch.load(folder/'adapter_final.pt',map_location='cpu',weights_only=True)
        resume=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(v,resume['adapter'][k]) for k,v in pkt['adapter'].items())
        net=copy.deepcopy(M7_MODEL)
        for name,rank in variant['ranks'].items():
            parent,leaf=name.rsplit('.',1);obj=net.get_submodule(parent)
            setattr(obj,leaf,M7LoRA(getattr(obj,leaf),rank))
        sd=net.state_dict()
        assert {k for k in sd if 'lora_' in k}==set(pkt['adapter'])
        sd.update({k:v.to(M7_DEVICE) for k,v in pkt['adapter'].items()})
        net.load_state_dict(sd,strict=True)
        assert all(torch.equal(net.state_dict()[k].cpu(),v) for k,v in pkt['adapter'].items())
        opt=torch.optim.AdamW([p for n,p in net.named_parameters() if 'lora_' in n],lr=1e-4)
        opt.load_state_dict(resume['optimizer'])
        rng=np.random.default_rng();rng.bit_generator.state=resume['numpy_rng']
        gen=torch.Generator();gen.set_state(resume['torch_rng'].cpu())
        iv_evidence.append({'variant':variant['id'],'step':resume['step'],'strict_restore':True,'optimizer_state_loaded':True,'numpy_torch_rng_restored_in_isolation':True,'cuda_rng_present':'cuda_rng' in resume})
        del opt,net,sd
    m7_log('phase_IV_restored',iv_evidence)
    M7_CHECKS['status']='recovered_pending_data_and_step0'
except Exception as exc:
    M7_CHECKS['status']='blocked';M7_CHECKS['error']=repr(exc)
    raise
finally:
    with (M7_OUT/'recovery_status.json').open('x') as f:json.dump(M7_CHECKS,f,indent=2,default=str)
    files={p.name:{'sha256':m7_sha(p),'bytes':p.stat().st_size} for p in M7_OUT.iterdir() if p.is_file()}
    with (M7_OUT/'checksums.json').open('x') as f:json.dump(files,f,indent=2)
    print('M07_RECOVERY_STATUS',json.dumps(M7_CHECKS,default=str),flush=True)
