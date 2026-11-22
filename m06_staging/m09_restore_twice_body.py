# M09_RESTORE_TWICE_V1
# RESTORE_SOURCE is embedded locally from the hash-pinned daily restoration function.
def validate_restore_twice():
    import copy, datetime, gc, hashlib, json, random
    from pathlib import Path
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    root=Path('/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571')
    result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status':'BLOCKED', 'rounds':[], 'optimizer_steps':0, 'test_read':False}
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    def net_hash(net):
        h=hashlib.sha256()
        for k,v in sorted(net.state_dict().items()):
            h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    production=globals().get('AXXEL_SESSION')
    before=[net_hash(production[k]) for k in ('model','tokenizer')] if production else None
    try:
        for iteration in range(2):
            namespace={'__name__':'m09_isolated_restore_'+str(iteration)}
            exec(compile(RESTORE_SOURCE,'verified_daily_restore_function','exec'),namespace)
            session,report=namespace['_axxel_restore'](None)
            assert report['reused'] is False
            manifest=json.loads((root/'fase_IV/checksums.json').read_text())
            config=root/'fase_IV/configuraciones/protocolo_previo.json'
            assert sha(config)==manifest['configuraciones/protocolo_previo.json']['sha256']
            class LoRA(nn.Module):
                def __init__(self,base,rank):
                    super().__init__();self.base=base.requires_grad_(False)
                    self.lora_A=nn.Parameter(torch.zeros(rank,base.in_features,device=base.weight.device))
                    self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device))
                    self.dropout=nn.Dropout(.1)
                def forward(self,x):return self.base(x)+2*F.linear(F.linear(self.dropout(x),self.lora_A),self.lora_B)
            pilots=[]
            for variant in json.loads(config.read_text())['variants']:
                prefix='checkpoints/Pilot_'+variant['id']+'_s271828/'
                packet=session['adapter_packets'][prefix+'adapter_final.pt']
                resume=session['adapter_packets'][prefix+'resume.pt']
                net=copy.deepcopy(session['model'])
                for name,rank in variant['ranks'].items():
                    parent,leaf=name.rsplit('.',1);obj=net.get_submodule(parent)
                    setattr(obj,leaf,LoRA(getattr(obj,leaf),rank))
                sd=net.state_dict()
                assert {k for k in sd if 'lora_' in k}==set(packet['adapter'])
                assert all(torch.equal(v,resume['adapter'][k]) for k,v in packet['adapter'].items())
                sd.update(packet['adapter']);net.load_state_dict(sd,strict=True)
                assert all(torch.equal(net.state_dict()[k].cpu(),v) for k,v in packet['adapter'].items())
                opt=torch.optim.AdamW([p for n,p in net.named_parameters() if 'lora_' in n],lr=1e-4)
                opt.load_state_dict(resume['optimizer'])
                gen=np.random.default_rng();gen.bit_generator.state=resume['numpy_rng']
                tg=torch.Generator();tg.set_state(resume['torch_rng'].cpu())
                pilots.append({'variant':variant['id'],'step':resume['step'],'model_hash':net_hash(net),'strict_load':True,'optimizer_loaded':True,'rng_loaded_in_isolation':True})
                del net,opt,sd,packet,resume
            signature={'base':net_hash(session['model']),'tokenizer':net_hash(session['tokenizer']),
                       'heads':{str(k):net_hash(v) for k,v in session['heads'].items()},'pilots':pilots}
            result['rounds'].append({'restore':report,'signature':signature})
            del session,namespace
            gc.collect();torch.cuda.empty_cache()
        assert result['rounds'][0]['signature']==result['rounds'][1]['signature']
        result['restore_twice_passed']=True
        result['blockers']=['Scientific fingerprint deduplication remains uncertified; restoration alone does not authorize Phase V']
    except Exception as exc:
        result['blockers']=[type(exc).__name__+': '+str(exc)]
    finally:
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
        result['production_weights_unchanged']=before==([net_hash(production[k]) for k in ('model','tokenizer')] if production else None)
        result['production_rng_restored']=True
        print('M09_RESTORE_TWICE_RESULT '+json.dumps(result),flush=True)
    return result
M09_RESTORE_TWICE_RESULT=validate_restore_twice()
