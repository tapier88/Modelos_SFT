RESTORE_SOURCE='# AXXEL_RUNTIME_RESTORE_V1\n"""Colab payload: restore verified objects, never train or write audited artifacts."""\n\ndef _axxel_restore(existing):\n    import datetime\n    import hashlib\n    import importlib.util\n    import json\n    import os\n    from pathlib import Path\n    import subprocess\n    import sys\n    import numpy as np\n    import pandas as pd\n    import torch\n    from torch import nn\n    from torch.nn import functional as F\n\n    root = Path(\'/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571\')\n    base_hash = \'a3c79ed2ca1f5b35c921fce7cf673db515dd7593e5552c643a977f8dc96ae6a9\'\n    tok_hash = \'e7885d9d7874cde53113b1cc43cbdf8f05941fb8d95bce0c38d03183665e4740\'\n\n    def digest(path):\n        h = hashlib.sha256()\n        with path.open(\'rb\') as f:\n            for b in iter(lambda: f.read(1048576), b\'\'):\n                h.update(b)\n        return h.hexdigest()\n\n    def net_hash(net):\n        h = hashlib.sha256()\n        for name, tensor in sorted(net.state_dict().items()):\n            h.update(name.encode()); h.update(str(tuple(tensor.shape)).encode())\n            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())\n        return h.hexdigest()\n\n    assert os.path.ismount(\'/content/drive\') and root.is_dir(), \'DRIVE_AUTH_REQUIRED\'\n    assert torch.cuda.is_available(), \'GPU unavailable; no silent CPU fallback\'\n    gpu = subprocess.run([\'nvidia-smi\', \'--query-compute-apps=pid\', \'--format=csv,noheader,nounits\'], capture_output=True, text=True, check=True, timeout=15)\n    others = [int(x.strip()) for x in gpu.stdout.splitlines() if x.strip().isdigit() and int(x.strip()) != os.getpid()]\n    assert not others, \'Other GPU processes active; restoration not started\'\n    # Existing state is reused only after checking actual model weights.\n    if existing and existing.get(\'fingerprint\') == base_hash + tok_hash:\n        assert net_hash(existing[\'model\']) == base_hash and net_hash(existing[\'tokenizer\']) == tok_hash\n        report = dict(existing[\'report\'], reused=True, timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())\n        return existing, report\n\n    manifests = {}\n    for phase, expected in [(\'fase_III\', \'731d1c5412d4836e16961ff2e3747e6fd4da124db81eaf20265a638b6c90d7a1\'),\n                            (\'fase_IV\', \'e9e0e3754deaafb0153d11dfce2bf0abd024669d2a253f2f7768c8a6bfa7effc\')]:\n        path = root / phase / \'checksums.json\'\n        assert digest(path) == expected, \'Manifest mismatch\'\n        manifests[phase] = json.loads(path.read_text())\n\n    def verified(phase, relative):\n        directory = (root / phase).resolve()\n        path = (directory / relative).resolve()\n        assert directory in path.parents and \'..\' not in Path(relative).parts\n        expected = manifests[phase][relative]\n        assert path.stat().st_size == expected[\'bytes\'] and digest(path) == expected[\'sha256\'], relative\n        return path\n\n    # Fixed audited source paths; do not execute the historical notebook or pipeline main.\n    source_root = root.parents[1] / \'Kronos_SFT_verificado_20261107/ejecuciones/20261112T022707Z_55c83b\'\n    for relative, expected in [\n        (\'kronos_pipeline.py\', \'1424fb451942c7e1e1823808405f428283f9b1aacf7b34d08df7ca9edb503016\'),\n        (\'vendor/model/kronos.py\', \'0a5f90282e2039c2de0771473419715c845def154896dbd0f5747837e6241032\'),\n        (\'vendor/model/module.py\', \'a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f\')]:\n        assert digest(source_root / relative) == expected, \'Audited source mismatch\'\n    # Disable bytecode writes beside audited files.\n    old_bytecode = sys.dont_write_bytecode\n    sys.dont_write_bytecode = True\n    try:\n        spec = importlib.util.spec_from_file_location(\'axxel_verified_pipeline\', source_root / \'kronos_pipeline.py\')\n        pipeline = importlib.util.module_from_spec(spec)\n        spec.loader.exec_module(pipeline)\n        expected_revisions = {\'Kronos-small\': \'901c26c1332695a2a8f243eb2f37243a37bea320\',\n                              \'Kronos-Tokenizer-base\': \'0e0117387f39004a9016484a186a908917e22426\'}\n        assert pipeline.REVISIONS == expected_revisions\n        from huggingface_hub import snapshot_download\n        weights = Path(\'/content/axxel_runtime_cache\')\n        for repo, revision in expected_revisions.items():\n            snapshot_download(\'NeoQuasar/\' + repo, revision=revision, local_dir=str(weights / repo), allow_patterns=[\'config.json\', \'model.safetensors\'])\n        model, tokenizer = pipeline.load_models(weights, torch.device(\'cuda\'))\n    finally:\n        sys.dont_write_bytecode = old_bytecode\n    model.requires_grad_(False).eval(); tokenizer.requires_grad_(False).eval()\n    assert net_hash(model) == base_hash and net_hash(tokenizer) == tok_hash\n\n    class Head(nn.Module):\n        def __init__(self):\n            super().__init__(); self.hidden = nn.Linear(512, 64); self.output = nn.Linear(64, 4)\n        def forward(self, x):\n            return self.output(F.gelu(self.hidden(x)))\n\n    prefix = \'checkpoints/fase_III/7587400db64558e6/\'\n    with np.load(verified(\'fase_III\', prefix + \'train_only_scalers.npz\'), allow_pickle=False) as z:\n        scalers = {k: z[k] for k in z.files}\n    heads, resumes = {}, {}\n    # Use saved 2020 development features only; no original datasets or reserved years.\n    parts = []\n    for name in sorted(manifests[\'fase_III\']):\n        if name.startswith(prefix + \'features_development_2020_\') and name.endswith(\'.npz\'):\n            with np.load(verified(\'fase_III\', name), allow_pickle=False) as z:\n                parts.append({k: z[k] for k in [\'ids\', \'hidden\', \'last\']})\n    cache = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}\n    assert np.array_equal(cache[\'ids\'], np.arange(950))\n    features = cache[\'hidden\'].astype(np.float32).copy()\n    features -= scalers[\'hidden_mean\']\n    features /= scalers[\'hidden_scale\']\n    results = []\n    with torch.random.fork_rng(devices=[0]):\n        for seed in [271828, 271829, 271830]:\n            packet = torch.load(verified(\'fase_III\', prefix + f\'Head_Huber_s{seed}_best.pt\'), map_location=\'cpu\', weights_only=True)\n            assert packet[\'base_hash\'] == base_hash and packet[\'tokenizer_hash\'] == tok_hash\n            head = Head().cuda(); head.load_state_dict(packet[\'head\'], strict=True); head.requires_grad_(False).eval()\n            resume = torch.load(verified(\'fase_III\', prefix + f\'Head_Huber_s{seed}_resume.pt\'), map_location=\'cpu\', weights_only=True)\n            assert {\'optimizer\', \'numpy_rng\', \'step\'} <= set(resume)\n            heads[seed] = head; resumes[seed] = resume\n            with torch.no_grad():\n                loc = head(torch.as_tensor(features, device=\'cuda\')).cpu().numpy().astype(np.float64)\n            predictions = cache[\'last\'].astype(np.float64)[:, None] * (1 + loc * scalers[\'return_scale\'] / 10000)\n            ref = pd.read_parquet(verified(\'fase_III\', f\'metricas/Head_Huber_s{seed}.parquet\'))\n            saved = ref.pivot(index=\'window\', columns=\'horizon\', values=\'prediction\').sort_index().to_numpy()\n            delta = abs(predictions - saved)\n            assert delta.max() <= .001 and delta.mean() <= .00001, \'Cached-feature prediction equivalence failed\'\n            results.append({\'seed\': seed, \'max_abs_price_diff\': float(delta.max()), \'mean_abs_price_diff\': float(delta.mean())})\n    # Retain audited adapter/optimizer/RNG packets without applying pilots to the base model.\n    adapters = {}\n    for name in manifests[\'fase_IV\']:\n        if name.startswith(\'checkpoints/\') and name.endswith(\'.pt\'):\n            adapters[name] = torch.load(verified(\'fase_IV\', name), map_location=\'cpu\', weights_only=True)\n    report = {\'timestamp_utc\': datetime.datetime.now(datetime.timezone.utc).isoformat(), \'status\': \'RESTORED\',\n              \'reused\': False, \'torch\': torch.__version__, \'cuda\': torch.version.cuda, \'gpu\': torch.cuda.get_device_name(0),\n              \'base_hash\': base_hash, \'tokenizer_hash\': tok_hash, \'heads\': 3, \'adapter_packets\': len(adapters),\n              \'cached_feature_equivalence\': results, \'optimizer_steps\': 0, \'training_enabled\': False,\n              \'resume_packets_loaded\': True, \'full_data_preflight\': \'NOT_RUN\', \'test_read\': False}\n    state = {\'fingerprint\': base_hash + tok_hash, \'model\': model, \'tokenizer\': tokenizer, \'heads\': heads,\n             \'scalers\': scalers, \'resume_packets\': resumes, \'adapter_packets\': adapters, \'pipeline\': pipeline, \'report\': report}\n    return state, report\n\n'
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
