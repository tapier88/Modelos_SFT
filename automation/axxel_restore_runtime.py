# AXXEL_RUNTIME_RESTORE_V1
"""Colab payload: restore verified objects, never train or write audited artifacts."""

def _axxel_restore(existing):
    import datetime
    import hashlib
    import importlib.util
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.nn import functional as F

    root = Path('/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571')
    base_hash = 'a3c79ed2ca1f5b35c921fce7cf673db515dd7593e5552c643a977f8dc96ae6a9'
    tok_hash = 'e7885d9d7874cde53113b1cc43cbdf8f05941fb8d95bce0c38d03183665e4740'

    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda: f.read(1048576), b''):
                h.update(b)
        return h.hexdigest()

    def net_hash(net):
        h = hashlib.sha256()
        for name, tensor in sorted(net.state_dict().items()):
            h.update(name.encode()); h.update(str(tuple(tensor.shape)).encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    assert os.path.ismount('/content/drive') and root.is_dir(), 'DRIVE_AUTH_REQUIRED'
    assert torch.cuda.is_available(), 'GPU unavailable; no silent CPU fallback'
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
    others = [int(x.strip()) for x in gpu.stdout.splitlines() if x.strip().isdigit() and int(x.strip()) != os.getpid()]
    assert not others, 'Other GPU processes active; restoration not started'
    # Existing state is reused only after checking actual model weights.
    if existing and existing.get('fingerprint') == base_hash + tok_hash:
        assert net_hash(existing['model']) == base_hash and net_hash(existing['tokenizer']) == tok_hash
        report = dict(existing['report'], reused=True, timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        return existing, report

    manifests = {}
    for phase, expected in [('fase_III', '731d1c5412d4836e16961ff2e3747e6fd4da124db81eaf20265a638b6c90d7a1'),
                            ('fase_IV', 'e9e0e3754deaafb0153d11dfce2bf0abd024669d2a253f2f7768c8a6bfa7effc')]:
        path = root / phase / 'checksums.json'
        assert digest(path) == expected, 'Manifest mismatch'
        manifests[phase] = json.loads(path.read_text())

    def verified(phase, relative):
        directory = (root / phase).resolve()
        path = (directory / relative).resolve()
        assert directory in path.parents and '..' not in Path(relative).parts
        expected = manifests[phase][relative]
        assert path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'], relative
        return path

    # Fixed audited source paths; do not execute the historical notebook or pipeline main.
    source_root = root.parents[1] / 'Kronos_SFT_verificado_20261107/ejecuciones/20261112T022707Z_55c83b'
    for relative, expected in [
        ('kronos_pipeline.py', '1424fb451942c7e1e1823808405f428283f9b1aacf7b34d08df7ca9edb503016'),
        ('vendor/model/kronos.py', '0a5f90282e2039c2de0771473419715c845def154896dbd0f5747837e6241032'),
        ('vendor/model/module.py', 'a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f')]:
        assert digest(source_root / relative) == expected, 'Audited source mismatch'
    # Disable bytecode writes beside audited files.
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location('axxel_verified_pipeline', source_root / 'kronos_pipeline.py')
        pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline)
        expected_revisions = {'Kronos-small': '901c26c1332695a2a8f243eb2f37243a37bea320',
                              'Kronos-Tokenizer-base': '0e0117387f39004a9016484a186a908917e22426'}
        assert pipeline.REVISIONS == expected_revisions
        from huggingface_hub import snapshot_download
        weights = Path('/content/axxel_runtime_cache')
        for repo, revision in expected_revisions.items():
            snapshot_download('NeoQuasar/' + repo, revision=revision, local_dir=str(weights / repo), allow_patterns=['config.json', 'model.safetensors'])
        model, tokenizer = pipeline.load_models(weights, torch.device('cuda'))
    finally:
        sys.dont_write_bytecode = old_bytecode
    model.requires_grad_(False).eval(); tokenizer.requires_grad_(False).eval()
    assert net_hash(model) == base_hash and net_hash(tokenizer) == tok_hash

    class Head(nn.Module):
        def __init__(self):
            super().__init__(); self.hidden = nn.Linear(512, 64); self.output = nn.Linear(64, 4)
        def forward(self, x):
            return self.output(F.gelu(self.hidden(x)))

    prefix = 'checkpoints/fase_III/7587400db64558e6/'
    with np.load(verified('fase_III', prefix + 'train_only_scalers.npz'), allow_pickle=False) as z:
        scalers = {k: z[k] for k in z.files}
    heads, resumes = {}, {}
    # Use saved 2020 development features only; no original datasets or reserved years.
    parts = []
    for name in sorted(manifests['fase_III']):
        if name.startswith(prefix + 'features_development_2020_') and name.endswith('.npz'):
            with np.load(verified('fase_III', name), allow_pickle=False) as z:
                parts.append({k: z[k] for k in ['ids', 'hidden', 'last']})
    cache = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    assert np.array_equal(cache['ids'], np.arange(950))
    features = cache['hidden'].astype(np.float32).copy()
    features -= scalers['hidden_mean']
    features /= scalers['hidden_scale']
    results = []
    with torch.random.fork_rng(devices=[0]):
        for seed in [271828, 271829, 271830]:
            packet = torch.load(verified('fase_III', prefix + f'Head_Huber_s{seed}_best.pt'), map_location='cpu', weights_only=True)
            assert packet['base_hash'] == base_hash and packet['tokenizer_hash'] == tok_hash
            head = Head().cuda(); head.load_state_dict(packet['head'], strict=True); head.requires_grad_(False).eval()
            resume = torch.load(verified('fase_III', prefix + f'Head_Huber_s{seed}_resume.pt'), map_location='cpu', weights_only=True)
            assert {'optimizer', 'numpy_rng', 'step'} <= set(resume)
            heads[seed] = head; resumes[seed] = resume
            with torch.no_grad():
                loc = head(torch.as_tensor(features, device='cuda')).cpu().numpy().astype(np.float64)
            predictions = cache['last'].astype(np.float64)[:, None] * (1 + loc * scalers['return_scale'] / 10000)
            ref = pd.read_parquet(verified('fase_III', f'metricas/Head_Huber_s{seed}.parquet'))
            saved = ref.pivot(index='window', columns='horizon', values='prediction').sort_index().to_numpy()
            delta = abs(predictions - saved)
            assert delta.max() <= .001 and delta.mean() <= .00001, 'Cached-feature prediction equivalence failed'
            results.append({'seed': seed, 'max_abs_price_diff': float(delta.max()), 'mean_abs_price_diff': float(delta.mean())})
    # Retain audited adapter/optimizer/RNG packets without applying pilots to the base model.
    adapters = {}
    for name in manifests['fase_IV']:
        if name.startswith('checkpoints/') and name.endswith('.pt'):
            adapters[name] = torch.load(verified('fase_IV', name), map_location='cpu', weights_only=True)
    report = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'RESTORED',
              'reused': False, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
              'base_hash': base_hash, 'tokenizer_hash': tok_hash, 'heads': 3, 'adapter_packets': len(adapters),
              'cached_feature_equivalence': results, 'optimizer_steps': 0, 'training_enabled': False,
              'resume_packets_loaded': True, 'full_data_preflight': 'NOT_RUN', 'test_read': False}
    state = {'fingerprint': base_hash + tok_hash, 'model': model, 'tokenizer': tokenizer, 'heads': heads,
             'scalers': scalers, 'resume_packets': resumes, 'adapter_packets': adapters, 'pipeline': pipeline, 'report': report}
    return state, report


import json as _axxel_json
AXXEL_SESSION, _axxel_report = _axxel_restore(globals().get('AXXEL_SESSION'))
print('AXXEL_RESTORE_RESULT', _axxel_json.dumps(_axxel_report, sort_keys=True))
