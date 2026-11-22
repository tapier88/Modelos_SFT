# M08C_READONLY_DIAGNOSTIC_V1 — NOT AUTHORIZED FOR REMOTE EXECUTION YET
"""One-shot read-only inspection; no mount, load, install, training or file writes.
Only pinned phase manifests and their checkpoint entries are read.
The caller must verify notebook identity, avoid a busy kernel and obtain approval.
"""

def _m08c_inspect(namespace):
    import datetime
    import hashlib
    import importlib.metadata
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    def query(args):
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=20)
            return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
        except Exception as exc:
            return {'status': 'UNKNOWN', 'error': str(exc)}

    def sha(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    result = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'RESEARCH_ONLY': True, 'orders_enabled': False, 'test_read': False,
              'READY': False, 'training_active': 'UNKNOWN', 'pid': os.getpid()}
    result['gpu'] = query(['nvidia-smi', '--query-gpu=name,uuid,memory.total,memory.used,memory.free,driver_version', '--format=csv,noheader'])
    result['gpu_processes'] = query(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'])
    result['processes'] = query(['ps', '-eo', 'pid,ppid,comm'])
    result['system_memory'] = Path('/proc/meminfo').read_text()
    torch = sys.modules.get('torch')
    try:
        result['torch_installed_version'] = importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:
        result['torch_installed_version'] = 'NOT_INSTALLED'
    result['torch_loaded_version'] = getattr(torch, '__version__', None)
    result['torch_cuda_build'] = getattr(getattr(torch, 'version', None), 'cuda', None)
    result['cuda_runtime_usability'] = 'UNKNOWN: no CUDA initialization or tensor operation performed'
    # Inspect names/types only; do not invoke methods or restore model objects.
    result['candidate_objects'] = {name: type(obj).__module__ + '.' + type(obj).__name__
        for name, obj in namespace.items()
        if name.startswith(('M7_', 'M6_', 'm6_')) or name in ('model', 'tokenizer', 'predictor')}
    result['models_compatible'] = 'UNKNOWN: object names/types do not prove readiness'
    drive = Path('/content/drive/MyDrive/AXXEL_CAPITAL')
    result['drive'] = {'mountpoint': os.path.ismount('/content/drive'),
                       'AXXEL_CAPITAL_is_dir': drive.is_dir(),
                       'read_search_access': os.access(drive, os.R_OK | os.X_OK)}
    if not (result['drive']['mountpoint'] and result['drive']['AXXEL_CAPITAL_is_dir']):
        result['blocker'] = 'Drive unavailable; no mounting attempted. Human authorization may be required.'
        return result
    root = drive / 'MISION06_ALINEACION/20261114T222455Z_32d5e571'
    expected = {'fase_III': '731d1c5412d4836e16961ff2e3747e6fd4da124db81eaf20265a638b6c90d7a1',
                'fase_IV': 'e9e0e3754deaafb0153d11dfce2bf0abd024669d2a253f2f7768c8a6bfa7effc'}
    result['phases'] = {}
    for phase, known_hash in expected.items():
        directory = (root / phase).resolve()
        manifest = directory / 'checksums.json'
        actual = sha(manifest)
        entry = {'manifest_sha256': actual, 'manifest_matches_M07': actual == known_hash}
        result['phases'][phase] = entry
        if actual != known_hash:
            result['blocker'] = 'Manifest mismatch; checkpoint reads stopped'
            return result
        files = json.loads(manifest.read_text())
        entry['checkpoints'] = []
        for rel, metadata in files.items():
            relative = Path(rel)
            if not rel.startswith('checkpoints/') or relative.suffix != '.pt':
                continue
            target = (directory / relative).resolve()
            if directory not in target.parents or '..' in relative.parts:
                raise ValueError('Unsafe checkpoint path')
            exists = target.is_file()
            digest = sha(target) if exists else None
            entry['checkpoints'].append({'path': str(target), 'exists': exists,
                'sha256': digest, 'matches': exists and digest == metadata['sha256'] and target.stat().st_size == metadata['bytes']})
    return result


try:
    import json as _m08c_json
    print('M08C_READONLY_RESULT', _m08c_json.dumps(_m08c_inspect(dict(globals())), sort_keys=True))
finally:
    globals().pop('_m08c_inspect', None)
    globals().pop('_m08c_json', None)
