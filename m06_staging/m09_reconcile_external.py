"""Reconcile captured scientific evidence locally; never accesses Colab or datasets."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/axxel/m09_external_20261121'
external_path = OUT / 'bootstrap_external/result.json'
legacy_path = ROOT / 'output/axxel/m09_legacy_static_20261121/bootstrap_legacy_static/result.json'
external = json.loads(external_path.read_text())['result']
legacy = json.loads(legacy_path.read_text())['result']
assert not external['errors'] and not legacy['errors']
assert len(legacy['checkpoints']) == 4
metadata = external['metadata']
legacy_by_path = {x['path']: x for x in legacy['checkpoints']}
records = []
for checkpoint in external['checkpoints']:
    path = checkpoint['path']
    reason = None
    evidence = None
    absolute = external['root'] + '/' + path
    if absolute in legacy_by_path:
        item = legacy_by_path[absolute]
        assert len(item['parameter_names']) == 127
        assert not item['lora_keys'] and not item['regression_head_keys']
        reason = 'Original backbone parameters; no LoRA or expected Huber head keys in static checkpoint schema'
        evidence = {'checkpoint_sha256': item['sha256'], 'source': str(legacy_path.relative_to(ROOT))}
    else:
        for item in metadata:
            fields = item['scientific_fields']
            if item['path'].endswith('/resultados/configuration.json'):
                directory = item['path'].removesuffix('configuration.json')
                if path.startswith(directory) and fields.get('loss') == 'horizon_only_two_token_levels':
                    reason = 'Token-level objective differs from continuous Huber objective'
            elif item['path'].endswith('/auditoria/protocolo.json'):
                directory = item['path'].removesuffix('auditoria/protocolo.json')
                targets = fields.get('targets', fields.get('target_modules', []))
                if path.startswith(directory) and any('dep_layer.cross_attn.' in x for x in targets):
                    reason = 'Dependency cross-attention adapted; differs from both proposed arms'
            if reason:
                evidence = {'metadata': item['path'], 'sha256': item['sha256']}
                break
    records.append({'checkpoint': path, 'classification': 'DISTINCT_METHOD' if reason else 'UNRESOLVED',
                    'reason': reason, 'evidence': evidence})
unresolved = [x for x in records if x['classification'] == 'UNRESOLVED']
assert len(records) == 68 and not unresolved
result = {'status': 'PASS_WITH_DOCUMENTED_SCOPE', 'scope': external['root'],
          'exclusions': 'datasets, reserved years/test/holdout, credentials, tokenizer paths; MISION06 reconciled separately',
          'external_directories': len(external['directories']), 'external_metadata_files': len(metadata),
          'external_checkpoints_classified': len(records), 'unresolved': unresolved,
          'records': records, 'interpretation': 'No equivalent persisted experiment found among inspected candidates. Not a claim about unrelated Drive locations or excluded data directories.',
          'phase_V_started': False,
          'evidence_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (external_path, legacy_path)}}
destination = OUT / 'scientific_reconciliation.json'
with destination.open('x', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(json.dumps({'status': result['status'], 'classified': len(records), 'unresolved': len(unresolved), 'output': str(destination)}))
