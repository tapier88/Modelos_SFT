"""Local, read-only audit of explicit Phase V evidence; never imports training code.

Writes a new evidence directory with exclusive creation. No dataset, checkpoint,
network, model inference, or reserved-test access.
"""
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result_path = ROOT / 'output/axxel/m10_phase_v_20261122/bootstrap_phase_v/result.json'
    reconciled = ROOT / 'output/axxel/m10_reconciled_20261122T120350'
    staging = ROOT / 'm06_staging'
    protocol_path = staging / 'audit_20261120/PROTOCOLO_FASE_V_JOINT_PROPUESTO_20261120.json'
    inputs = {}

    def read(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text(encoding='utf-8-sig'))

    envelope = read(result_path)
    assert sha(result_path) == 'e78f0bbbf46ab0693ced209b1b2642bd6b23960e922b0774463a83a000b89321'
    for name, expected in read(reconciled / 'checksums.json').items():
        path = reconciled / name
        assert path.parent == reconciled and sha(path) == expected, name
        inputs[str(path.relative_to(ROOT))] = sha(path)
    protocol = read(protocol_path)
    pins = read(staging / 'm10_source_manifest.json')
    assert sha(protocol_path) == pins['protocol_sha256']
    payload_path = staging / 'm10_phase_v_payload.py'
    assert sha(payload_path) == pins['payload_sha256'] == envelope['source_sha256']
    inputs[str(payload_path.relative_to(ROOT))] = sha(payload_path)
    # Literal parsing only: never execute the payload.
    bindings = {}
    for node in ast.parse(payload_path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith('M10_') and name != 'M10_RESULT':
                bindings[name] = ast.literal_eval(node.value)
    assert bindings['M10_PROTOCOL'] == protocol
    for name, filename in [('M10_BODY_SOURCE', 'm10_phase_v.py'), ('M10_EVALUATION_SOURCE', 'm10_evaluation.py')]:
        path = staging / filename
        assert bindings[name] == path.read_text(encoding='utf-8')
        inputs[str(path.relative_to(ROOT))] = sha(path)
    source = '\n'.join(bindings[k] for k in ['M10_BODY_SOURCE', 'M10_EVALUATION_SOURCE', 'M10_DATA_SOURCE'])
    assert hashlib.sha256(source.encode()).hexdigest() == pins['scientific_source_sha256']
    notebook = read(reconciled / 'current.ipynb')
    cells = [c for c in notebook['cells'] if c.get('metadata', {}).get('id') == envelope['health_cell_id']]
    assert len(cells) == 1
    output = ''.join(''.join(o.get('text', [])) for o in cells[0].get('outputs', []))
    saved_results = [json.loads(line.split('M10_PHASE_V_RESULT ', 1)[1]) for line in output.splitlines() if line.startswith('M10_PHASE_V_RESULT ')]
    result = envelope['result']
    assert saved_results == [result]
    assert result['status'] == 'COMPLETED' and result['test_read'] is False and result['orders_enabled'] is False
    assert len(result['runs']) == 6
    assert {(r['arm'], r['seed']) for r in result['runs']} == {(arm, seed) for arm in protocol['arms'] for seed in protocol['seeds']}
    assert all(r['step'] == 300 and r['status'] == 'COMPLETED' for r in result['runs'])
    evaluation = result['evaluation']
    metrics = evaluation['metrics']
    rows = []
    for seed in protocol['seeds']:
        initial, control, joint = [metrics[f'{prefix}{seed}']['MAE'] for prefix in ['Head_Huber_s', 'control_head_only_s', 'joint_QV_head_s']]
        rows.append(dict(seed=seed, initial_MAE=initial, control_MAE=control, joint_MAE=joint,
                         control_minus_initial=control-initial, joint_minus_control=joint-control,
                         joint_minus_persistence=joint-metrics['Persistencia']['MAE']))
    for arm, stored in evaluation['seed_dispersion'].items():
        values = [metrics[f'{arm}_s{s}']['MAE'] for s in protocol['seeds']]
        assert np.isclose(np.mean(values), stored['mean'], rtol=0, atol=1e-12)
        assert np.isclose(np.std(values, ddof=1), stored['std_ddof1'], rtol=0, atol=1e-12)
    pairs = evaluation['paired_comparisons']
    assert len(pairs) == 9
    for pair in pairs:
        delta = metrics[pair['candidate']]['MAE'] - metrics[pair['reference']]['MAE']
        assert abs(delta-pair['delta_MAE']) < 1e-11
        assert pair['CI_simultaneous_Bonferroni9'][0] <= 0 <= pair['CI_simultaneous_Bonferroni9'][1]
    assert evaluation['followup_gate'] is False
    scaler_path = staging / 'audit_20261120/fase_III/checkpoints/fase_III/7587400db64558e6/train_only_scalers.npz'
    assert sha(scaler_path) == protocol['shared_model']['scalers_sha256']
    inputs[str(scaler_path.relative_to(ROOT))] = sha(scaler_path)
    with np.load(scaler_path, allow_pickle=False) as z:
        return_scale = z['return_scale'].tolist()
    progress = [json.loads(line.split('M10_PROGRESS ', 1)[1]) for line in output.splitlines() if line.startswith('M10_PROGRESS ')]
    checkpoints = [p for p in progress if p.get('stage') == 'checkpoint_committed']
    initial_mean = np.mean([r['initial_MAE'] for r in rows])
    control_mean = evaluation['seed_dispersion']['control_head_only']['mean']
    joint_mean = evaluation['seed_dispersion']['joint_QV_head']['mean']
    now = dt.datetime.now(dt.timezone.utc)
    dest = ROOT / 'output/axxel' / ('m11_audit_' + now.strftime('%Y%m%dT%H%M%S%fZ'))
    dest.mkdir(exist_ok=False)
    audit = dict(observed_at=now.isoformat(), research_only=True, orders_enabled=False, test_read=False,
                 training_executed=False, inference_executed=False,
                 scope='Local aggregate/source/hash audit; notebook outputs HISTORICAL; no raw Phase V prediction or remote checkpoint revalidation',
                 checks_passed=True, rows=rows, return_scale=return_scale,
                 mean_control_minus_initial=float(control_mean-initial_mean),
                 mean_joint_minus_control=float(joint_mean-control_mean),
                 control_relative_excess_percent=float(100*(control_mean/metrics['Persistencia']['MAE']-1)),
                 joint_relative_excess_percent=float(100*(joint_mean/metrics['Persistencia']['MAE']-1)),
                 paired_comparisons=pairs, historical_checkpoint_progress=checkpoints, inputs_sha256=inputs)
    for name, value in [('audit.json', audit), ('input_hashes.json', inputs)]:
        with (dest / name).open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    (dest / 'ESTADO_ANTERIOR.md').write_bytes((ROOT / 'ESTADO_ACTUAL_KRONOS.md').read_bytes())
    manifest = {p.name: sha(p) for p in dest.iterdir() if p.is_file()}
    with (dest / 'checksums.json').open('x') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(dict(directory=str(dest), rows=rows, return_scale=return_scale,
                          checkpoint_progress_count=len(checkpoints), checks_passed=True), indent=2))


if __name__ == '__main__':
    main()
