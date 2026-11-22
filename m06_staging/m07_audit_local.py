"""Revalidate historical artifacts without overwriting historical outputs."""
import datetime, hashlib, json, pathlib

root = pathlib.Path(__file__).resolve().parent
original = root / 'audit_20261120/audit_verify.py'
out = root / ('m07_audit_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
out.mkdir(exist_ok=False)
source = original.read_text(encoding='utf-8').split("(ROOT/'audit_evidence.json').write_text")[0]
scope = {'__file__': str(original), '__name__': '__main__'}
try:
    exec(compile(source, str(original), 'exec'), scope)
    result = scope['comparison']
    result['verification_script_sha256'] = hashlib.sha256(original.read_bytes()).hexdigest()
    result['observed_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / 'evidence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['phase_IV_files_verified', 'phase_IV_manifest_sha256', 'head_benchmarks', 'metrics_recomputed', 'paired_intervals_recomputed']}, indent=2))
except Exception as exc:
    (out / 'failure.json').write_text(json.dumps({'error': repr(exc)}), encoding='utf-8')
    raise
finally:
    print('OUTPUT', out)
