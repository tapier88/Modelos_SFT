"""Assemble reviewed experiment sources. Does not connect to Colab or execute training."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
protocol_path=ROOT/'m06_staging/audit_20261120/PROTOCOLO_FASE_V_JOINT_PROPUESTO_20261120.json'
protocol_bytes=protocol_path.read_bytes()
assert hashlib.sha256(protocol_bytes).hexdigest()=='c36d6a9e5524e1f19101d59e8e06221e7b01594d0f1dc9f9e7ad3bd95f58aac2'
protocol=json.loads(protocol_bytes)
data=(ROOT/'m06_staging/m06_data.py').read_text(encoding='utf-8').split("for rel in ['auditoria/ventanas_m06.parquet'")[0]
assert '.to_parquet(' not in data and '.to_csv(' not in data
body=(ROOT/'m06_staging/m10_phase_v.py').read_text(encoding='utf-8')
evaluation=(ROOT/'m06_staging/m10_evaluation.py').read_text(encoding='utf-8')
activity=(ROOT/'m06_staging/m09_activity_gate_v4.py').read_text(encoding='utf-8')
activity=activity.rsplit('\nm09_activity_gate(globals())',1)[0]
activity=activity.replace("print('M09_ACTIVITY_V4_RESULT '+json.dumps(report),flush=True)","print('M09_ACTIVITY_V4_RESULT '+json.dumps(report),flush=True)\n    return report")
for code in (body,evaluation,activity,data):ast.parse(code)
source_hash=hashlib.sha256((body+'\n'+evaluation+'\n'+data).encode()).hexdigest()
bindings={'M10_AUTHORIZED':True,'M10_PROTOCOL':protocol,'M10_PROTOCOL_SHA':hashlib.sha256(protocol_bytes).hexdigest(),
          'M10_DATA_SOURCE':data,'M10_ACTIVITY_SOURCE':activity,'M10_EVALUATION_SOURCE':evaluation,'M10_BODY_SOURCE':body,'M10_SOURCE_SHA':source_hash}
payload='\n'.join(k+'='+repr(v) for k,v in bindings.items())+'\n'+body
ast.parse(payload)
destination=ROOT/'m06_staging/m10_phase_v_payload.py'
destination.write_text(payload,encoding='utf-8')
pin=hashlib.sha256(destination.read_bytes()).hexdigest()
metadata={'payload_sha256':pin,'scientific_source_sha256':source_hash,'protocol_sha256':bindings['M10_PROTOCOL_SHA']}
(ROOT/'m06_staging/m10_source_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(json.dumps(metadata))
