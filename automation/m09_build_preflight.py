"""Build a new validation payload without modifying historical scientific sources."""
import ast
import json
import pathlib

ROOT=pathlib.Path(__file__).resolve().parents[1]
data=(ROOT/'m06_staging/m06_data.py').read_text(encoding='utf-8').split("for rel in ['auditoria/ventanas_m06.parquet'")[0]
ast.parse(data)
assert '.to_parquet(' not in data and '.to_csv(' not in data
protocol=json.loads((ROOT/'m06_staging/audit_20261120/PROTOCOLO_FASE_V_JOINT_PROPUESTO_20261120.json').read_text(encoding='utf-8'))
body=(ROOT/'m06_staging/m09_preflight_body.py').read_text(encoding='utf-8')
source='# M09_SCIENTIFIC_PREFLIGHT_V1\nDATA_SOURCE='+repr(data)+'\nPROTOCOL='+repr(protocol)+'\n'+body
ast.parse(source)
(ROOT/'m06_staging/m09_preflight_payload.py').write_text(source,encoding='utf-8')
print('Built M09 payload; not executed')
