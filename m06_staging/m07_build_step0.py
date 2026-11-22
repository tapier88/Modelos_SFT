import ast
from pathlib import Path
root=Path(__file__).resolve().parent
data=(root/'m06_data.py').read_text(encoding='utf-8').split("for rel in ['auditoria/ventanas_m06.parquet'")[0]
ast.parse(data)
assert '.to_parquet(' not in data and '.to_csv(' not in data
tail=(root/'m07_data_step0_tail.py').read_text(encoding='utf-8')
code='# M07_DATA_STEP0_PREFLIGHT_V1\nassert M7_CHECKS["status"] == "recovered_pending_data_and_step0"\nM7_DATA_SOURCE='+repr(data)+'\n'+tail
ast.parse(code)
(root/'m07_data_step0_preflight.py').write_text(code,encoding='utf-8')
print('Prepared step0 checks; no original data writes or optimizer steps')
