"""Reconcile exact historical insertion and fresh notebook without mutating either."""
import ast, datetime, difflib, hashlib, json, pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'automation'))
from m07_integrity import source, cell_id, verify_preserved, sha
out=ROOT/'m06_staging'/datetime.datetime.now(datetime.timezone.utc).strftime('m07_reconciliation_%Y%m%dT%H%M%SZ')
out.mkdir(exist_ok=False)
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
before=load(ROOT/'output/playwright/m07_20261120T224039276321Z/before.ipynb')
inserted=load(ROOT/'output/playwright/m07_20261120T224039276321Z/inserted_before_execution.ipynb')
current_path=ROOT/'output/playwright/m07_20261120T224746735062Z/before.ipynb'
current=load(current_path)
local_path=ROOT/'m06_staging/m07_mount_drive.py'
local=local_path.read_text(encoding='utf-8')
ast.parse(local)
mount=next(c for c in current['cells'] if cell_id(c)=='SOeTIq7YHoEB')
runtime=current['cells'][30]
assert cell_id(runtime)=='9wcbmH4SFnx3' and source(runtime).startswith('# M07_RUNTIME_READONLY_PREFLIGHT')
assert sum(source(c).startswith('# M07_MOUNT_DRIVE_V1') for c in current['cells'])==1
verify_preserved(before,inserted,'SOeTIq7YHoEB')
verify_preserved(inserted,current)
assert source(mount)==source(next(c for c in inserted['cells'] if cell_id(c)=='SOeTIq7YHoEB'))
assert source(mount)==(ROOT/'m06_staging/m07_mount_drive_reviewed.py').read_text(encoding='utf-8')
assert source(runtime)==source(next(c for c in before['cells'] if cell_id(c)=='9wcbmH4SFnx3'))
diff=''.join(difflib.unified_diff(local.splitlines(True),source(mount).splitlines(True),fromfile='m07_mount_drive.py',tofile='notebook/SOeTIq7YHoEB'))
(out/'mount_source.diff').write_text(diff,encoding='utf-8')
mount_output=''.join(''.join(o.get('text',[])) for o in mount.get('outputs',[]))
result=json.loads(next(line.removeprefix('M07_MOUNT_RESULT ') for line in mount_output.splitlines() if line.startswith('M07_MOUNT_RESULT ')))
assert result['drive_available'] and result['canonical_run_available']
evidence={'fresh_notebook':str(current_path.relative_to(ROOT)), 'fresh_notebook_sha256':hashlib.sha256(current_path.read_bytes()).hexdigest(),
 'failed_comparison':"len(matches)==1 and source(matches[0][1]).strip()==code.strip()",'match_count':1,'comparison_equal':False,
 'local_source_sha256_utf8':sha(local),'local_file_sha256_bytes':hashlib.sha256(local_path.read_bytes()).hexdigest(),
 'mount_cell_source_sha256_utf8':sha(source(mount)), 'runtime_cell_source_sha256_utf8':sha(source(runtime)),
 'runtime_index':30,'runtime_id':cell_id(runtime),'mount_index':31,'mount_id':cell_id(mount),
 'mount_execution_count':mount.get('execution_count'),'mount_result':result,
 'historical_sources_unchanged':True,'mount_fully_inserted':True,'mount_unique':True,'runtime_cell_unchanged':True,
 'cause':'Four leading spaces added to final print during multiline editor insertion; print moved into if block.',
 'semantic_difference':'If Drive was already mounted, remote variant would not print M07_MOUNT_RESULT. Not semantically identical in general.',
 'decision':'Preserve executed remote cell and both local source variants. Do not rerun, delete, overwrite or claim original source equality.',
 'RESEARCH_ONLY':True,'test_read':False,'orders_enabled':False,'phase_V_started':False}
(out/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'ESTADO_ANTERIOR.md').write_text((ROOT/'ESTADO_ACTUAL_KRONOS.md').read_text(encoding='utf-8'),encoding='utf-8')
print(out)
print(json.dumps(evidence,ensure_ascii=True,indent=2))
