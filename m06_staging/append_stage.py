from pathlib import Path
import json,ast,sys
root=Path(__file__).resolve().parent
nb=json.loads((root/sys.argv[1]).read_text(encoding='utf-8'))
original=json.loads(json.dumps(nb['cells'][:40]))
for file in sys.argv[2:]:
    source=(root/file).read_text(encoding='utf-8')
    ast.parse(source); compile(source,file,'exec')
    cid=Path(file).stem+'_v1'
    assert all(c.get('metadata',{}).get('id')!=cid for c in nb['cells'])
    nb['cells'].append({'cell_type':'code','metadata':{'id':cid},'id':cid,'source':source.splitlines(keepends=True),'execution_count':None,'outputs':[]})
assert nb['cells'][:40]==original
(root/'notebook_stage.ipynb').write_text(json.dumps(nb,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'cells':len(nb['cells']),'preserved_historical':40,'new_ids':[c['metadata']['id'] for c in nb['cells'][40:]]}))
