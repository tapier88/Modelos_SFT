from pathlib import Path
import ast,hashlib,json
r=Path(__file__).resolve().parent
source=(r/'m06_heads.py').read_text(encoding='utf-8')
# La reparación histórica ya se realizó; no escribir de nuevo su informe.
begin=source.index("if 'M6_RECOVERY' in globals():")
end=source.index('M6_HEAD_SPEC=',begin)
source=source[:begin]+source[end:]
class AtomicPhase(ast.NodeTransformer):
    def visit_Name(self,node):
        names={'M6_DIR':'M6_HEAD_DIR','m6_save_json':'m6_phase_json','m6_figsave':'m6_phase_figsave'}
        if node.id in names: node.id=names[node.id]
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute):
            if node.func.attr in ['to_csv','to_parquet']:
                method='m6_phase_csv' if node.func.attr=='to_csv' else 'm6_phase_parquet'
                node.args=[node.func.value]+node.args;node.func=ast.Name(id=method,ctx=ast.Load())
            elif isinstance(node.func.value,ast.Name) and (node.func.value.id,node.func.attr) in [('np','savez_compressed'),('torch','save')]:
                node.func=ast.Name(id='m6_phase_npz' if node.func.attr=='savez_compressed' else 'm6_phase_torch',ctx=ast.Load())
        return node
tree=ast.fix_missing_locations(AtomicPhase().visit(ast.parse(source)))
body=ast.unparse(tree)
header=(r/'head_phase_io.py').read_text(encoding='utf-8')
tail='''
m6_phase_verify()
assert m6_stable.inventory()==json.loads((m6_stable.OPS/'artifact_manifest_before_recovery.json').read_text())
m6_phase_json('phase_status.json',{'complete':True,'phase':'III','models':M6_HEAD_TABLE.model.tolist(),
 'base_hash':M6_BASE_HASH,'tokenizer_hash':M6_TOK_HASH,'train_windows':8428,'dev_windows':950,
 'seeds':M6_PLAN['head_seeds'],'frozen_transformer':True,'frozen_adapters':True})
M6_PHASE_III_DONE=True
'''
wrapped=header+"\nM6_HEAD_EXEC_SOURCE="+repr(body)+"\n"
wrapped+='''
if (M6_HEAD_DIR/'phase_status.json').exists() and json.loads((M6_HEAD_DIR/'phase_status.json').read_text()).get('complete'):
    M6_HEAD_TABLE=m6_phase_restore();M6_PHASE_III_DONE=True
    display(M6_HEAD_TABLE)
    print('FASE_III_RECUPERADA_COMPLETA; sin entrenamiento ni extracción repetidos')
else:
    exec(compile(M6_HEAD_EXEC_SOURCE,'M06_phase_III_atomic.py','exec'))
'''+''.join('    '+line+'\n' for line in tail.splitlines())
ast.parse(wrapped);compile(wrapped,'M06_heads_atomic','exec')
dest=r/'m06_heads_atomic_v1.py';dest.write_text(wrapped,encoding='utf-8')
print(json.dumps({'path':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'bytes':dest.stat().st_size}))
