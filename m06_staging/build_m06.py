from pathlib import Path
import ast
import hashlib
import json

root = Path(__file__).resolve().parent
nb = json.loads((root / 'notebook_original.ipynb').read_text(encoding='utf-8'))
original = json.loads(json.dumps(nb))
assert len(nb['cells']) == 40
assert not any(c.get('metadata', {}).get('id', '').startswith('m06_') for c in nb['cells'])

def add(kind, cell_id, source):
    cell = {'cell_type': kind, 'metadata': {'id': cell_id}, 'id': cell_id,
            'source': source.splitlines(keepends=True)}
    if kind == 'code':
        ast.parse(source)
        compile(source, cell_id, 'exec')
        cell.update(execution_count=None, outputs=[])
    nb['cells'].append(cell)

add('markdown', 'm06_inicio_20261114', '''# MISIÓN 06 · Diagnóstico causal y alineación predictiva

**Estado: preparación inicial; experimentos M06 no ejecutados.**

Train 2010–2019 · Desarrollo 2020 · L=56 · H=4 · RESEARCH ONLY · ORDERS DISABLED.

Ejecutar solo esta sección. **No ejecutar las recuperaciones de M05 ni «Ejecutar todas»**:
las celdas históricas cargan también 2021–2022. Sus resultados y checkpoints se conservan.

Primero ejecutar M06 · 0 y revisar el hardware y la procedencia. Después M06 · 1
construye un índice independiente de los archivos permitidos. Las fases I–VI aún
requieren implementación, ejecución y verificación en este mismo cuaderno.
''')
for name, cell_id in [('m06_preflight.py', 'm06_preflight_20261114'), ('m06_data.py', 'm06_datos_20261114')]:
    add('code', cell_id, (root / name).read_text(encoding='utf-8'))
add('markdown', 'm06_revision_20261114', (root / 'ESTADO_M06.md').read_text(encoding='utf-8'))

assert nb['cells'][:40] == original['cells']
assert nb['metadata'] == original['metadata']
assert all(c['execution_count'] is None and c['outputs'] == [] for c in nb['cells'][40:] if c['cell_type'] == 'code')
out = root / 'notebook_actualizado.ipynb'
out.write_text(json.dumps(nb, ensure_ascii=False), encoding='utf-8')
checks = {'historical_cells_preserved': 40, 'new_cells': 4,
          'code_cells_compiled': 2, 'colab_execution': False,
          'historical_cell_objects_equal': True, 'metadata_unchanged': True,
          'new_code_outputs_empty': True,
          'notebook_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}
(root / 'verificacion_preparacion.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
print(json.dumps(checks, indent=2))
