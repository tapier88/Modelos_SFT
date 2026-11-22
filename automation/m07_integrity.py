"""Source transport and notebook preservation checks (no browser side effects)."""
import ast, difflib, hashlib, json

def source(cell):
    value=cell.get('source', '')
    return ''.join(value) if isinstance(value,list) else value

def cell_id(cell):
    primary=cell.get('id')
    secondary=cell.get('metadata',{}).get('id')
    assert not (primary and secondary and primary!=secondary), 'Conflicting cell IDs'
    return primary or secondary

def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def transport(code, filename):
    """Two physical lines avoid Monaco's automatic indentation of pasted blocks.

    The complete original source is one Python string literal, not shell code.
    Both the payload and the exact transport source are verified before execution.
    """
    ast.parse(code)
    marker=code.splitlines()[0]
    assert marker.startswith('# M07_')
    return marker+'\nexec(compile('+repr(code)+', '+repr(filename)+", 'exec'))\n"

def verify_source(actual, expected, directory):
    # Same leading/trailing whitespace policy as original assertion; never normalize indentation.
    info={'actual_sha256':sha(actual),'expected_sha256':sha(expected),
          'exact_equal':actual==expected,'strip_equal':actual.strip()==expected.strip()}
    (directory/'source_comparison.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    if actual.strip()!=expected.strip():
        (directory/'source_mismatch.diff').write_text(''.join(difflib.unified_diff(expected.splitlines(True),actual.splitlines(True),fromfile='expected',tofile='notebook')),encoding='utf-8')
        raise AssertionError('Source mismatch')

def verify_preserved(before, after, added_id=None):
    old_ids=[cell_id(c) for c in before['cells']]
    new_ids=[cell_id(c) for c in after['cells']]
    assert all(old_ids) and all(new_ids)
    assert len(set(old_ids))==len(old_ids) and len(set(new_ids))==len(new_ids), 'Duplicate cell IDs'
    assert [cid for cid in new_ids if cid!=added_id]==old_ids, 'Existing cells removed or reordered'
    original={cell_id(c):c for c in before['cells']}
    for c in after['cells']:
        cid=cell_id(c)
        if cid in original:
            assert source(c)==source(original[cid]) and c['cell_type']==original[cid]['cell_type'], 'Historical cell changed'

def assert_idle(nb, runtime_text):
    assert not any(c.get('metadata',{}).get('executionInfo',{}).get('status') in ['busy','running','pending'] for c in nb['cells']), 'Active cell metadata'
    assert not any(word in runtime_text.casefold() for word in ['esperando a que finalice','en ejecución','executing','busy','desconectado','disconnected']), 'Runtime not idle/connected'
    assert any(word in runtime_text.casefold() for word in ['conectado a','connected to']), 'Runtime connection unverified'
