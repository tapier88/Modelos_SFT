"""Back up notebook via UI, append one read-only runtime diagnostic cell."""
import datetime, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'automation'))
from controlar_kronos_colab import menu_item
from playwright.sync_api import sync_playwright
out = pathlib.Path(__file__).parent / ('m07_runtime_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
out.mkdir(exist_ok=False)
code = '''# M07_RUNTIME_READONLY_PREFLIGHT
import os, json, subprocess
from pathlib import Path
_m07_names = sorted(k for k in globals() if k.startswith(('m6_', 'M6_')))
_m07_root = Path('/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION/20261114T222455Z_32d5e571')
_m07_info = dict(RESEARCH_ONLY=True, orders_enabled=False, test_read=False, pid=os.getpid(), loaded_names=_m07_names, drive_available=Path('/content/drive/MyDrive').is_dir(), canonical_run_available=_m07_root.is_dir())
_m07_info['gpu'] = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader'],capture_output=True,text=True).stdout
_m07_info['processes'] = subprocess.run(['ps','-eo','pid,comm'],capture_output=True,text=True).stdout
print('M07_RUNTIME_EVIDENCE_BEGIN')
print(json.dumps(_m07_info, sort_keys=True))
print('M07_RUNTIME_EVIDENCE_END')
'''
with sync_playwright() as p:
    b = p.chromium.connect_over_cdp('http://127.0.0.1:9222', timeout=10000)
    pages = [page for ctx in b.contexts for page in ctx.pages if page.url.split('?')[0].split('#')[0] == 'https://colab.research.google.com/drive/1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss']
    assert len(pages) == 1
    page = pages[0]
    page.keyboard.press('Escape')
    menu_item(page,[r'^Archivo$',r'^File$']).click()
    (out / 'menu_snapshot.txt').write_text(page.locator('body').aria_snapshot(), encoding='utf-8')
    menu_item(page,[r'Descargar',r'Download']).hover()
    with page.expect_download(timeout=30000) as pending:
        menu_item(page,[r'^(Descargar\s+|Download\s+)?\.ipynb$']).click()
    pending.value.save_as(str(out / 'notebook_before.ipynb'))
    nb = json.loads((out / 'notebook_before.ipynb').read_text(encoding='utf-8'))
    assert not any(c.get('metadata',{}).get('executionInfo',{}).get('status') in ('busy','running','pending') for c in nb['cells']), 'Active cell detected'
    assert not any('M07_RUNTIME_READONLY_PREFLIGHT' in ''.join(c.get('source',[])) for c in nb['cells']), 'Diagnostic already exists'
    page.keyboard.press('Escape')
    page.get_by_role('button', name='Código Insertar celda de código debajo Ctrl+M B', exact=True).click()
    page.keyboard.insert_text(code)
    diagnostic = page.get_by_role('region').filter(has_text='M07_RUNTIME_READONLY_PREFLIGHT').filter(has=page.get_by_role('button',name='Ejecutar celda',exact=True))
    assert 'M07_RUNTIME_READONLY_PREFLIGHT' in page.locator('body').inner_text(), 'Insertion not confirmed; do not execute'
    page.keyboard.press('Control+Enter')
    page.wait_for_timeout(10000)
    text = page.locator('body').inner_text()
    (out / 'visible_after.txt').write_text(text,encoding='utf-8')
    print('\n'.join(line for line in text.splitlines() if any(s in line for s in ['M07_RUNTIME','drive_available','Tesla','loaded_names'])).encode('ascii','backslashreplace').decode())
print('OUTPUT',out)
