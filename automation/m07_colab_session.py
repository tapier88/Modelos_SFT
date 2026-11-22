"""Bounded UI operations on the single authorized notebook; never run historical cells."""
import argparse, datetime, json, pathlib, re
from playwright.sync_api import sync_playwright
from controlar_kronos_colab import menu_item
from m07_integrity import transport, verify_source, verify_preserved, assert_idle, cell_id, source, sha

parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['inspect', 'append', 'snapshot', 'runverified', 'connectdrive'])
parser.add_argument('--source', type=pathlib.Path)
args = parser.parse_args()
out = pathlib.Path(__file__).resolve().parents[1] / 'output/playwright' / datetime.datetime.now(datetime.timezone.utc).strftime('m07_%Y%m%dT%H%M%S%fZ')
out.mkdir(parents=True, exist_ok=False)
def download(page, name):
    page.keyboard.press('Escape')
    menu_item(page, [r'^Archivo$',r'^File$']).click()
    menu_item(page, [r'Descargar',r'Download']).hover()
    with page.expect_download(timeout=30000) as pending:
        menu_item(page,[r'^(Descargar\s+|Download\s+)?\.ipynb$']).click()
    pending.value.save_as(str(out / name))
    return json.loads((out / name).read_text(encoding='utf-8'))
with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp('http://127.0.0.1:9222',timeout=10000)
    pages=[x for c in browser.contexts for x in c.pages if x.url.split('?')[0].split('#')[0]=='https://colab.research.google.com/drive/1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss']
    assert len(pages)==1, 'Ambiguous notebook'
    page=pages[0]
    print('TITLE',page.title())
    print('RUNTIME',page.locator('colab-connect-button').aria_snapshot())
    (out/'identity.json').write_text(json.dumps({'url':page.url,'title':page.title(),'runtime':page.locator('colab-connect-button').aria_snapshot()},ensure_ascii=False,indent=2),encoding='utf-8')
    if args.action=='connectdrive':
        page.get_by_role('button',name='Conectar con Google Drive',exact=True).click(timeout=5000)
        page.wait_for_timeout(3000)
        print('AUTH_WINDOWS',json.dumps([{'host':'accounts.google.com','title':x.title()} for c in browser.contexts for x in c.pages if 'accounts.google.com' in x.url],ensure_ascii=True))
    if args.action in ['inspect','append','runverified']:
        nb=download(page,'before.ipynb')
        print('CELLS',len(nb['cells']))
        for c in nb['cells']:
            if 'M07_' in ''.join(c.get('source',[]))[:100]:
                print('M07_CELL',c.get('id',c.get('metadata',{}).get('id')),json.dumps(c.get('outputs',[]),ensure_ascii=True)[:1800])
    if args.action=='append':
        assert args.source
        code=args.source.read_text(encoding='utf-8')
        marker=code.splitlines()[0]
        assert marker.startswith('# M07_')
        assert not any(marker in ''.join(c.get('source',[])) for c in nb['cells']), 'Existing cell: inspect, do not duplicate'
        assert_idle(nb,page.locator('colab-connect-button').aria_snapshot())
        transmitted=transport(code,args.source.name)
        (out/'original_source.py').write_text(code,encoding='utf-8')
        (out/'transport_source.py').write_text(transmitted,encoding='utf-8')
        page.keyboard.press('Escape')
        page.get_by_role('button',name='Código Insertar celda de código debajo Ctrl+M B',exact=True).click()
        page.keyboard.insert_text(transmitted)
        added=download(page,'inserted_before_execution.ipynb')
        matches=[(i,c) for i,c in enumerate(added['cells']) if marker in ''.join(c.get('source',[]))]
        assert len(matches)==1, 'Source mismatch: marker not unique'
        verify_source(source(matches[0][1]),transmitted,out)
        verify_preserved(nb,added,cell_id(matches[0][1]))
        assert_idle(added,page.locator('colab-connect-button').aria_snapshot())
        index=matches[0][0]
        region=page.get_by_role('region',name=re.compile(r'^Celda '+str(index)+r': Celda de código;'))
        assert region.count()==1, 'Ambiguous new cell'
        region.get_by_role('button',name='Ejecutar celda',exact=True).click()
        page.wait_for_timeout(6000)
    if args.action=='runverified':
        code=args.source.read_text(encoding='utf-8')
        marker=code.splitlines()[0]
        assert marker.startswith('# M07_')
        matches=[(i,c) for i,c in enumerate(nb['cells']) if marker in ''.join(c.get('source',[]))]
        assert len(matches)==1, 'Source mismatch: marker not unique'
        verify_source(source(matches[0][1]),code,out)
        assert_idle(nb,page.locator('colab-connect-button').aria_snapshot())
        index, cell=matches[0]
        assert not cell.get('outputs') and cell.get('execution_count') is None, 'Already executed; do not resend'
        region=page.get_by_role('region',name=re.compile(r'^Celda '+str(index)+r': Celda de código;'))
        assert region.count()==1
        region.get_by_role('button',name='Ejecutar celda',exact=True).click()
        page.wait_for_timeout(6000)
    snap=page.locator('body').aria_snapshot()
    (out/'snapshot.txt').write_text(snap,encoding='utf-8')
    print('DIALOGS',json.dumps(page.get_by_role('dialog').all_text_contents(),ensure_ascii=True))
    print('TAIL',snap[-900:].encode('ascii','backslashreplace').decode())
print('EVIDENCE',out)
