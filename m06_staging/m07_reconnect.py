"""Reconnect existing Colab through visible UI; never restart or run cells."""
import datetime, json, pathlib
from playwright.sync_api import sync_playwright
out = pathlib.Path(__file__).parent / ('m07_reconnect_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
result = {}
try:
    with sync_playwright() as p:
        b = p.chromium.connect_over_cdp('http://127.0.0.1:9222', timeout=10000)
        pages = [page for ctx in b.contexts for page in ctx.pages if page.url.split('?')[0].split('#')[0] == 'https://colab.research.google.com/drive/1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss']
        assert len(pages) == 1
        page = pages[0]
        button = page.get_by_role('button', name='Volver a conectar', exact=True)
        result['reconnect_buttons'] = button.count()
        if button.count() == 1:
            button.click(timeout=5000)
            page.wait_for_timeout(10000)
        result['buttons'] = page.get_by_role('button').all_text_contents()
        result['dialogs'] = page.get_by_role('dialog').all_text_contents()
        result['runtime_controls'] = page.locator('colab-connect-button').aria_snapshot()
        result['tail'] = page.locator('body').inner_text()[-3500:]
except Exception as exc:
    result['error'] = repr(exc)
with out.open('x', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(out)
print(json.dumps({k:v for k,v in result.items() if k not in ('buttons','tail')}, ensure_ascii=True))
