"""Read-only CDP and exact-notebook inspection; exclusive timestamped evidence."""
import datetime, json, pathlib, urllib.request

out = pathlib.Path(__file__).parent / ('m07_probe_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
record = {'RESEARCH_ONLY': True, 'orders_enabled': False, 'test_read': False}
try:
    with urllib.request.urlopen('http://127.0.0.1:9222/json/version', timeout=5) as r:
        record['cdp_version'] = json.load(r).get('Browser')
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp('http://127.0.0.1:9222', timeout=10000)
        pages = [page for ctx in browser.contexts for page in ctx.pages if page.url.split('?')[0].split('#')[0].rstrip('/') == 'https://colab.research.google.com/drive/1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss']
        record['exact_notebook_tabs'] = len(pages)
        if len(pages) == 1:
            record['url'] = pages[0].url
            record['title'] = pages[0].title()
            record['visible_text'] = pages[0].locator('body').inner_text(timeout=15000)
except Exception as exc:
    record['error'] = type(exc).__name__ + ': ' + str(exc)
with out.open('x', encoding='utf-8') as f:
    json.dump(record, f, ensure_ascii=False, indent=2)
print(out)
print(json.dumps(record, ensure_ascii=True))
