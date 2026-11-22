"""Read existing Colab UI only. No kernel requests, clicks, scripts or navigation."""
import argparse
import datetime
import hashlib
import json
import pathlib
import re
from urllib.parse import urlsplit

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK_ID = '1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss'
URL = 'https://colab.research.google.com/drive/' + NOTEBOOK_ID
EVIDENCE = ROOT / 'm06_staging/m07_reconciliation_20261120T224952Z/final_result.json'


def exact_notebook(url):
    p = urlsplit(url)
    return p.scheme == 'https' and p.netloc == 'colab.research.google.com' and p.path.rstrip('/') == '/drive/' + NOTEBOOK_ID


def item(status, value, source, date):
    return dict(status=status, value=value, evidence=source, date=date)


def classify(snapshot, now):
    result = {key: item('UNKNOWN', 'Not verified', reason, now) for key, reason in {
        'notebook': 'Exact unique tab required',
        'runtime': 'Unambiguous connection control required',
        'gpu': 'GPU hardware identity unavailable without runtime query',
        'gpu_memory': 'VRAM unavailable; system RAM is not VRAM',
        'system_memory': 'Connection control RAM reading unavailable',
        'cuda': 'Runtime query required; saved outputs excluded',
        'pytorch': 'Runtime query required; saved outputs excluded',
        'drive': 'Mount and AXXEL_CAPITAL accessibility require runtime verification',
        'models': 'Loaded objects require runtime verification',
        'active_cells': 'No positive execution indicator observed; absence does not prove idle',
        'training_processes': 'Background processes cannot be inferred from cell controls',
        'checkpoint': 'Remote manifest and checkpoint revalidation not performed',
    }.items()}
    if snapshot.get('exact_tabs') != 1 or not exact_notebook(snapshot.get('url', '')):
        return result
    result['notebook'] = item('CURRENT', snapshot['url'], 'Exact URL and unique CDP page; ' + snapshot.get('title', ''), now)
    text = snapshot.get('connection', '')
    connected = bool(re.search(r'Conectado a|Connected to', text, re.I))
    disconnected = bool(re.search(r'Desconectado|Disconnected|Volver a conectar|Reconnect|button "(?:Conectar|Connect)"', text, re.I))
    if connected != disconnected:
        result['runtime'] = item('CURRENT', 'connected' if connected else 'disconnected', text, now)
    if connected and not disconnected:
        if '(GPU)' in text:
            result['gpu'] = item('CURRENT', 'GPU backend; hardware model UNKNOWN', 'colab-connect-button', now)
        ram = re.search(r'RAM: ([\d.,]+) GB/([\d.,]+) GB', text)
        if ram:
            result['system_memory'] = item('CURRENT', 'used/total: ' + ram.group(1) + '/' + ram.group(2) + ' GB', 'colab-connect-button; rounded UI reading', now)
        if snapshot.get('stop_buttons', 0) > 0:
            result['active_cells'] = item('CURRENT', 'Execution stop control present', 'Visible stop-execution buttons; not proof of training', now)
    return result


def historical():
    if not EVIDENCE.exists():
        return {}
    raw = EVIDENCE.read_bytes()
    data = json.loads(raw)
    report = data['report']
    source = str(EVIDENCE.relative_to(ROOT)) + '; local sha256=' + hashlib.sha256(raw).hexdigest()
    date = report['timestamp']
    return {
        'gpu': item('HISTORICAL', report['gpu'].strip(), source, date),
        'checkpoint': item('HISTORICAL', {'manifest_sha256': data['manifest_sha256'], 'path': data['output'], 'files_verified_then': data['files_verified']}, source, date),
        'experiment': item('HISTORICAL', {'status': report['status'], 'step0': report['step0']['step0'], 'phase_V_started': report['phase_V_started']}, source, date),
    }


def collect():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    snapshot = {}
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp('http://127.0.0.1:9222', timeout=10000)
            pages = [page for context in browser.contexts for page in context.pages if exact_notebook(page.url)]
            snapshot['exact_tabs'] = len(pages)
            if len(pages) == 1:
                page = pages[0]
                snapshot.update(url=page.url, title=page.title())
                control = page.locator('colab-connect-button')
                if control.count() == 1:
                    snapshot['connection'] = control.aria_snapshot(timeout=5000)
                snapshot['stop_buttons'] = page.get_by_role('button', name=re.compile(r'^(Detener ejecución|Interrumpir ejecución|Stop execution|Interrupt execution)$', re.I)).count()
                # Only region labels, never cell source/output or hidden app state.
                regions = page.get_by_role('region').all()
                labels = [r.get_attribute('aria-label') for r in regions]
                labels = [label for label in labels if label and re.match(r'^(Celda|Cell) \d+:', label)]
                snapshot['cell_region_count'] = len(labels)
                snapshot['cell_labels_sha256'] = hashlib.sha256(json.dumps(labels, ensure_ascii=True).encode()).hexdigest()
    except Exception as exc:
        snapshot['error'] = type(exc).__name__ + ': ' + str(exc)
    return dict(observed_at=now, readonly=True, snapshot=snapshot, components=classify(snapshot, now), historical=historical())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = collect()
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=True, indent=2)


if __name__ == '__main__':
    main()
