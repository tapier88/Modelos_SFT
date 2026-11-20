"""Bounded read-only connectivity probe. Never starts or closes Chrome."""
import datetime, hashlib, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent / 'm07_access_20261120'
ROOT.mkdir(exist_ok=True)
child = r'''
import json
from playwright.sync_api import sync_playwright
print('PLAYWRIGHT_START', flush=True)
try:
    with sync_playwright() as p:
        print('CONNECT_OVER_CDP_ENTER', flush=True)
        browser = p.chromium.connect_over_cdp('http://127.0.0.1:9222', timeout=10000)
        print(json.dumps({'connected': True, 'pages': [page.url for context in browser.contexts for page in context.pages]}), flush=True)
except Exception as e:
    print(json.dumps({'connected':False,'error_type':type(e).__name__,'error':str(e)}),flush=True)
'''
record = {'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'endpoint':'http://127.0.0.1:9222','operation':'chromium.connect_over_cdp',
          'test_read':False,'orders_enabled':False,'RESEARCH_ONLY':True}
try:
    result = subprocess.run([sys.executable,'-u','-c',child], capture_output=True, text=True, timeout=20)
    record.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
except subprocess.TimeoutExpired as e:
    record.update(timeout_seconds=20,stdout=(e.stdout or b'').decode(errors='replace'),stderr=(e.stderr or b'').decode(errors='replace'))
path = ROOT / 'playwright_probe.json'
with path.open('x',encoding='utf-8') as f:
    json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(record,ensure_ascii=True))
print('SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
