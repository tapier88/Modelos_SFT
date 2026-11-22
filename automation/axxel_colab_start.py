"""Daily Colab transport: one pinned health cell, exact source and fresh results.
No historical cell execution, model loading, runtime restart or training.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import re
import time
import uuid

from controlar_kronos_colab import menu_item, output_text
from m07_integrity import assert_idle, cell_id, source, verify_preserved, verify_source
from m08_readonly_status import exact_notebook

ROOT = pathlib.Path(__file__).resolve().parents[1]
URL = 'https://colab.research.google.com/drive/1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss'
MARKER = '# AXXEL_DAILY_HEALTH_V1'
DIAGNOSTIC = ROOT / 'm06_staging/m08c_readonly_diagnostic.py'
PIN = '890b369671fa0a424f2389b3d12a9c33c5ef22f6ec389cd26501234cb1a34620'
RESULT_PREFIX = 'M08C_READONLY_RESULT '
JOB = 'health'
EXECUTION_TIMEOUT_SECONDS = None
POLL_SECONDS = 3
TRAINING_AUTHORIZED = False


def select_job(job):
    global MARKER, DIAGNOSTIC, PIN, RESULT_PREFIX, JOB
    JOB = job
    if job == 'restore':
        MARKER = '# AXXEL_RUNTIME_RESTORE_V1'
        DIAGNOSTIC = ROOT / 'automation/axxel_restore_runtime.py'
        PIN = 'eee7d175c899f956bcc68a3496fecd0558c47d3316ebd21c0e6f4daed4c3c223'
        RESULT_PREFIX = 'AXXEL_RESTORE_RESULT '
    elif job == 'preflight':
        MARKER = '# M09_SCIENTIFIC_PREFLIGHT_V1'
        DIAGNOSTIC = ROOT / 'm06_staging/m09_preflight_payload.py'
        PIN = '609f16ce0ad204ffabafdfd44d14349e4c057974c9ae17d523d8748720e1fae1'
        RESULT_PREFIX = 'M09_PREFLIGHT_RESULT '


def payload():
    raw = DIAGNOSTIC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        raise ValueError('Diagnostic source hash mismatch')
    code = raw.decode('utf-8')
    return MARKER + '\nexec(compile(' + repr(code) + ", 'axxel_daily_health.py', 'exec'))\n"


def _download(page, directory, name):
    assert exact_notebook(page.url), 'Notebook identity changed'
    page.keyboard.press('Escape')
    menu_item(page, [r'^Archivo$', r'^File$']).click()
    menu_item(page, [r'Descargar', r'Download']).hover()
    page.wait_for_timeout(600)
    with page.expect_download(timeout=30000) as pending:
        menu_item(page, [r'^(Descargar\s+|Download\s+)?\.ipynb$']).click()
    target = directory / name
    pending.value.save_as(str(target))
    return json.loads(target.read_text(encoding='utf-8'))


def download(page, directory, name):
    # UI/download retries are safe; execution submission is never retried.
    for attempt in range(3):
        try:
            return _download(page, directory, name)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1)


def pick(browser, open_missing=False):
    pages = [p for c in browser.contexts for p in c.pages if exact_notebook(p.url)]
    if not pages and open_missing and browser.contexts:
        page = browser.contexts[0].new_page()
        page.goto(URL, wait_until='domcontentloaded', timeout=60000)
        pages = [page] if exact_notebook(page.url) else []
    if len(pages) != 1:
        raise RuntimeError('Exactly one master notebook required; no tabs created')
    return pages[0]


def connected(page):
    return page.locator('colab-connect-button').aria_snapshot(timeout=5000)


def ensure_connected(page):
    clicked = False
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if re.search('Conectado a|Connected to', connected(page), re.I):
            return clicked
        button = page.locator('colab-connect-button').get_by_role('button', name=re.compile(r'^(Volver a conectar|Conectar|Reconnect|Connect)(\b|\s)', re.I))
        if button.count() > 1:
            raise RuntimeError('Connection button ambiguous; no reconnect submitted')
        if not clicked and button.count() == 1:
            button.click(timeout=5000)
            clicked = True
        time.sleep(2)
    raise RuntimeError('Connection not established; inspect Colab account/quota/dialog')


def check_idle(page, nb):
    assert_idle(nb, connected(page))
    # This gates cell scheduling, not a false assertion about background jobs.
    stops = page.get_by_role('button', name=re.compile(r'^(Detener ejecución|Interrumpir ejecución|Stop execution|Interrupt execution)$', re.I))
    if stops.count():
        raise RuntimeError('Execution active: no cell scheduled')


def find_health(nb):
    matches = [(i, c) for i, c in enumerate(nb['cells']) if source(c).splitlines()[:1] == [MARKER]]
    if len(matches) > 1:
        raise RuntimeError('Duplicate health cells; not executing')
    return matches


def fresh_result(cell, started, prior_output_id=None):
    for line in output_text(cell).splitlines():
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line.split(' ', 1)[1])
            stamp = datetime.datetime.fromisoformat(result['timestamp_utc'])
            output_id = cell.get('metadata', {}).get('outputId')
            # Colab/local clocks can differ. Output identity is the execution witness.
            fresh = (bool(output_id) and output_id != prior_output_id) if prior_output_id is not None else stamp >= started
            if fresh:
                return result
    return None


def run(directory):
    from playwright.sync_api import sync_playwright
    if JOB == 'phase_v' and not TRAINING_AUTHORIZED:
        raise RuntimeError('Phase V requires explicit authorization; daily start cannot train')
    expected = payload()
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp('http://127.0.0.1:9222', timeout=60000)
        page = pick(browser, open_missing=True)
        if JOB == 'phase_v':
            window_session = page.context.new_cdp_session(page)
            window = window_session.send('Browser.getWindowForTarget')
            window_session.send('Browser.setWindowBounds', {'windowId': window['windowId'], 'bounds': {'windowState': 'normal'}})
            page.bring_to_front()
            window_session.detach()
        reconnection = ensure_connected(page)
        before = download(page, directory, 'before.ipynb')
        pending = [p for p in (ROOT / 'output/axxel').glob('*/bootstrap*/transaction.json')
                   if not (p.parent / 'result.json').exists() and json.loads(p.read_text()).get('job', 'health') == JOB]
        if pending:
            if len(pending) != 1:
                raise RuntimeError('Multiple uncertain transactions; no execution submitted')
            old = json.loads(pending[0].read_text())
            if old['execution_submitted']:
                found = [c for c in before['cells'] if cell_id(c) == old['cell_id']]
                assert len(found) == 1, 'Pending cell missing'
                verify_source(source(found[0]), expected, directory)
                prior_id = old.get('prior_output_id')
                if prior_id is None:
                    original = json.loads((pending[0].parent / 'before.ipynb').read_text(encoding='utf-8'))
                    originals = [c for c in original['cells'] if cell_id(c) == old['cell_id']]
                    prior_id = originals[0].get('metadata', {}).get('outputId', '') if originals else ''
                recovered = fresh_result(found[0], datetime.datetime.fromisoformat(old['started']), prior_id)
                if recovered is None:
                    raise RuntimeError('Previous execution uncertain; no resubmission. Inspect its cell output')
                report = {'status': 'PROBE_RECOVERED', 'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          'notebook': page.url, 'health_cell_id': old['cell_id'], 'inserted': False,
                          'recovered_transaction': str(pending[0]), 'source_sha256': PIN,
                          'result': recovered, 'training_enabled': JOB == 'phase_v' and TRAINING_AUTHORIZED}
                for dest in (pending[0].parent / 'result.json', directory / 'result.json'):
                    dest.write_text(json.dumps(report, indent=2))
                return report
        check_idle(page, before)
        matches = find_health(before)
        inserted = False
        if not matches:
            page.keyboard.press('Escape')
            button = page.get_by_role('button', name=re.compile(r'^(Código|Code) Insert', re.I))
            if button.count() != 1:
                raise RuntimeError('Unique insert-code control unavailable')
            button.click()
            page.keyboard.insert_text(expected)
            nb = download(page, directory, 'inserted.ipynb')
            matches = find_health(nb)
            assert len(matches) == 1, 'New health cell missing'
            verify_preserved(before, nb, cell_id(matches[0][1]))
            inserted = True
        else:
            nb = before
        index, cell = matches[0]
        verify_source(source(cell), expected, directory)
        # Stronger equality than whitespace-normalized transport verification.
        assert source(cell) == expected or source(cell) == expected.rstrip('\n'), 'Source mismatch: exact transport'
        check_idle(page, nb)
        started = datetime.datetime.now(datetime.timezone.utc)
        journal = {'job': JOB, 'cell_id': cell_id(cell), 'inserted': inserted, 'started': started.isoformat(), 'execution_submitted': False,
                   'prior_output_id': cell.get('metadata', {}).get('outputId', '')}
        (directory / 'transaction.json').write_text(json.dumps(journal, indent=2))
        region = page.get_by_role('region', name=re.compile(r'^(Celda|Cell) ' + str(index) + r':'))
        assert region.count() == 1, 'Ambiguous target cell'
        run_button = region.get_by_role('button', name=re.compile(r'^(Ejecutar celda|Run cell)$'))
        # Record intent before click: an uncertain delivery is never retried here.
        journal['execution_submitted'] = True
        (directory / 'transaction.json').write_text(json.dumps(journal, indent=2))
        run_button.click(timeout=5000)
        deadline = time.monotonic() + (EXECUTION_TIMEOUT_SECONDS or (900 if JOB == 'preflight' else 600 if JOB == 'restore' else 150))
        iteration = 0
        progress_seen = set()
        while time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
            iteration += 1
            after = download(page, directory, f'poll_{iteration}.ipynb')
            verify_preserved(nb, after)
            found = [c for c in after['cells'] if cell_id(c) == journal['cell_id']]
            assert len(found) == 1
            if JOB == 'phase_v':
                for line in output_text(found[0]).splitlines():
                    if line.startswith('M10_PROGRESS ') and line not in progress_seen:
                        print(line, flush=True)
                        progress_seen.add(line)
            result = fresh_result(found[0], started, journal['prior_output_id'])
            if result is not None:
                report = {'status': 'PROBE_COMPLETED', 'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          'notebook': page.url, 'runtime_reconnected': reconnection, 'health_cell_id': journal['cell_id'],
                          'inserted': inserted, 'cells_before': len(before['cells']), 'cells_after': len(after['cells']),
                          'source_sha256': PIN, 'result': result, 'training_enabled': JOB == 'phase_v' and TRAINING_AUTHORIZED}
                (directory / 'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=True))
                return report
            if any(o.get('output_type') == 'error' for o in found[0].get('outputs', [])):
                raise RuntimeError('Diagnostic cell errored; inspect local backup, no automatic retry')
        raise RuntimeError('No fresh result; not resubmitting. Inspect transaction and notebook')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--job', choices=['health', 'restore', 'preflight'], default='health')
    args = parser.parse_args()
    select_job(args.job)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = ROOT / 'output/axxel_start.lock'
    lock.parent.mkdir(exist_ok=True)
    try:
        handle = lock.open('x')
    except FileExistsError:
        raise SystemExit('Another start/uncertain previous transaction: inspect output/axxel_start.lock')
    try:
        handle.write(json.dumps({'audit': str(args.output), 'id': uuid.uuid4().hex}))
        handle.close()
        result = run(args.output)
        print(json.dumps({'job': JOB, 'status': result['status'], 'result_file': str(args.output / 'result.json'), 'inserted': result['inserted']}, ensure_ascii=True))
    except Exception as exc:
        error = {'status': 'BLOCKED', 'error': str(exc), 'training_enabled': False}
        (args.output / 'error.json').write_text(json.dumps(error, indent=2))
        print(json.dumps(error))
        raise SystemExit(2)
    finally:
        if not handle.closed:
            handle.close()
        lock.unlink()


if __name__ == '__main__':
    main()
