"""Control seguro de Kronos_SFT_verificado mediante Chrome CDP.

Python 3.10+. Dependencia: python -m pip install playwright
No descarga Chromium, abre Chrome ni cambia firewall/permisos.
Ejecuta secciones 1-8; RUN_FULL permanece False. No hace evaluación ampliada.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

NOTEBOOK_ID = "1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss"
TITLE = "Kronos_SFT_verificado.ipynb"
NOTEBOOK_URL = f"https://colab.research.google.com/drive/{NOTEBOOK_ID}"
EXPECTED = [
    ("91a8ed4382", "9250f0f4f7bc2c7dcf246229115337d0e391dad5c20ad7dfe84a24b825f0b08a"),
    ("a401cc0e76", "a0ed447340167d5a0faabc133c77b7fb570ceae93bb483c8509879fccad3cc75"),
    ("2996f1757a", "d5a12dfaa12778f3c15b5c27c152369364e15785577ec285faa40bf4f4f31542"),
    ("012e0eee30", "05b41402e737810ed20788e810793cc0d319385e2a32bed57da70e9cd9d6d609"),
    ("aa19065f86", "457587795b3c0154d70954149e343951fae41da6f5fa155da410e3e373d32e16"),
    ("115826fbfc", "f57aa0d22dd185a4d496c9819255ad16835e9e2bc754243198b2774a107b763f"),
    ("6097261eb4", "b5b10030484eb18d22b71c1148415165ffac31cdf9c3360d828e6de5f1d0650d"),
    ("63c268d733", "992c96ddba49ce6da41b97c22104fbe5300ba4d68769f43d371c61b3f779d5e2"),
    ("3efdb367cf", "a21d681f09f5c926e9a6bfbb946fc220b84bd13ca7cc573d2237fa53463c167a"),
]
LOG = logging.getLogger("kronos_cdp")


class AutomationError(RuntimeError):
    pass


def valid_notebook_url(url):
    p = urllib.parse.urlsplit(url)
    return (p.scheme == "https" and p.hostname == "colab.research.google.com"
            and p.port in (None, 443) and not p.username and not p.password
            and p.path.rstrip("/") == f"/drive/{NOTEBOOK_ID}")


def ensure_page(page):
    if page.is_closed() or not valid_notebook_url(page.url):
        raise AutomationError("La pestaña dejó de ser el cuaderno autorizado. Se detuvo el control.")


def endpoint_check(endpoint):
    p = urllib.parse.urlsplit(endpoint)
    if (p.scheme != "http" or p.hostname not in ("localhost", "127.0.0.1", "::1")
            or p.username or p.password or p.query or p.fragment or p.path not in ("", "/")):
        raise AutomationError("El endpoint debe ser HTTP local, sin credenciales ni rutas.")
    endpoint = endpoint.rstrip("/")
    # No se consultan cookies, tokens, almacenamiento de Chrome ni páginas ajenas.
    try:
        with urllib.request.urlopen(endpoint + "/json/version", timeout=5) as response:
            version = json.load(response)
        if not version.get("webSocketDebuggerUrl"):
            raise AutomationError("El puerto responde, pero no expone un endpoint CDP.")
        with urllib.request.urlopen(endpoint + "/json/list", timeout=5) as response:
            targets = json.load(response)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise AutomationError(f"No fue posible acceder a {endpoint}: {exc}") from exc
    matches = [t for t in targets if t.get("type") == "page" and valid_notebook_url(t.get("url", ""))]
    if not matches:
        raise AutomationError("CDP responde, pero no contiene la URL exacta de Kronos_SFT_verificado.")
    if len(matches) != 1:
        raise AutomationError("Hay varias pestañas del mismo cuaderno. Deja una sola antes de ejecutar.")
    LOG.info("CDP accesible: %s; cuaderno identificado por URL.", version.get("Browser"))
    return endpoint


def cell_id(cell):
    return cell.get("id") or cell.get("metadata", {}).get("id")


def cell_text(cell):
    text = cell.get("source", "")
    return "".join(text) if isinstance(text, list) else text


def output_text(cell):
    parts = []
    for output in cell.get("outputs", []):
        value = output.get("text", output.get("data", {}).get("text/plain", ""))
        parts.append("".join(value) if isinstance(value, list) else value)
    return "\n".join(parts)


def code_cells(notebook):
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def verify_notebook(notebook, reject_running=False):
    cells = code_cells(notebook)
    if len(cells) != len(EXPECTED):
        raise AutomationError("Cambió la cantidad de celdas; se requiere revisar el cuaderno.")
    for number, (cell, (expected_id, expected_hash)) in enumerate(zip(cells, EXPECTED), 1):
        digest = hashlib.sha256(cell_text(cell).encode("utf-8")).hexdigest()
        if cell_id(cell) != expected_id or digest != expected_hash:
            raise AutomationError(f"La sección {number} cambió respecto al código revisado. No se ejecutará.")
        state = cell.get("metadata", {}).get("executionInfo", {}).get("status")
        if reject_running and state in ("busy", "running", "pending"):
            raise AutomationError("Hay una ejecución activa; no se reinicia ni se envían ejecuciones adicionales.")
    # El hash de la sección 8 certifica RUN_FULL=False; el de la 9, RESUME=False.
    return cells


def visible_one(locator):
    found = [locator.nth(i) for i in range(locator.count()) if locator.nth(i).is_visible()]
    return found[0] if len(found) == 1 else None


def menu_item(page, patterns):
    for pattern in patterns:
        for role in ("menuitem", "button"):
            item = visible_one(page.get_by_role(role, name=re.compile(pattern, re.I)))
            if item is not None:
                return item
        item = visible_one(page.get_by_text(re.compile(pattern, re.I)))
        if item is not None:
            return item
    raise AutomationError("No se encontró un control inequívoco: " + "/".join(patterns))


def download_notebook(page, destination):
    """Descarga el estado actual mediante la interfaz pública, sin APIs privadas de Colab."""
    ensure_page(page)
    page.keyboard.press("Escape")
    menu_item(page, [r"^Archivo$", r"^File$"]).click()
    download = menu_item(page, [r"^Descargar$", r"^Download$"])
    download.hover()
    page.wait_for_timeout(400)
    try:
        item = menu_item(page, [r"^(Descargar\s+|Download\s+)?\.ipynb$"])
    except AutomationError:
        download.click()
        page.wait_for_timeout(300)
        item = menu_item(page, [r"^(Descargar\s+|Download\s+)?\.ipynb$"])
    with page.expect_download(timeout=30000) as pending:
        item.click()
    result = pending.value
    failure = result.failure()
    if failure:
        raise AutomationError("La descarga del estado del cuaderno falló: " + failure)
    result.save_as(str(destination))
    notebook = json.loads(destination.read_text(encoding="utf-8"))
    verify_notebook(notebook)
    return notebook


def restart_runtime(page):
    ensure_page(page)
    menu_item(page, [r"^Entorno de ejecución$", r"^Runtime$"]).click()
    menu_item(page, [r"^Reiniciar sesión(?:\s*\(.+\))?$", r"^Restart session(?:\s*\(.+\))?$",
                     r"^Reiniciar entorno de ejecución(?:\s*\(.+\))?$",
                     r"^Restart runtime(?:\s*\(.+\))?$"]).click()
    page.wait_for_timeout(500)
    dialog = visible_one(page.get_by_role("dialog"))
    if dialog is None:
        raise AutomationError("No se encontró la confirmación de reinicio; comprueba Colab antes de seguir.")
    if not re.search(r"reiniciar|restart", dialog.inner_text(), re.I):
        raise AutomationError("Apareció un diálogo distinto del reinicio esperado.")
    confirm = visible_one(dialog.get_by_role("button", name=re.compile(r"^(Reiniciar|Restart|Sí|Yes)$", re.I)))
    if confirm is None:
        raise AutomationError("No se encontró una confirmación inequívoca de reinicio.")
    confirm.click()
    dialog.wait_for(state="hidden", timeout=30000)
    LOG.info("Reinicio solicitado. La primera celda esperará al entorno de Colab.")


def execute_cell(page, identifier):
    ensure_page(page)
    page.goto(NOTEBOOK_URL + "#scrollTo=" + identifier, wait_until="domcontentloaded")
    ensure_page(page)
    selectors = [f'[id="cell-{identifier}"]', f'[id="{identifier}"]', f'[data-cell-id="{identifier}"]']
    cell = None
    for selector in selectors:
        candidate = page.locator(selector)
        try:
            candidate.first.wait_for(state="visible", timeout=2500)
        except Exception:
            continue
        cell = visible_one(candidate)
        if cell is not None:
            break
    if cell is None:
        raise AutomationError("La interfaz no expone la celda esperada: " + identifier)
    cell.scroll_into_view_if_needed()
    for selector in (".monaco-editor textarea", ".CodeMirror textarea", "textarea", '[role="textbox"]'):
        editor = visible_one(cell.locator(selector))
        if editor is not None:
            editor.focus()
            ensure_page(page)
            page.keyboard.press("Control+Enter")
            return
    # Colab puede presentar una celda plegada sin editor. Ejecutar su botón es equivalente.
    run = visible_one(cell.get_by_role("button", name=re.compile(r"^(Ejecutar celda|Run cell)", re.I)))
    if run is None:
        raise AutomationError("No se encontró el editor ni el botón de ejecutar de la celda identificada.")
    ensure_page(page)
    run.click()


def execution_info(cell):
    return cell.get("metadata", {}).get("executionInfo", {})


def fresh_completion(before, after):
    previous, current = execution_info(before), execution_info(after)
    # Una salida antigua o un contador visual no cuentan como una ejecución nueva.
    return (current.get("timestamp") is not None
            and current.get("timestamp") != previous.get("timestamp")
            and current.get("status") in ("ok", "error"))


def wait_for_cell(page, before, number, directory, timeout, poll):
    deadline = time.monotonic() + timeout
    previous_progress = None
    while time.monotonic() < deadline:
        ensure_page(page)
        page.wait_for_timeout(poll * 1000)
        dialog = visible_one(page.get_by_role("dialog"))
        if dialog is not None:
            raise AutomationError("Colab requiere intervención en un diálogo. Atiéndelo directamente en el navegador.")
        notebook = download_notebook(page, directory / "estado_actual.ipynb")
        after = code_cells(notebook)[number - 1]
        info = execution_info(after)
        progress = (info.get("status"), info.get("timestamp"))
        if progress != previous_progress:
            LOG.info("Sección %s: estado=%s; esperando evidencia de ejecución nueva.", number, info.get("status", "sin estado"))
            previous_progress = progress
        if not fresh_completion(before, after):
            continue
        errors = [o for o in after.get("outputs", []) if o.get("output_type") == "error"]
        if errors or info.get("status") == "error":
            detail = "; ".join(f"{e.get('ename')}: {e.get('evalue')}" for e in errors)
            raise AutomationError(f"Falló la sección {number}: {detail or 'error reportado por Colab'}")
        if number == 3 and "kronos_pipeline.py" not in output_text(after):
            raise AutomationError("La sección 3 terminó sin confirmar la ruta del pipeline.")
        LOG.info("Sección %s completada y verificada.", number)
        return notebook
    raise AutomationError(f"Tiempo agotado en sección {number}. Puede seguir ejecutándose en Colab; no se reenvía.")


def read_status(notebook):
    raw = output_text(code_cells(notebook)[6]).lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(raw)
    except ValueError as exc:
        raise AutomationError("La sección 7 no contiene un estado JSON verificable.") from exc
    return value


def verify_final(notebook, old_run_id):
    state = read_status(notebook)
    keys = ("data_verified", "windows_prepared", "model_loaded", "smoke_passed", "sources_unchanged")
    if not all(state.get(k) is True for k in keys):
        raise AutomationError("El estado final no confirma todas las comprobaciones técnicas.")
    if not state.get("run_id") or state["run_id"] == old_run_id:
        raise AutomationError("El estado corresponde a una ejecución anterior.")
    if state.get("full_training_done") is not False or state.get("test_evaluated") is not False:
        raise AutomationError("El alcance de ejecución difiere de la prueba corta autorizada.")
    for timeframe in ("M5", "M15"):
        result = state.get("experiments", {}).get(timeframe, {}).get("training", {})
        if result.get("steps") != 3 or not all(result.get(k) is True for k in
                ("weights_changed", "tokenizer_unchanged", "checkpoint_reload_identical")):
            raise AutomationError("Resultado de entrenamiento incompleto para " + timeframe)
    return state


def run(args, evidence):
    evidence["playwright_installed"] = importlib.util.find_spec("playwright") is not None
    endpoint = endpoint_check(args.endpoint)
    evidence["cdp_accessible"] = True
    if not evidence["playwright_installed"]:
        raise AutomationError("Falta Playwright en este Python. Instala con: python -m pip install playwright")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        # Se conecta al Chrome existente. No inicia otro navegador.
        browser = playwright.chromium.connect_over_cdp(endpoint, timeout=15000)
        pages = [p for context in browser.contexts for p in context.pages if valid_notebook_url(p.url)]
        if len(pages) != 1:
            raise AutomationError("No se encontró exactamente una pestaña autorizada después de conectar.")
        page = pages[0]
        ensure_page(page)
        if TITLE.casefold() not in page.title().casefold():
            raise AutomationError("La URL coincide, pero el título del cuaderno es distinto.")
        page.bring_to_front()
        page.set_default_timeout(10000)
        with tempfile.TemporaryDirectory(prefix="kronos_cdp_") as temporary:
            directory = Path(temporary)
            notebook = download_notebook(page, directory / "estado_inicial.ipynb")
            verify_notebook(notebook, reject_running=True)
            evidence["live_notebook_verified"] = True
            if args.check_only:
                evidence["outcome"] = "CONNECTION_AND_NOTEBOOK_VERIFIED_NO_EXECUTION"
                return
            try:
                old_run_id = read_status(notebook).get("run_id")
            except AutomationError:
                old_run_id = None
            if args.restart_runtime:
                restart_runtime(page)
            elif execution_info(code_cells(notebook)[2]).get("timestamp"):
                raise AutomationError("El cuaderno ya ejecutó el módulo. Usa --restart-runtime para repetir 1-8 con un runtime limpio.")
            for number, (identifier, _) in enumerate(EXPECTED[:8], 1):
                before = code_cells(notebook)[number - 1]
                LOG.info("Ejecutando sección %s/8.", number)
                execute_cell(page, identifier)
                evidence["execution_requested"] = True
                notebook = wait_for_cell(page, before, number, directory, args.timeout, args.poll)
                evidence["completed_sections"].append(number)
                if number == 3:
                    LOG.info("kronos_pipeline.py extraído, importado y verificado por la sección 3.")
            evidence["status"] = verify_final(notebook, old_run_id)
            evidence["outcome"] = "SMOKE_VERIFIED_SECTIONS_1_TO_8"
            LOG.info("FINALIZADO: prueba corta M5/M15 confirmada; run_id=%s", evidence["status"]["run_id"])
        # No browser.close(): el navegador pertenece al usuario. Salir de Playwright desconecta al cliente.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://localhost:9222")
    parser.add_argument("--restart-runtime", action="store_true", help="Reinicia el runtime; descarta sus variables en memoria.")
    parser.add_argument("--check-only", action="store_true", help="Verifica conexión y cuaderno sin ejecutar celdas.")
    parser.add_argument("--timeout", type=int, default=1800, help="Máximo de segundos por celda.")
    parser.add_argument("--poll", type=int, default=10, help="Intervalo de comprobación en segundos.")
    parser.add_argument("--result", type=Path, default=Path(__file__).with_name("resultado_control_colab.json"))
    args = parser.parse_args()
    if args.timeout < 1 or not 1 <= args.poll <= 30:
        parser.error("timeout debe ser positivo y poll debe estar entre 1 y 30.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evidence = {"started_utc": datetime.now(timezone.utc).isoformat(), "notebook_id": NOTEBOOK_ID,
                "cdp_accessible": False, "live_notebook_verified": False,
                "execution_requested": False, "completed_sections": [], "outcome": "NOT_STARTED"}
    code = 0
    try:
        run(args, evidence)
    except KeyboardInterrupt:
        code = 130
        evidence.update(outcome="INTERRUPTED", error="Interrumpido localmente. Colab puede continuar; revisa su estado.")
        LOG.error(evidence["error"])
    except Exception as exc:
        code = 1
        evidence.update(outcome="BLOCKED_OR_FAILED", error=f"{type(exc).__name__}: {exc}")
        LOG.error(evidence["error"])
        if not evidence.get("playwright_installed"):
            LOG.error("Playwright tampoco está instalado en este intérprete.")
        LOG.error("No se relanzarán automáticamente celdas ni se modificarán permisos.")
    finally:
        evidence["finished_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            args.result.parent.mkdir(parents=True, exist_ok=True)
            args.result.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            LOG.info("Diagnóstico guardado en %s", args.result.resolve())
        except OSError as exc:
            LOG.error("No fue posible guardar el diagnóstico: %s", exc)
            code = code or 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
