import ast
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / 'automation/m08_readonly_status.py'
spec = importlib.util.spec_from_file_location('m08_status', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ReadOnlyStatusTests(unittest.TestCase):
    def test_exact_identity_rejects_similar_urls(self):
        self.assertTrue(m.exact_notebook(m.URL + '?authuser=0#scrollTo=x'))
        for url in [m.URL + 'extra', m.URL.replace('https:', 'http:'), m.URL.replace('colab.research.google.com', 'evil.example')]:
            self.assertFalse(m.exact_notebook(url))

    def test_current_connection_does_not_promote_saved_results(self):
        result = m.classify({'exact_tabs': 1, 'url': m.URL, 'connection': 'Conectado a (GPU) RAM: 1.59 GB/12.67 GB'}, 'now')
        self.assertEqual(result['runtime']['value'], 'connected')
        self.assertEqual(result['gpu']['status'], 'CURRENT')
        for key in ['drive', 'models', 'cuda', 'pytorch', 'gpu_memory', 'training_processes', 'checkpoint', 'active_cells']:
            self.assertEqual(result[key]['status'], 'UNKNOWN', key)
        self.assertEqual(result['system_memory']['status'], 'CURRENT')

    def test_disconnected_ambiguous_and_missing_are_conservative(self):
        for text, expected in [('Disconnected', 'CURRENT'), ('button "Conectar"', 'CURRENT'), ('Volver a conectar T4 Hacer clic para conectarse', 'CURRENT'), ('Connecting', 'UNKNOWN'), ('Connected to Disconnected', 'UNKNOWN')]:
            result = m.classify({'exact_tabs': 1, 'url': m.URL, 'connection': text}, 'now')
            self.assertEqual(result['runtime']['status'], expected)
            self.assertEqual(result['gpu']['status'], 'UNKNOWN')
        for count in [0, 2]:
            self.assertTrue(all(x['status'] == 'UNKNOWN' for x in m.classify({'exact_tabs': count, 'url': m.URL}, 'now').values()))

    def test_history_has_original_date_and_local_source_hash(self):
        history = m.historical()
        self.assertTrue(history)
        for entry in history.values():
            self.assertEqual(entry['status'], 'HISTORICAL')
            self.assertEqual(entry['date'], '2026-11-20T22:59:16.904248+00:00')
            self.assertIn('sha256=', entry['evidence'])

    def test_readonly_browser_call_allowlist(self):
        tree = ast.parse(PATH.read_text(encoding='utf-8'))
        forbidden = {'click', 'press', 'evaluate', 'evaluate_handle', 'send', 'goto', 'new_page', 'new_context', 'launch', 'close', 'fill', 'insert_text', 'request', 'post', 'exec', 'eval'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
                self.assertNotIn(name, forbidden)

    def test_active_cell_indicator_not_training_assertion(self):
        result = m.classify({'exact_tabs': 1, 'url': m.URL, 'connection': 'Connected to (GPU)', 'stop_buttons': 1}, 'now')
        self.assertEqual(result['active_cells']['status'], 'CURRENT')
        self.assertEqual(result['training_processes']['status'], 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
