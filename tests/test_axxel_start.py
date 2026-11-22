import ast
import datetime
import json
import pathlib
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'automation'))
import axxel_colab_start as runner


class StartSafety(unittest.TestCase):
    def test_pinned_source_changes_rejected(self):
        with patch.object(pathlib.Path, 'read_bytes', return_value=b'print("altered")'):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                runner.payload()

    def test_duplicate_health_cells_rejected(self):
        cell = {'source': runner.MARKER + '\npass'}
        with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
            runner.find_health({'cells': [cell, cell]})

    def test_old_result_not_accepted(self):
        start = datetime.datetime(2026, 11, 21, tzinfo=datetime.timezone.utc)
        def cell(stamp):
            return {'outputs': [{'text': runner.RESULT_PREFIX + json.dumps({'timestamp_utc': stamp})}]}
        self.assertIsNone(runner.fresh_result(cell('2026-11-20T00:00:00+00:00'), start))
        self.assertIsNotNone(runner.fresh_result(cell('2026-11-21T00:00:01+00:00'), start))

    def test_restore_has_no_training_or_artifact_writes(self):
        code = (ROOT / 'automation/axxel_restore_runtime.py').read_text(encoding='utf-8')
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ['backward', 'step', 'save', 'save_pretrained', 'to_parquet', 'to_csv', 'write_text', 'write_bytes'])
        self.assertIn('weights_only=True', code)
        self.assertIn('features_development_2020_', code)
        self.assertIn('full_data_preflight', code)

    def test_output_identity_handles_clock_skew_without_stale_acceptance(self):
        local = datetime.datetime(2026, 11, 21, tzinfo=datetime.timezone.utc)
        cell = {'metadata': {'outputId': 'new'}, 'outputs': [{'text': runner.RESULT_PREFIX + json.dumps({'timestamp_utc': '2026-11-20T23:59:59.800000+00:00'})}]}
        self.assertIsNotNone(runner.fresh_result(cell, local, 'old'))
        self.assertIsNone(runner.fresh_result(cell, local, 'new'))

    def test_restore_and_health_are_only_allowed_jobs(self):
        code = (ROOT / 'automation/axxel_colab_start.py').read_text(encoding='utf-8')
        self.assertIn("choices=['health', 'restore', 'preflight']", code)
        self.assertNotIn('--job preflight', (ROOT/'scripts/axxel.ps1').read_text(encoding='utf-8-sig'))
        self.assertIn("journal['execution_submitted'] = True", code)
        self.assertIn('Previous execution uncertain; no resubmission', code)


if __name__ == '__main__':
    unittest.main()
