import hashlib, json, pathlib, unittest, uuid

ROOT=pathlib.Path(__file__).resolve().parents[1]

def verify_manifest(root, manifest):
    for rel, item in manifest.items():
        path=(root/rel).resolve()
        if root.resolve() not in path.parents: raise ValueError('path escape')
        if not path.is_file() or path.stat().st_size!=item['bytes']: raise ValueError('size mismatch')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']: raise ValueError('hash mismatch')

class ControllerAcceptance(unittest.TestCase):
    def test_persistent_mission_evidence(self):
        e=json.loads((ROOT/'m06_staging/m07_reconciliation_20261120T224952Z/final_result.json').read_text())
        self.assertEqual(e['report']['status'],'reconciliation_and_recovery_verified')
        self.assertFalse(e['report']['phase_V_started'])
        self.assertTrue((ROOT/'AGENTS.md').is_file())
    def test_manifest_accepts_and_rejects_corruption_in_isolated_copy(self):
        d=ROOT/'tests'/('_isolated_'+uuid.uuid4().hex)
        d.mkdir()
        try:
            p=d/'state.bin';p.write_bytes(b'valid')
            manifest={'state.bin':{'bytes':5,'sha256':hashlib.sha256(b'valid').hexdigest()}}
            verify_manifest(d,manifest)
            p.write_bytes(b'bad!!')
            with self.assertRaises(ValueError):verify_manifest(d,manifest)
        finally:
            for item in d.iterdir():item.unlink()
            d.rmdir()
    def test_no_historical_cell_execution_in_controller(self):
        src=(ROOT/'scripts/axxel.ps1').read_text(encoding='utf-8')
        self.assertIn('notebook_before.ipynb',src)
        self.assertNotIn('Control+Enter',src)
        self.assertIn('1W7zo2Q-gXoyiVu2nZNnOMJ5RXG8Hymss',src)
    def test_ignore_rules_cover_sensitive_and_large_files(self):
        ignore=(ROOT/'.gitignore').read_text(encoding='utf-8')
        for value in ['.env','*.pt','*.parquet','output/'] : self.assertIn(value,ignore)

if __name__=='__main__':unittest.main()
