import json, pathlib, unittest, uuid
from m07_integrity import transport, verify_source, verify_preserved, assert_idle

class IntegrityTests(unittest.TestCase):
    def test_indent_corruption_rejected(self):
        expected='# M07_TEST\nif True:\n    x=1\ny=2\n'
        tmp=pathlib.Path(__file__).resolve().parents[1]/'tests'/('_integrity_'+uuid.uuid4().hex)
        tmp.mkdir()
        try:
            with self.assertRaisesRegex(AssertionError,'Source mismatch'):
                verify_source(expected.replace('\ny=','\n    y='),expected,tmp)
        finally:
            for item in tmp.iterdir():item.unlink()
            tmp.rmdir()
    def test_transport_roundtrip_and_behavior(self):
        code='# M07_TEST\nif False:\n    x=1\ny="literal \\n and quote"\n'
        wire=transport(code,'probe.py')
        self.assertEqual(len(wire.splitlines()),2)
        namespace={};exec(wire,namespace)
        self.assertNotIn('x',namespace)
        self.assertEqual(namespace['y'],'literal \n and quote')
    def test_deleted_cell_rejected(self):
        c={'id':'a','cell_type':'code','source':['x=1']}
        with self.assertRaises(AssertionError):verify_preserved({'cells':[c]},{'cells':[]})
    def test_busy_rejected(self):
        with self.assertRaises(AssertionError):assert_idle({'cells':[]},'Esperando a que finalice la ejecución actual. Conectado a GPU')
    def test_duplicate_id_rejected(self):
        c={'id':'a','cell_type':'code','source':['x=1']}
        with self.assertRaises(AssertionError):verify_preserved({'cells':[c]},{'cells':[c,c]})

if __name__=='__main__':unittest.main()
