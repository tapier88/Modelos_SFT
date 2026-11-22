import ast
import pathlib
import re
import subprocess
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]

class RecoverSafety(unittest.TestCase):
    def test_execution_function_and_all_four_calls_removed(self):
        text=(ROOT/'scripts/axxel.ps1').read_text(encoding='utf-8-sig')
        for forbidden in ['Invoke-VerifiedCell','M07_REMOTE_INVENTORY','M07_RECOVERY_PREFLIGHT','M07_DATA_STEP0','M07_FINALIZE_RECOVERY','Ejecutar celda','insert_text']:
            self.assertNotIn(forbidden,text)
        self.assertIn('RECOVER_DIAGNOSTIC_ONLY',text)

    def test_only_embedded_python_is_download_code(self):
        text=(ROOT/'scripts/axxel.ps1').read_text(encoding='utf-8-sig')
        blocks=re.findall(r"\$py=@'\n(.*?)\n'@",text,re.S)
        self.assertEqual(len(blocks),1)
        tree=ast.parse(blocks[0])
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        for call in calls:
            if isinstance(call.func,ast.Name):self.assertNotIn(call.func.id,['exec','eval','compile'])
            if isinstance(call.func,ast.Attribute):self.assertNotIn(call.func.attr,['evaluate','insert_text','new_page','goto','send','step','backward'])
        presses=[n.args[0].value for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='press']
        self.assertEqual(presses,['Escape'])

    def test_four_commands_local_simulation(self):
        r=subprocess.run(['powershell.exe','-NoProfile','-File',str(ROOT/'tests/recover_safety.ps1')],capture_output=True,text=True,timeout=60)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertIn('PASS: 16 cases',r.stdout)

if __name__=='__main__':unittest.main()
