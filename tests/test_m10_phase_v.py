import ast
import copy
import hashlib
import io
import math
from pathlib import Path
import uuid
import unittest

import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
TREE=ast.parse((ROOT/'m06_staging/m10_phase_v.py').read_text(encoding='utf-8'))

def definition(name,namespace):
    node=copy.deepcopy(next(n for n in ast.walk(TREE) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name))
    # Only device placement changes for the isolated CPU fixture.
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and n.value=='cuda':n.value='cpu'
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'actual_m10_definition','exec'),namespace)
    return namespace[name]

class PhaseVChecks(unittest.TestCase):
    def test_persistence_conflicts_never_overwrite(self):
        ns={'Path':Path,'hashlib':hashlib,'os':__import__('os')}
        definition('digest',ns);put=definition('put',ns)
        temp_root=ROOT/'output/local_tests';temp_root.mkdir(parents=True,exist_ok=True)
        d=temp_root/uuid.uuid4().hex
        d.mkdir()
        try:
            p=d/'checkpoint';put(p,b'original');put(p,b'original')
            with self.assertRaisesRegex(AssertionError,'Conflicting'):put(p,b'changed')
            self.assertEqual(p.read_bytes(),b'original')
        finally:
            if (d/'checkpoint').exists():(d/'checkpoint').unlink()
            d.rmdir()

    def test_actual_normalization_matches_inplace_float32_scaler(self):
        import numpy as np
        h=np.array([[.3,2.1],[3.2,-4.3]],dtype=np.float32)
        mean=np.array([.123456789,.543212345],dtype=np.float64)
        scale=np.array([1.23456789,2.3456789],dtype=np.float64)
        expected=h.copy();expected-=mean;expected/=scale
        normalize=definition('normalize',{'mean':torch.tensor(mean),'scale':torch.tensor(scale)})
        self.assertTrue(np.array_equal(normalize(torch.tensor(h)).numpy(),expected))

    def test_lora_zero_step_and_exact_optimizer_rng_resume(self):
        LoRA=definition('LoRA',{'torch':torch,'nn':nn,'F':F,'math':math})
        torch.manual_seed(271828)
        layer=LoRA(nn.Linear(5,4));layer.eval();x=torch.randn(8,5);target=torch.ones(8,4)
        self.assertTrue(torch.equal(layer(x),layer.base(x)))
        layer.dropout.train();opt=torch.optim.AdamW([layer.lora_A,layer.lora_B],lr=1e-4,weight_decay=0)
        def step(net,optimizer):
            optimizer.zero_grad();loss=F.smooth_l1_loss(net(x),target);loss.backward()
            self.assertIsNone(net.base.weight.grad)
            optimizer.step();return loss.item()
        step(layer,opt)
        self.assertEqual(layer.lora_A.grad.abs().sum().item(),0)
        b=io.BytesIO();torch.save({'model':layer.state_dict(),'opt':opt.state_dict(),'rng':torch.get_rng_state()},b)
        expected=step(layer,opt)
        self.assertGreater(layer.lora_A.grad.abs().sum().item(),0)
        packet=torch.load(io.BytesIO(b.getvalue()),weights_only=True)
        restored=copy.deepcopy(layer);restored.load_state_dict(packet['model'],strict=True)
        other=torch.optim.AdamW([restored.lora_A,restored.lora_B],lr=1e-4,weight_decay=0);other.load_state_dict(packet['opt'])
        torch.set_rng_state(packet['rng']);self.assertEqual(step(restored,other),expected)
        for k,v in layer.state_dict().items():self.assertTrue(torch.equal(v,restored.state_dict()[k]))

    def test_four_microbatches_match_effective_batch_gradient(self):
        torch.manual_seed(4);a=nn.Sequential(nn.Linear(5,8),nn.GELU(),nn.Linear(8,4));b=copy.deepcopy(a)
        x=torch.randn(64,5);y=torch.randn(64,4)
        F.smooth_l1_loss(a(x),y).backward()
        for i in range(4):(F.smooth_l1_loss(b(x[i*16:(i+1)*16]),y[i*16:(i+1)*16])/4).backward()
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-7)

    def test_daily_transport_denies_training_without_authorization(self):
        import sys
        from unittest.mock import patch
        sys.path.insert(0,str(ROOT/'automation'))
        import axxel_colab_start as transport
        with patch.object(transport,'JOB','phase_v'),patch.object(transport,'TRAINING_AUTHORIZED',False):
            with self.assertRaisesRegex(RuntimeError,'requires explicit authorization'):transport.run(ROOT/'output')

if __name__=='__main__':unittest.main()
