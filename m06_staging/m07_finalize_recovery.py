# M07_FINALIZE_RECOVERY_V1
import json, datetime, subprocess
assert M7_STEP0['status']=='passed_checks_executed_training_still_disabled'
assert M7_CHECKS['status']=='recovered_pending_data_and_step0'
assert m7_hash(M7_MODEL)==model_info['base_hash'] and m7_hash(M7_TOK)==model_info['tokenizer_hash']
# A separate gradient-only probe checks actual LoRA training dropout, without any optimizer update.
joint=copy.deepcopy(M7_MODEL)
torch.manual_seed(271828)
for block in joint.transformer:
    for leaf in ['q_proj','v_proj']:
        layer=M7LoRA(getattr(block.self_attn,leaf),8)
        nn.init.kaiming_uniform_(layer.lora_A,a=5**.5)
        setattr(block.self_attn,leaf,layer)
joint.eval()
for module in joint.modules():
    if isinstance(module,M7LoRA):module.dropout.train()
head=copy.deepcopy(M7_HEADS[271828]).train().requires_grad_(True)
probe=m7_batch(M7_TRAIN,[0,1,2,3])
h=m7_hidden(joint,probe)
hm=torch.as_tensor(M7_SCALERS['hidden_mean'],device=M7_DEVICE,dtype=torch.float64)
hs=torch.as_tensor(M7_SCALERS['hidden_scale'],device=M7_DEVICE,dtype=torch.float64)
h=((h.double()-hm).float().double()/hs).float()
y=(10000*(probe['truth'][:,:,3].double()/probe['last_close'][:,None].double()-1)/torch.as_tensor(M7_SCALERS['return_scale'],device=M7_DEVICE)).float()
F.smooth_l1_loss(head(h),y,beta=1).backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
assert any(p.grad.abs().sum()>0 for p in head.parameters())
assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for n,p in joint.named_parameters() if 'lora_B' in n)
assert all(p.grad is None for n,p in joint.named_parameters() if 'lora_' not in n)
assert all(p.grad is None for p in M7_TOK.parameters())
del joint,head
for seed,pred in M7_PRED0.items():
    p=M7_OUT/f'step0_predictions_s{seed}.npz'
    with p.open('xb') as f:np.savez_compressed(f,prediction=pred,truth=M7_TRUTH,last_close=M7_LAST,window=np.arange(950))
    with np.load(p,allow_pickle=False) as z:
        assert np.array_equal(z['prediction'],pred) and np.array_equal(z['truth'],M7_TRUTH)
report={'status':'reconciliation_and_recovery_verified','timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'step0':M7_STEP0,'dropout_train_gradient_probe_passed':True,'optimizer_steps':0,
        'phase_V_started':False,'phase_V_authorized':False,'RESEARCH_ONLY':True,'test_read':False,'orders_enabled':False,
        'pending':['scientific deduplication beyond metadata inventory','later LoRA_A learning and optimizer/RNG resume equivalence without altering historical artifacts','full new namespace recovery twice including data/model/optimizer','Phase V implementation and recovery transaction tests'],
        'gpu':subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader'],capture_output=True,text=True).stdout}
with (M7_OUT/'final_recovery_report.json').open('x') as f:json.dump(report,f,indent=2,default=str)
final_manifest={p.name:{'sha256':m7_sha(p),'bytes':p.stat().st_size} for p in M7_OUT.iterdir() if p.is_file()}
manifest_path=M7_OUT/'final_checksums.json'
with manifest_path.open('x') as f:json.dump(final_manifest,f,indent=2)
for rel,expected in json.loads(manifest_path.read_text()).items():
    p=M7_OUT/rel
    assert p.stat().st_size==expected['bytes'] and m7_sha(p)==expected['sha256']
print('M07_FINAL_RECOVERY',json.dumps({'report':report,'output':str(M7_OUT),'manifest_sha256':m7_sha(manifest_path),'files_verified':len(final_manifest)},default=str),flush=True)
