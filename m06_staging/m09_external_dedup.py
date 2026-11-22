# M09_EXTERNAL_DEDUP_V1
def external_inventory():
    import datetime, hashlib, json, os, re
    from pathlib import Path
    root=Path('/content/drive/MyDrive/AXXEL_CAPITAL')
    report={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'directories':[], 'metadata':[], 'checkpoints':[], 'excluded':[], 'errors':[], 'test_read':False,'optimizer_steps':0}
    assert os.path.ismount('/content/drive') and root.is_dir()
    forbidden=re.compile(r'2023|2024|dataset|\btest\b|holdout|reserved|reservado|secret|credential|token|ticks',re.I)
    keys={'experiment','status','config','configuration','training','train','model','model_id','tokenizer','seed','seeds','steps','step','max_steps','lr','learning_rate','optimizer','loss','objective','arms','variants','rank','alpha','lora','lora_config','targets','target_modules','trainable_parameters','base_hash','tokenizer_hash','pipeline_sha256','scientific_fingerprint','fingerprint','context','context_length','L','H','horizon','timeframe','budget','head','head_trainable','aux_head','metrics','epochs'}
    count=0
    for directory,dirs,files in os.walk(root):
        rel=Path(directory).relative_to(root)
        dirs[:]=sorted(d for d in dirs if not forbidden.search(d) and d not in ('MISION06_ALINEACION','__pycache__','.git'))
        report['directories'].append(str(rel))
        count+=1
        if count>10000:report['errors'].append('Directory bound reached');break
        for name in sorted(files):
            p=Path(directory)/name;rp=str(p.relative_to(root))
            if forbidden.search(name):
                report['excluded'].append(rp);continue
            if p.suffix.lower() in ('.pt','.pth','.safetensors'):
                report['checkpoints'].append({'path':rp,'bytes':p.stat().st_size})
            if p.suffix.lower()!='.json':continue
            # Read configuration/manifests only; never evaluation metrics or data files.
            if not any(k in name.lower() for k in ('config','manifest','protoc','status','registry','training','resume','param')):
                continue
            try:
                if p.stat().st_size>2000000:raise ValueError('Metadata review bound exceeded')
                raw=p.read_bytes();obj=json.loads(raw)
                report['metadata'].append({'path':rp,'sha256':hashlib.sha256(raw).hexdigest(),'keys':sorted(obj) if isinstance(obj,dict) else ['list'], 'scientific_fields':{k:v for k,v in obj.items() if k in keys and k!='metrics'} if isinstance(obj,dict) else {}})
            except Exception as exc:report['errors'].append({'path':rp,'error':type(exc).__name__+': '+str(exc)})
    print('M09_EXTERNAL_RESULT '+json.dumps(report),flush=True)
external_inventory()
