# M09_DEDUP_INVENTORY_V1
def m09_inventory():
    import datetime, hashlib, json, os
    from pathlib import Path
    root=Path('/content/drive/MyDrive/AXXEL_CAPITAL/MISION06_ALINEACION')
    result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':str(root),'metadata':[],'checkpoints':[],'excluded_directories':[],'errors':[],'test_read':False,'optimizer_steps':0}
    assert os.path.ismount('/content/drive') and root.is_dir()
    def digest(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1048576),b''):h.update(b)
        return h.hexdigest()
    for directory,dirs,files in os.walk(root):
        keep=[]
        for d in dirs:
            if any(x in d.lower() for x in ('2023','2024','dataset','secret','credential','token','cache','__pycache__')):
                result['excluded_directories'].append(str((Path(directory)/d).relative_to(root)))
            else:keep.append(d)
        dirs[:]=keep
        for name in files:
            if any(x in name.lower() for x in ('2023','2024','secret','credential','token')):continue
            p=Path(directory)/name;rel=str(p.relative_to(root))
            if p.suffix in ('.pt','.pth','.safetensors'):
                result['checkpoints'].append({'path':rel,'bytes':p.stat().st_size,'sha256':digest(p)})
            elif p.suffix=='.json':
                if p.stat().st_size>2000000:
                    result['errors'].append({'path':rel,'error':'metadata exceeds review bound'});continue
                try:
                    obj=json.loads(p.read_text())
                    entry={'path':rel,'sha256':digest(p),'keys':sorted(obj) if isinstance(obj,dict) else ['list']}
                    if isinstance(obj,dict):
                        keys=('experiment','status','arms','variants','optimizer','loss','budget','seeds','seed','steps','step','base_hash','tokenizer_hash','pipeline_sha256','head_initialization','shared_model','rank','alpha','lr','aux_head','trainable_parameters','fingerprint','scientific_fingerprint')
                        entry['scientific_fields']={k:obj[k] for k in keys if k in obj}
                    result['metadata'].append(entry)
                except Exception as exc:result['errors'].append({'path':rel,'error':type(exc).__name__})
    print('M09_DEDUP_RESULT '+json.dumps(result),flush=True)
m09_inventory()
