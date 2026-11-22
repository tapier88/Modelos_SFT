# M09_LEGACY_STATIC_V1
def inspect_legacy():
    import datetime,hashlib,json
    from pathlib import Path
    import pickletools, zipfile
    root=Path('/content/drive/MyDrive/AXXEL_CAPITAL/Kronos_SFT_verificado_20261107/ejecuciones/20261112T022707Z_55c83b')
    result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoints':[],'errors':[],'test_read':False,'optimizer_steps':0}
    for tf in ('M15','M5'):
        for name in ('best.pt','last.pt'):
            p=root/'resultados'/tf/name
            try:
                h=hashlib.sha256()
                with p.open('rb') as f:
                    for b in iter(lambda:f.read(1048576),b''):h.update(b)
                with zipfile.ZipFile(p) as archive:
                    names=[n for n in archive.namelist() if n.endswith('/data.pkl') or n=='data.pkl']
                    assert len(names)==1
                    raw=archive.read(names[0])
                strings=[arg for op,arg,pos in pickletools.genops(raw) if op.name in ('UNICODE','BINUNICODE','SHORT_BINUNICODE','BINUNICODE8')]
                parameter_names=sorted(set(s for s in strings if s.endswith(('.weight','.bias','.lora_A','.lora_B')) or 'lora_' in s))
                assert len(parameter_names)>100
                result['checkpoints'].append({'path':str(p),'sha256':h.hexdigest(),'inspection':'pickle opcode parsing only; no deserialization','parameter_names':parameter_names,'lora_keys':[s for s in parameter_names if 'lora_' in s],'regression_head_keys':[s for s in parameter_names if s in ('hidden.weight','hidden.bias','output.weight','output.bias') or s.endswith(('.hidden.weight','.output.weight'))]})
            except Exception as exc:result['errors'].append({'path':str(p),'error':type(exc).__name__+': '+str(exc)})
    print('M09_LEGACY_STATIC_RESULT '+json.dumps(result),flush=True)
inspect_legacy()
