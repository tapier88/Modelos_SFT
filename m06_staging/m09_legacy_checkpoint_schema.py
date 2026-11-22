# M09_LEGACY_SCHEMA_V1
def inspect_legacy():
    import datetime,hashlib,json
    from pathlib import Path
    import torch
    root=Path('/content/drive/MyDrive/AXXEL_CAPITAL/Kronos_SFT_verificado_20261107/ejecuciones/20261112T022707Z_55c83b')
    result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoints':[],'errors':[],'test_read':False,'optimizer_steps':0}
    for tf in ('M15','M5'):
        for name in ('best.pt','last.pt'):
            p=root/'resultados'/tf/name
            try:
                h=hashlib.sha256()
                with p.open('rb') as f:
                    for b in iter(lambda:f.read(1048576),b''):h.update(b)
                packet=torch.load(p,map_location='cpu',weights_only=True)
                schemas={}
                for key,value in packet.items():
                    if isinstance(value,dict):
                        tensors={k:list(v.shape) for k,v in value.items() if isinstance(v,torch.Tensor)}
                        if tensors:schemas[key]=tensors
                result['checkpoints'].append({'path':str(p),'sha256':h.hexdigest(),'keys':sorted(packet),'tensor_schemas':schemas})
                del packet
            except Exception as exc:result['errors'].append({'path':str(p),'error':type(exc).__name__+': '+str(exc)})
    print('M09_LEGACY_RESULT '+json.dumps(result),flush=True)
inspect_legacy()
