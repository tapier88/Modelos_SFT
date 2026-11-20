"""Restore completed later phases, without training or forecasting. Explicit M06 scope."""
def restore_completed(ns):
    import json,hashlib
    from pathlib import Path
    import numpy as np,pandas as pd,torch
    root=ns['M6_DIR'];stable=ns['m6_stable'];evidence={}
    for phase,expected,table_name in [('fase_III',8,'cabezas_resultados_2020.csv'),('fase_IV',6,'pilotos_resultados_2020.csv')]:
        folder=root/phase;status=folder/'phase_status.json'
        if not status.exists():
            evidence[phase]={'complete':False,'reason':'not completed; resume its checkpointed cell'};continue
        manifest=json.loads((folder/'checksums.json').read_text())
        for rel,v in manifest.items():
            p=folder/rel
            assert p.is_file() and p.stat().st_size==v['bytes'] and stable.sha(p)==v['sha256'], str(p)
        assert json.loads(status.read_text())['complete'] is True
        table=pd.read_csv(folder/'metricas'/table_name);assert len(table)==expected
        for row in table.to_dict(orient='records'):
            f=pd.read_parquet(folder/'metricas'/(row['model']+'.parquet'))
            assert len(f)==3800 and pd.to_datetime(f.decision_utc,utc=True).dt.year.eq(2020).all()
            assert f[['window','horizon']].duplicated().sum()==0
            p=f.pivot(index='window',columns='horizon',values='prediction').sort_index().to_numpy()
            y=f.pivot(index='window',columns='horizon',values='truth').sort_index().to_numpy()
            assert p.shape==(950,4) and np.isfinite(p).all() and np.array_equal(y,ns['M6_Y'])
            assert abs(ns['m6_metrics'](p)['MAE']-row['MAE'])<1e-10
            ns['M6_PRED'][row['model']]=p
        ns['M6_CANDIDATE_META'].update(json.loads((folder/'configuraciones/candidatos.json').read_text()))
        flag='M6_PHASE_III_DONE' if phase=='fase_III' else 'M6_PHASE_IV_DONE'
        ns[flag]=True;ns['M6_HEAD_TABLE' if phase=='fase_III' else 'M6_IV_TABLE']=table
        evidence[phase]={'complete':True,'files_verified':len(manifest),'candidates':len(table),'manifest_sha256':stable.sha(folder/'checksums.json')}
        if phase=='fase_III':
            from sklearn.preprocessing import StandardScaler
            from torch import nn
            from torch.nn import functional as F
            checkpoints=list((folder/'checkpoints/fase_III').glob('*'))
            assert len(checkpoints)==1
            fdir=checkpoints[0]
            class RestoredReturnHead(nn.Module):
                def __init__(self,prob=False):
                    super().__init__();self.probabilistic=prob;self.hidden=nn.Linear(512,64);self.output=nn.Linear(64,8 if prob else 4)
                def forward(self,x):
                    y=self.output(F.gelu(self.hidden(x)))
                    return (y[:,:4],F.softplus(y[:,4:])+1e-4) if self.probabilistic else (y,None)
            models={}
            for row in table.to_dict(orient='records'):
                name=row['model']
                if not name.startswith('Head_'):continue
                packet=torch.load(fdir/(name+'_best.pt'),map_location='cpu',weights_only=False)
                assert packet['base_hash']==ns['M6_BASE_HASH'] and packet['tokenizer_hash']==ns['M6_TOK_HASH']
                model=RestoredReturnHead(packet['loss']=='Laplace_NLL').to(ns['M6_DEVICE'])
                model.load_state_dict(packet['head']);model.requires_grad_(False).eval();models[name]=model
            with np.load(fdir/'train_only_scalers.npz',allow_pickle=False) as z:
                hs=StandardScaler();hs.mean_=z['hidden_mean'];hs.scale_=z['hidden_scale'];hs.var_=hs.scale_**2;hs.n_features_in_=512
                ns['M6_HSCALER']=hs;ns['M6_Y_SCALE']=z['return_scale']
            ns['M6_HEAD_MODELS']=models;ns['M6_FDIR']=fdir;ns['M6_HEAD_DIR']=folder
    return evidence
