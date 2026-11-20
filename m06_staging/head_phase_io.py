# Infraestructura de fase: resultados separados, sin alterar manifiesto Fases I-II.
import io
assert globals().get('M06_ENVIRONMENT_STABLE') is True
assert json.loads((m6_stable.OPS/'environment_status.json').read_text())['M06_ENVIRONMENT_STABLE'] is True
assert globals().get('M6_PHASE_II_DONE') is True
M6_HEAD_DIR=M6_DIR/'fase_III'
for sub in ['metricas','graficas','configuraciones','checkpoints']:
    (M6_HEAD_DIR/sub).mkdir(parents=True,exist_ok=True)
M6_HEAD_MANIFEST_PATH=M6_HEAD_DIR/'checksums.json'
M6_HEAD_MANIFEST=json.loads(M6_HEAD_MANIFEST_PATH.read_text()) if M6_HEAD_MANIFEST_PATH.exists() else {}

def m6_phase_verify():
    for rel,v in M6_HEAD_MANIFEST.items():
        p=M6_HEAD_DIR/rel
        assert p.is_file() and p.stat().st_size==v['bytes'] and m6_stable.sha(p)==v['sha256'], 'Fase III artifact mismatch: '+rel
m6_phase_verify()

def m6_phase_bytes(path,data):
    path=Path(path).resolve(); assert M6_HEAD_DIR.resolve() in path.parents
    rel=str(path.relative_to(M6_HEAD_DIR.resolve()))
    expected=M6_HEAD_MANIFEST.get(rel,{}).get('sha256')
    if path.exists() and expected is None: raise RuntimeError('Untracked phase artifact; inspect before overwrite: '+rel)
    manifest_hash=m6_stable.sha(M6_HEAD_MANIFEST_PATH) if M6_HEAD_MANIFEST_PATH.exists() else None
    info=m6_stable.atomic_bytes(path,data,expected_sha=expected)
    M6_HEAD_MANIFEST[rel]={'bytes':info['bytes'],'sha256':info['sha256']}
    m6_stable.atomic_json(M6_HEAD_MANIFEST_PATH,M6_HEAD_MANIFEST,expected_sha=manifest_hash)
    return info

def m6_phase_json(rel,obj):
    clean=m6_stable.json_value(obj)
    return m6_phase_bytes(M6_HEAD_DIR/rel,json.dumps(clean,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False).encode())

def m6_phase_npz(path,**values):
    buf=io.BytesIO(); np.savez_compressed(buf,**values)
    with np.load(io.BytesIO(buf.getvalue()),allow_pickle=False) as z:
        assert set(z.files)==set(values)
        for k,v in values.items(): assert np.array_equal(z[k],np.asarray(v),equal_nan=True)
    return m6_phase_bytes(path,buf.getvalue())

def m6_phase_torch(obj,path):
    buf=io.BytesIO();torch.save(obj,buf)
    return m6_phase_bytes(path,buf.getvalue())

def m6_phase_csv(frame,path,**kwargs):
    return m6_phase_bytes(path,frame.to_csv(**kwargs).encode())

def m6_phase_parquet(frame,path,**kwargs):
    buf=io.BytesIO();frame.to_parquet(buf,**kwargs)
    assert len(pd.read_parquet(io.BytesIO(buf.getvalue())))==len(frame)
    return m6_phase_bytes(path,buf.getvalue())

def m6_phase_figsave(name):
    buf=io.BytesIO();plt.tight_layout();plt.savefig(buf,format='png',dpi=150,bbox_inches='tight')
    m6_phase_bytes(M6_HEAD_DIR/'graficas'/name,buf.getvalue());plt.show();plt.close()

def m6_phase_restore():
    table=pd.read_csv(M6_HEAD_DIR/'metricas/cabezas_resultados_2020.csv')
    assert len(table)==8
    for row in table.to_dict(orient='records'):
        frame=pd.read_parquet(M6_HEAD_DIR/'metricas'/(row['model']+'.parquet'))
        p=frame.pivot(index='window',columns='horizon',values='prediction').sort_index().to_numpy()
        assert p.shape==(950,4) and abs(m6_metrics(p)['MAE']-row['MAE'])<1e-10
        M6_PRED[row['model']]=p
    M6_CANDIDATE_META.update(json.loads((M6_HEAD_DIR/'configuraciones/candidatos.json').read_text()))
    return table
