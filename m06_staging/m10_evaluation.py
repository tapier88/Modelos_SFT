"""Fixed development evaluation for M06_FV_JOINT_RETURN_V1; no model fitting."""
def m10_evaluate(out,root,protocol,predictions,references,truth,last,index,curves,timings):
    import datetime,hashlib,io,json,math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    folder=out/'evaluation'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    def save_json(name,value):
        with (folder/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    def stats(p,y=truth,l=last):
        err=p-y;actual=y-l[:,None];forecast=p-l[:,None];valid=actual!=0
        return {'MAE':float(abs(err).mean()),'RMSE':float(np.sqrt((err**2).mean())),
                'bias':float(err.mean()),'direction_error':float((np.sign(actual[valid])!=np.sign(forecast[valid])).mean()),
                'direction_accuracy':float((np.sign(actual[valid])==np.sign(forecast[valid])).mean()),
                'magnitude_MAE':float(abs(abs(forecast)-abs(actual)).mean()),
                'MAE_by_horizon':abs(err).mean(0).tolist()}
    refs=dict(references)
    # Reuse audited greedy baseline only when its checksum is available; never recompute an old experiment.
    baseline_paths=[root/'metricas/Kronos_base.parquet',root/'fase_I/metricas/Kronos_base.parquet']
    baseline_status='UNAVAILABLE_UNVERIFIED'
    for path in baseline_paths:
        if not path.exists():continue
        for manifest_path in (root/'checksums.json',root/'fase_I/checksums.json'):
            if not manifest_path.exists():continue
            manifest=json.loads(manifest_path.read_text())
            try:relative=str(path.relative_to(manifest_path.parent))
            except ValueError:continue
            entry=manifest.get(relative)
            if entry is None:continue
            expected=entry.get('sha256') if isinstance(entry,dict) else entry
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,'Greedy reference hash mismatch'
            frame=pd.read_parquet(path)
            actual=frame.pivot(index='window',columns='horizon',values='truth').sort_index().to_numpy()
            assert np.array_equal(actual,truth)
            refs['Kronos_base']=frame.pivot(index='window',columns='horizon',values='prediction').sort_index().to_numpy()
            baseline_status='VERIFIED';break
        if baseline_status=='VERIFIED':break
    all_predictions={**refs,**predictions}
    table={name:stats(p) for name,p in all_predictions.items()}
    per_hour=[]
    for name,p in all_predictions.items():
        for hour in sorted(index.ny_hour.unique()):
            ids=np.flatnonzero(index.ny_hour.eq(hour))
            per_hour.append({'model':name,'hour':hour,**stats(p[ids],truth[ids],last[ids])})
    sessions=sorted(index.ny_date.unique());session_ids=[np.flatnonzero(index.ny_date.eq(s)) for s in sessions]
    assert len(sessions)==243
    rng=np.random.default_rng(271828);B=10000;block=5;S=len(sessions)
    starts=rng.integers(0,S,size=(B,math.ceil(S/block)))
    samples=((starts[:,:,None]+np.arange(block))%S).reshape(B,-1)[:,:S]
    counts=np.array([len(ids)*4 for ids in session_ids]);denominator=counts[samples].sum(1)
    paired=[];gate=True
    for seed in protocol['seeds']:
        candidate=f'joint_QV_head_s{seed}';pred=predictions[candidate]
        for reference in ('Persistencia','Head_Huber_s271830',f'control_head_only_s{seed}'):
            ref=all_predictions[reference]
            delta=abs(pred-truth)-abs(ref-truth)
            sums=np.array([delta[ids].sum() for ids in session_ids])
            distribution=sums[samples].sum(1)/denominator
            ci95=np.quantile(distribution,[.025,.975]);ci_sim=np.quantile(distribution,[.05/9/2,1-.05/9/2])
            horizon=delta.mean(0);hours=[float(delta[np.flatnonzero(index.ny_hour.eq(h))].mean()) for h in sorted(index.ny_hour.unique())]
            passed=bool(delta.mean()<0 and ci_sim[1]<0 and max(horizon)<=0 and max(hours)<=0)
            gate=gate and passed
            paired.append({'candidate':candidate,'reference':reference,'delta_MAE':float(delta.mean()),
                           'CI95':ci95.tolist(),'CI_simultaneous_Bonferroni9':ci_sim.tolist(),
                           'delta_by_horizon':horizon.tolist(),'delta_by_hour':hours,'followup_component_passed':passed})
    dispersion={}
    for arm in ('control_head_only','joint_QV_head'):
        values=[table[f'{arm}_s{s}']['MAE'] for s in protocol['seeds']]
        dispersion[arm]={'seed_MAEs':values,'mean':float(np.mean(values)),'std_ddof1':float(np.std(values,ddof=1))}
    save_json('metrics.json',table);save_json('metrics_by_NY_hour.json',per_hour);save_json('paired_bootstrap.json',paired);save_json('seed_dispersion.json',dispersion)
    pd.DataFrame([{'model':name,**{k:v for k,v in metrics.items() if k!='MAE_by_horizon'}} for name,metrics in table.items()]).to_csv(folder/'metrics.csv',index=False)
    # Six plots from measured quantities. Development points exist only at 0 and 300.
    plot_files=[]
    def finish(name):
        fig=plt.gcf();fig.tight_layout();fig.savefig(folder/name,dpi=150,bbox_inches='tight');display(fig);plt.close(fig);plot_files.append(name)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for arm,ax in zip(('control_head_only','joint_QV_head'),axes):
        for seed in protocol['seeds']:
            name=f'{arm}_s{seed}';c=curves[name]
            ax.plot([x['step'] for x in c],[x['loss'] for x in c],alpha=.75,label=f'train {seed}')
            run=out/arm/f'seed_{seed}'
            dev0=json.loads((run/'step0_metrics.json').read_text())['loss'];dev300=json.loads((run/'final_metrics.json').read_text())['loss']
            ax.scatter([0,300],[dev0,dev300],marker='x',s=45,label=f'dev endpoints {seed}')
        ax.set(title=arm,xlabel='Optimizer step',ylabel='Scaled-return Huber');ax.legend(fontsize=6)
    finish('01_training_validation.png')
    plt.figure(figsize=(9,4));names=['Persistencia','Head_Huber_s271830',*predictions]
    plt.bar(range(len(names)),[table[n]['MAE'] for n in names]);plt.xticks(range(len(names)),names,rotation=45,ha='right',fontsize=7);plt.ylabel('MAE price');finish('02_MAE_architectures.png')
    plt.figure(figsize=(9,4))
    for name in names:plt.plot(range(1,5),table[name]['MAE_by_horizon'],marker='o',label=name)
    plt.xticks(range(1,5));plt.xlabel('Horizon (15 min)');plt.ylabel('MAE price');plt.legend(fontsize=6);finish('03_MAE_horizons.png')
    plt.figure(figsize=(11,4));x=np.arange(40)
    plt.plot(x,truth[:40,3],color='black',label='True H4 close')
    for name in ('Persistencia','control_head_only_s271828','joint_QV_head_s271828'):plt.plot(x,all_predictions[name][:40,3],label=name,alpha=.8)
    plt.xlabel('First 40 development decision windows (fixed slice)');plt.ylabel('XAUUSD price');plt.legend(fontsize=7);finish('04_prediction_vs_actual.png')
    plt.figure(figsize=(9,4))
    for name in predictions:plt.hist((predictions[name]-truth).ravel(),bins=70,histtype='step',density=True,label=name)
    plt.xlabel('Prediction minus true price');plt.ylabel('Density');plt.legend(fontsize=6);finish('05_error_distribution.png')
    fig,ax=plt.subplots(figsize=(10,6))
    for i,row in enumerate(paired):
        lo,hi=row['CI_simultaneous_Bonferroni9'];ax.plot([lo,hi],[i,i],color='tab:blue');ax.scatter(row['delta_MAE'],i,color='black',s=15)
    ax.axvline(0,color='red',linestyle='--');ax.set_yticks(range(9),[r['candidate']+' - '+r['reference'] for r in paired],fontsize=7);ax.set_xlabel('Paired delta MAE; simultaneous intervals across 9 contrasts');finish('06_paired_intervals.png')
    conclusion={'candidate_followup_gate':bool(gate),'independent_holdout':False,
                'superiority_claim':'No independent predictive superiority claim: development 2020 was reused for prior selection.',
                'bootstrap':protocol['evaluation']['bootstrap'],'baseline_status':baseline_status}
    save_json('conclusion.json',conclusion)
    report='# Phase V: Joint LoRA + Huber\n\nDevelopment 2020 only; no test access.\n\n'
    report+='|Model|MAE|RMSE|Bias|Direction error|\n|---|---:|---:|---:|---:|\n'
    for name,m in table.items():report+=f"|{name}|{m['MAE']:.9f}|{m['RMSE']:.9f}|{m['bias']:.9f}|{m['direction_error']:.6f}|\n"
    report+='\nFollow-up gate: '+str(gate)+'. '+conclusion['superiority_claim']+'\n'
    report+='\nPaired bootstrap: circular blocks of five NY sessions, 10,000 replicates, simultaneous Bonferroni bounds for nine fixed contrasts.\n'
    with (folder/'report.md').open('x') as f:f.write(report)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
    save_json('checksums.json',manifest)
    display(pd.DataFrame([{'model':n,'MAE':m['MAE'],'RMSE':m['RMSE']} for n,m in table.items()]))
    return {'directory':str(folder),'metrics':table,'paired_comparisons':paired,'seed_dispersion':dispersion,
            'followup_gate':bool(gate),'figures':plot_files,'manifest_sha256':hashlib.sha256((folder/'checksums.json').read_bytes()).hexdigest(),'baseline_status':baseline_status}
