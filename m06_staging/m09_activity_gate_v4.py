# M09_ACTIVITY_GATE_V4
def m09_activity_gate(namespace):
    import datetime, json, os, subprocess, sys, threading, time
    from pathlib import Path
    report={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'pid':os.getpid(),'samples':[],'blockers':[],'training_active':'UNKNOWN',
            'optimizer_steps':0,'test_read':False,'orders_enabled':False}
    current=threading.get_ident()
    safe_modules=('threading','queue','asyncio','selectors','concurrent.futures','zmq','tornado',
                  'ipykernel','IPython','google.colab','traitlets','logging','jupyter_client',
                  'debugpy','pydevd','_pydev','multiprocessing','prompt_toolkit','anyio')
    def process_inventory():
        rows=[]
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():continue
            try:
                comm=(entry/'comm').read_text().strip()
                if not any(x in comm.lower() for x in ('python','torchrun','accelerate','deepspeed')):continue
                args=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
                pid=int(entry.name)
                category='CURRENT_KERNEL' if pid==os.getpid() else 'UNCLASSIFIED'
                # Classify known Colab infrastructure without printing command arguments/secrets.
                for module in ('colab_kernel_launcher','google.colab._server','jupyter_http_over_ws','jupyter_server','notebook.notebookapp'):
                    if any(module in arg for arg in args):
                        if pid!=os.getpid():category='COLAB_SERVICE:'+module
                if pid!=os.getpid() and any(Path(arg).name in ('jupyter-notebook','jupyter-server') for arg in args):category='JUPYTER_SERVICE'
                entrypoint='UNKNOWN'
                args=[args[0]]+[arg for arg in args[1:] if arg not in ('-u','-B','-E','-s','-S','-I','-O','-OO')]
                if len(args)>2 and args[1]=='-m':entrypoint=args[2]
                elif len(args)>1 and not args[1].startswith('-'):entrypoint=args[1]
                elif len(args)>1 and args[1]=='-c':entrypoint='INLINE_CODE_REQUIRES_REVIEW'
                allowed_services={'/opt/google/drive/drive-filter.py','/usr/local/bin/colab-fileshim.py','debugpy.adapter','google.colab._kernel_launcher','google.colab._server'}
                if pid!=os.getpid() and entrypoint in allowed_services:category='VERIFIED_SERVICE_ENTRYPOINT'
                if pid!=os.getpid() and entrypoint.endswith('/debugpy/adapter') and entrypoint.startswith('/usr/local/lib/'):category='DEBUGPY_ADAPTER'
                maps=(entry/'maps').read_text()
                torch_cuda_maps=sorted(set(line.split()[-1] for line in maps.splitlines() if any(marker in line.lower() for marker in ('libtorch','libc10','/torch/','libcuda','libcudart'))))
                if category=='UNCLASSIFIED' and not torch_cuda_maps:category='PYTHON_WITHOUT_TORCH_OR_CUDA_MAPPINGS'
                rows.append({'pid':pid,'name':comm,'category':category,'entrypoint':entrypoint,'entrypoint_exists':Path(entrypoint).is_file() if entrypoint.startswith('/') else None,'torch_cuda_mapped_libraries':torch_cuda_maps})
            except FileNotFoundError:continue
            except PermissionError:rows.append({'pid':int(entry.name),'category':'UNREADABLE'})
        return rows
    for sample_index in range(3):
        frames=sys._current_frames();threads=[]
        for thread in threading.enumerate():
            if thread.ident==current:continue
            frame=frames.get(thread.ident);stack=[]
            while frame is not None:
                module=str(frame.f_globals.get('__name__',''))
                stack.append({'module':module,'function':frame.f_code.co_name})
                frame=frame.f_back
            unknown=[x for x in stack if not x['module'].startswith(safe_modules) and (x['module'],x['function']) not in {('pexpect.popen_spawn','_read_incoming'),('tqdm._monitor','run')}]
            training=[x for x in stack if x['module'].startswith(('torch.optim','torch.autograd')) or x['function'] in ('backward','train_step','training_step','fit')]
            threads.append({'name':thread.name,'alive':thread.is_alive(),'stack':stack,'unclassified_frames':unknown,'training_frames':training})
        del frames
        command=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)
        gpu=[]
        for line in command.stdout.splitlines():
            parts=[x.strip() for x in line.split(',')]
            if parts and parts[0].isdigit():gpu.append({'pid':int(parts[0]),'used_memory_mib':parts[1] if len(parts)>1 else None})
        processes=process_inventory()
        report['samples'].append({'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'threads':threads,'gpu_processes':gpu,'python_processes':processes})
        if any(t['training_frames'] for t in threads):report['blockers'].append('Training stack observed')
        if any(t['unclassified_frames'] for t in threads):report['blockers'].append('Unclassified background Python stack')
        if any(p['pid']!=os.getpid() for p in gpu):report['blockers'].append('Other GPU process active')
        if any(p['category'] in ('UNCLASSIFIED','UNREADABLE') for p in processes):report['blockers'].append('Unclassified Python process')
        if sample_index<2:time.sleep(2)
    session=namespace.get('AXXEL_SESSION')
    report['session_present']=bool(session)
    if not session:report['blockers'].append('AXXEL_SESSION absent')
    else:
        nets=[session['model'],session['tokenizer'],*session['heads'].values()]
        report['session_frozen']=all(not p.requires_grad and p.grad is None for net in nets for p in net.parameters())
        if not report['session_frozen']:report['blockers'].append('Production session not frozen or has gradients')
    report['blockers']=sorted(set(report['blockers']))
    if not report['blockers']:report['training_active']=False
    report['status']='PASS_CURRENT_ACTIVITY' if not report['blockers'] else 'BLOCKED'
    report['scope']='Three current samples: no active Kronos/PyTorch training in kernel stacks or other Python/GPU processes; library mappings classify otherwise unknown Python processes. Does not certify unrelated CPU frameworks or future scheduling.'
    print('M09_ACTIVITY_V4_RESULT '+json.dumps(report),flush=True)
m09_activity_gate(globals())
