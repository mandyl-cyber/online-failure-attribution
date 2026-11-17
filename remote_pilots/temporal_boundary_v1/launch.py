"""One-command CPU training with bounded fold parallelism and epoch resume."""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))


@contextmanager
def exclusive(path):
    with path.open('a+b') as fh:
        fh.seek(0);fh.write(b'0');fh.flush();fh.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(fh.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            if os.name=='nt':
                fh.seek(0);msvcrt.locking(fh.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(fh.fileno(),fcntl.LOCK_UN)


def verify_data(data):
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    for name,info in manifest['files'].items():
        if hashlib.sha256((data/name).read_bytes()).hexdigest()!=info['sha256']:
            raise ValueError('Data checksum mismatch: '+name)
    with gzip.open(data/'records.json.gz','rt',encoding='utf-8') as fh:records=json.load(fh)
    if len(records)!=manifest['trajectories'] or sum(len(r['ids']) for r in records)!=manifest['steps']:
        raise ValueError('Manifest counts mismatch')
    seen={}
    for r in records:
        if r['group'] in seen and seen[r['group']]!=r['fold']:raise ValueError('Cross-fold group')
        seen[r['group']]=r['fold']
    return records,manifest


def execute(a):
    from segmenter.durable_input import atomic_json
    data=Path(a.data).resolve();out=Path(a.out).resolve()
    versions={}
    for name in ('torch','numpy','scipy','sklearn','joblib','yaml'):
        try:versions[name]=getattr(importlib.import_module(name),'__version__','unknown')
        except ImportError as exc:
            raise RuntimeError('Missing dependency '+name+'; install requirements.txt in the training environment') from exc
    records,manifest=verify_data(data)
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(epochs=a.epochs,threads=a.threads,seed=41,hidden=128,device='cpu',folds=list(range(5)),
        input_files=manifest['files'],versions=versions,python=sys.version,
        code={name:hashlib.sha256((REPO/'segmenter'/name).read_bytes()).hexdigest() for name in
              ('train_temporal.py','temporal_boundary.py','boundary_features.py','boundary_model.py','eval_online.py','eval_segmenter.py')})
    with exclusive(out/'launcher.lock'):
        protocol_path=out/'launch_protocol.json'
        if protocol_path.exists():
            if json.loads(protocol_path.read_text(encoding='utf-8'))!=protocol:
                raise ValueError('Training config/code/environment changed; use a new --out')
        elif any(p.name!='launcher.lock' for p in out.iterdir()):
            raise ValueError('Nonempty output without protocol')
        atomic_json(protocol_path,protocol)
        if a.preflight:
            print(json.dumps(dict(status='preflight_pass',trajectories=len(records),steps=manifest['steps'],versions=versions),indent=2));return
        pending=list(range(5));running={};done=[];started=time.time()
        env=os.environ.copy()
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]=str(a.threads)
        env['PYTHONUNBUFFERED']='1'
        try:
            while pending or running:
                while pending and len(running)<a.workers:
                    fold=pending.pop(0);folder=out/f'run_fold{fold}'
                    command=[sys.executable,'-m','segmenter.train_temporal','--data',str(data),'--out',str(folder),
                        '--fold',str(fold),'--epochs',str(a.epochs),'--threads',str(a.threads),'--device','cpu']
                    if folder.exists():command.append('--resume')
                    log=(out/f'fold{fold}.log').open('a',encoding='utf-8')
                    process=subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env=env)
                    running[fold]=(process,log)
                    print(f'fold {fold} started; log={out / ("fold"+str(fold)+".log")}',flush=True)
                for fold,(process,log) in list(running.items()):
                    result=process.poll()
                    if result is None:continue
                    log.close();del running[fold]
                    if result:raise RuntimeError(f'Fold {fold} failed ({result}); inspect fold{fold}.log and rerun the same command')
                    done.append(fold);print(f'fold {fold} complete ({len(done)}/5)',flush=True)
                atomic_json(out/'progress.json',dict(completed=done,running=list(running),waiting=pending,elapsed_seconds=time.time()-started))
                if pending or running:time.sleep(1)
        finally:
            for process,log in running.values():
                if process.poll() is None:process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                log.close()
        from segmenter.eval_online import aggregate
        labels={r['traj_id']:dict(boundaries=r['boundaries']) for r in records}
        combined={k:{} for k in ('lr','immediate','confidence')};fold_reports=[]
        for fold in range(5):
            folder=out/f'run_fold{fold}'/f'fold{fold}'
            fold_reports.append(json.loads((folder/'report.json').read_text(encoding='utf-8')))
            predictions=json.loads((folder/'predictions.json').read_text(encoding='utf-8'))
            for name in combined:
                if set(combined[name])&set(predictions[name]):raise ValueError('Duplicate test trajectory')
                combined[name].update(predictions[name])
        if any(set(v)!=set(labels) for v in combined.values()):raise ValueError('Incomplete test coverage')
        report=dict(status='complete',scope='Previously inspected SWE task groups; no pristine test claim',
            trajectories=len(records),steps=manifest['steps'],elapsed_seconds=time.time()-started,
            aggregate={k:aggregate(v,labels) for k,v in combined.items()},folds=fold_reports)
        atomic_json(out/'summary.json',report);atomic_json(out/'predictions.json',combined)
        print('SUMMARY='+str(out/'summary.json'),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default=str(Path(__file__).parent/'data'))
    ap.add_argument('--out',default=str(REPO/'out/temporal_boundary_train_v1'))
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--epochs',type=int,default=15)
    ap.add_argument('--preflight',action='store_true')
    a=ap.parse_args()
    if not 1<=a.workers<=5 or a.threads<1 or a.epochs<1:ap.error('Invalid resource configuration')
    execute(a)


if __name__=='__main__':main()
