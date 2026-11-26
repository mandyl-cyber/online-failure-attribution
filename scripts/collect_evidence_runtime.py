"""Actually executed controlled host tasks for runtime wiring; not agent rollouts."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from segmenter.evidence_runtime import RuntimeEvidence
from segmenter.stage_claim_ablation import save,require
from scripts.collect_evidence_state_tasks import TASKS


def collect(out):
    require(not out.exists(),'Use a new output directory')
    require(shutil.which('git') is not None,'git required for real positive commit controls')
    summaries=[];checks=[]
    for name,bug,fixed,assertion in TASKS:
        with tempfile.TemporaryDirectory(prefix='runtime_evidence_') as work:
            root=Path(work);path=root/'task.py'
            (root/'test_task.py').write_text('import unittest\nfrom task import solve\nclass Tests(unittest.TestCase):\n    def test_contract(self):\n        '+assertion+'\n',encoding='utf-8')
            # Repository setup is restricted to this fresh temporary workspace.
            for args in (['git','init'],['git','config','user.name','Evidence acceptance'],['git','config','user.email','acceptance@example.invalid']):
                subprocess.run(args,cwd=root,check=True,capture_output=True)
            runtime=RuntimeEvidence(name,root,out/name);seq=0
            def event_id():
                nonlocal seq
                seq+=1;return seq
            def command(actual,display):
                aid=event_id();runtime.bind(aid,path)
                runtime.feed(dict(id=aid,source='agent',action='run',args=dict(command=display)))
                run=subprocess.run(actual,cwd=root,capture_output=True,text=True,timeout=30)
                rows=runtime.feed(dict(id=event_id(),source='environment',observation='run',cause=aid,
                    content=run.stdout+run.stderr,extras=dict(exit_code=run.returncode)))
                return rows[-1]['fact'],run.returncode
            def statement(text):
                aid=event_id();runtime.bind(aid,path)
                runtime.feed(dict(id=aid,source='agent',action='message',args=dict(content=text)))
                return runtime.statement(aid,text,path)
            def check(label,item,expected):
                row=runtime.decide(item);d=row['decision']
                observed=d['state'] if d['state']!='assessed' else d['assertion_support']
                checks.append(dict(task=name,case=label,expected=expected,observed=observed,correct=observed==expected))
            for phase,body in [('failure',bug),('repair',fixed),('regression',bug)]:
                path.write_text('def solve(xs):\n    '+body+'\n',encoding='utf-8')
                f,exit_code=command([sys.executable,'-B','-m','unittest','-v','test_task'],'python -m unittest -v test_task')
                require((exit_code==0)==(phase=='repair'),'Unexpected controlled task outcome')
                stage=name+'-'+phase;runtime.seal(stage,f['step'])
                item=statement('All tests passed.')
                check(phase+'_before_publication',item,'pending_publication')
                runtime.publish(stage,[f['fact_id']])
                check(phase+'_after_publication',item,'supported' if phase=='repair' else 'not_established')
            check('commit_without_tool',statement('The changes have been committed.'),'not_established')
            subprocess.run(['git','add','task.py','test_task.py'],cwd=root,check=True,capture_output=True)
            f,exit_code=command(['git','commit','-m','snapshot'],'git commit -m snapshot')
            require(exit_code==0,'Commit positive control failed')
            runtime.seal(name+'-commit',f['step']);runtime.publish(name+'-commit',[f['fact_id']])
            check('commit_with_tool',statement('The changes have been committed.'),'supported')
            check('ambiguous_prose',statement('It should probably work now.'),'pending_decomposition')
            summaries.append(runtime.close())
    result=dict(sessions=summaries,checks=checks,total_checks=len(checks),correct=sum(c['correct'] for c in checks),
        scope='Actual host execution and atomic evidence publication on four controlled authored tasks; statements are scripted probes, not autonomous agent output; stages are explicit harness callbacks, not GRU boundaries or LoRA extraction.')
    save(out/'summary.json',result)
    print({k:v for k,v in result.items() if k not in ('sessions','checks')})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('out/evidence_runtime_v1'))
    collect(p.parse_args().out)


if __name__=='__main__':main()
