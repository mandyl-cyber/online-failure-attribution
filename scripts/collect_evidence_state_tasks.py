"""Execute newly authored unittest contrasts in isolated temporary directories.

Never executes trajectory commands. These are engineering tasks, not a held-out
agent failure-detection benchmark. No GPU, API or training required.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
from segmenter.evidence_state import EvidenceState,native_fact
from segmenter.stage_claim_ablation import digest,save

TASKS=[
    ('capped_sum','return sum(xs)','return min(sum(xs),10)','self.assertEqual(solve([8,7]),10)'),
    ('positive_count','return len(xs)','return sum(x>0 for x in xs)','self.assertEqual(solve([-1,0,2]),1)'),
    ('empty_product','return 0','import math\n    return math.prod(xs)','self.assertEqual(solve([]),1)'),
    ('tail_copy','return xs','return xs[1:]','self.assertEqual(solve([1,2,3]),[2,3])'),
]


def collect():
    results=[]
    for name,bug,fixed,assertion in TASKS:
        with tempfile.TemporaryDirectory(prefix='evidence_task_') as workspace:
            root=Path(workspace)
            (root/'test_task.py').write_text('import unittest\nfrom task import solve\nclass Tests(unittest.TestCase):\n    def test_contract(self):\n        '+assertion+'\n',encoding='utf-8')
            state=EvidenceState(); runs=[]; decisions=[]
            def check(label,version,step,expected,obj=name):
                c=dict(kind='current_test_pass',object_id=obj,version=version,prefix_step=step)
                d=state.assess(c)
                decisions.append(dict(case=label,claim=c,expected=expected,decision=d,correct=d['assertion_support']==expected))
            for step,body in [(1,bug),(3,fixed),(5,bug)]:
                code='def solve(xs):\n    '+body+'\n'
                (root/'task.py').write_text(code,encoding='utf-8')
                version=digest(code.encode())
                # -B alone does not prevent reading stale pyc; this collector
                # disables bytecode writing on every process from the first run.
                command=[sys.executable,'-B','-m','unittest','-v','test_task']
                result=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=20)
                action=dict(action='run',id=step*2,args=dict(command='python -m unittest -v test_task'))
                observation=dict(observation='run',id=step*2+1,content=result.stdout+result.stderr,extras=dict(exit_code=result.returncode))
                f=native_fact(action,observation,step,[name+':A'+str(step),name+':O'+str(step)],dict(object_id=name,version=version))
                state.append(f); runs.append(dict(action=action,observation=observation,fact=f,code=code,executed_command=command))
                check('initial_failure' if step==1 else 'repaired' if step==3 else 'regression',version,step,'supported' if step==3 else 'not_established')
                if step==3:
                    check('unrelated_object',version,step,'not_established',obj='other-'+name)
                    check('unverified_new_version',version+'-changed',step,'not_established')
                    check('missing_version',None,step,'not_established')
            results.append(dict(task=name,runs=runs,decisions=decisions))
    save(Path('out/evidence_state_new_tasks_v1/results.json'),dict(tasks=results,
        cases=sum(len(t['decisions']) for t in results),correct=sum(d['correct'] for t in results for d in t['decisions']),
        scope='Four new authored, actually executed engineering tasks; trusted file-hash bindings. Not autonomous agent trajectories, online alarms, or held-out generalization.'))
    print(dict(tasks=len(results),cases=sum(len(t['decisions']) for t in results),correct=sum(d['correct'] for t in results for d in t['decisions'])))


if __name__=='__main__':collect()
