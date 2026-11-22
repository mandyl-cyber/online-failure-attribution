"""Expand an evidence case inventory and author prospective engineering tasks."""
from collections import Counter
from pathlib import Path
import json
import hashlib
import subprocess
import sys
import tempfile

from scripts.run_stage_claim_pipeline import read,save
from scripts.audit_neural_ledger_v3_return import require
from segmenter import stage_claim_summary as core


TASKS=[
    ('stable_unique',
     'Return the unique string values in first-appearance order without mutating the input list; handle empty input.',
     'def solve(values):\n    return sorted(set(values))\n',
     'def solve(values):\n    return list(dict.fromkeys(values))\n',
     ["self.assertEqual(solve([]), [])","self.assertEqual(solve(['b','a','b','c']), ['b','a','c'])","x=['b','a','b']; before=list(x); solve(x); self.assertEqual(x,before)"]),
    ('bounded_chunks',
     'Split a list into chunks of positive size, preserving order and the final partial chunk. Raise ValueError for size <= 0 and do not mutate input.',
     'def solve(values, size):\n    return [values[i:i+size] for i in range(0,len(values)-size+1,size)]\n',
     'def solve(values, size):\n    if size <= 0:\n        raise ValueError("positive size required")\n    return [values[i:i+size] for i in range(0,len(values),size)]\n',
     ["self.assertEqual(solve([1,2,3,4,5],2), [[1,2],[3,4],[5]])","self.assertEqual(solve([],2), [])","with self.assertRaises(ValueError): solve([1],0)","with self.assertRaises(ValueError): solve([1],-1)"]),
    ('prefix_totals',
     'Return running totals including an initial zero, so output length is input length + 1. Preserve the input, including negative values.',
     'def solve(values):\n    for i in range(1,len(values)):\n        values[i] += values[i-1]\n    return values\n',
     'def solve(values):\n    result=[0]\n    for value in values:\n        result.append(result[-1]+value)\n    return result\n',
     ["self.assertEqual(solve([]), [0])","self.assertEqual(solve([2,-1,4]), [0,2,1,5])","x=[1,2,3]; before=list(x); solve(x); self.assertEqual(x,before)"]),
    ('config_overrides',
     'Return a new dictionary with overrides applied. Ignore only override values equal to None; preserve False, 0 and empty strings. Do not mutate either input.',
     'def solve(base, overrides):\n    base.update({k:v for k,v in overrides.items() if v})\n    return base\n',
     'def solve(base, overrides):\n    result=dict(base)\n    result.update({k:v for k,v in overrides.items() if v is not None})\n    return result\n',
     ["self.assertEqual(solve({'a':1},{'a':0}), {'a':0})","self.assertEqual(solve({'a':True},{'a':False,'b':''}), {'a':False,'b':''})","self.assertEqual(solve({'a':1},{'a':None}), {'a':1})","x={'a':1}; y={'b':2}; z=solve(x,y); self.assertEqual(x,{'a':1}); self.assertEqual(y,{'b':2}); self.assertIsNot(x,z)"]),
]


def authored_tasks():
    result=[]
    for name,requirements,buggy,reference,checks in TASKS:
        tests='import unittest\nfrom task import solve\n\nclass TaskTests(unittest.TestCase):\n'
        for index,check in enumerate(checks):
            tests+=f'    def test_{index}(self):\n        '+check+'\n'
        result.append(dict(task_id='claim-prospective-v1:'+name,code_name='task.py',test_name='test_task.py',
            code=buggy,tests=tests,reference_code=reference,
            task='Fix /workspace/task.py. '+requirements+' Inspect the implementation, reproduce failures, edit the saved implementation, run python -m unittest -v and finish. Do not modify test_task.py. No third-party dependencies are needed.',
            scope='Newly authored synthetic engineering task, not a CodeTraceBench sample or semantic claim gold.'))
    return result


def preflight_tasks(tasks):
    root=Path('out/claim_prospective_preflight_v1')
    root.mkdir(parents=True,exist_ok=True)
    reports=[]
    for task in tasks:
        exits={}
        for label,code in [('buggy',task['code']),('reference',task['reference_code'])]:
            directory=Path(tempfile.mkdtemp(prefix=label+'-',dir=root))
            (directory/'task.py').write_text(code,encoding='utf-8')
            (directory/'test_task.py').write_text(task['tests'],encoding='utf-8')
            completed=subprocess.run([sys.executable,'-m','unittest','-q'],cwd=directory,capture_output=True,timeout=10)
            (directory/'tests.log').write_bytes(completed.stdout+completed.stderr)
            exits[label]=completed.returncode
        require(exits['buggy']!=0 and exits['reference']==0,'Fixture did not distinguish bug and reference: '+task['task_id'])
        reports.append(dict(task_id=task['task_id'],**exits))
    return reports


def main():
    out=Path('pilot/claim_sft_next_v1')
    out.mkdir(parents=True,exist_ok=True)
    manifest_path=Path('codetrace_bench/bench_manifest.full.jsonl')
    manifest=[json.loads(x) for x in manifest_path.read_text(encoding='utf-8').splitlines()]
    by_run={r['traj_id']:r for r in manifest}
    digest=lambda name:hashlib.sha256(name.strip().lower().encode()).hexdigest()
    selection=read(Path('pilot/swe_purpose_continuous_v1/selection.json'))
    names={}
    for selected in selection:
        name=by_run[selected['traj_id']]['task_name'].strip().lower()
        require(selected['group'] not in names or names[selected['group']]==name,'Conflicting source task group')
        names[selected['group']]=name
    source=Path('pilot/claim_sft_reviewed_v4_scopefix')
    splits={s:[json.loads(x) for x in (source/f'{s}.jsonl').read_text(encoding='utf-8').splitlines()] for s in ('train','dev')}
    train_groups={r['group'] for r in splits['train']}
    dev_groups={r['group'] for r in splits['dev']}
    require(not train_groups & dev_groups,'Train/dev overlap')
    require((train_groups|dev_groups)<=names.keys(),'Canonical task identity unresolved')
    canonical_train={digest(names[g]) for g in train_groups}
    canonical_dev={digest(names[g]) for g in dev_groups}
    require(not canonical_train & canonical_dev,'Canonical train/dev task overlap')
    smoke=read(Path('eval/online_ledger_smoke_v1/report.json'))
    mapping=[]
    for run in smoke['runs']:
        row=by_run[run['trajectory']]
        group=digest(row['task_name'])
        mapping.append(dict(session='swe' if run['source']=='SWE' else 'terminal',trajectory=run['trajectory'],
            task=row['task_name'],canonical_group=group,in_train=group in canonical_train,in_dev=group in canonical_dev,
            use='development_only',novel_to_project=False))
    mapping.append(dict(session='live_smoke',task='legacy merge_ranges synthetic smoke',canonical_group=None,
                        identity='Authored smoke definition, not a CodeTraceBench task',use='development_only',novel_to_project=False))
    facts=read(Path('pilot/claim_train_critical_coverage_v1/facts.json'))
    bank=[]
    for fact in facts:
        row=splits['train'][fact['index']]
        require(row['group']==fact['group'] and row['job_id']==fact['job_id'],'Training fact mapping changed')
        packet=core.strict_json(row['prompt'][1]['content'])
        chunks={c['id']:c['text'] for unit in packet['units'] for field in ('action','observation') for parent in unit[field] for c in parent['chunks']}
        require(all(chunks[eid]==text for eid,text in fact['evidence'].items()),'Fact evidence changed')
        bank.append(dict(case_id=fact['fact_id'],origin='existing_AI_train_fact_inventory',usage='train_reference_audit',
            group=fact['group'],canonical_task=names[fact['group']],category=fact['category'],
            fact=fact['text'],evidence=fact['evidence'],reference_job_id=row['job_id'],reviewer_kind='AI',
            adds_training_example=False))
    for case in read(Path('pilot/claim_evidence_standard_v1/cases.json')):
        bank.append(dict(case_id=case['case_id'],origin='v4_known_development_return',usage='development_only',
            group=next(m['canonical_group'] for m in mapping if m['session']==case['session']),
            category=case['family'],fact=case['reference_claim']['text'],evidence=case['evidence_refs'],
            reviewer_kind='AI',adds_training_example=False))
    tasks=authored_tasks()
    checks=preflight_tasks(tasks)
    task_root=Path('pilot/claim_prospective_tasks_v1')
    task_root.mkdir(parents=True,exist_ok=True)
    selection=[]
    for task in tasks:
        filename=task['task_id'].split(':')[-1]+'.json'
        save(task_root/filename,task)
        selection.append(dict(task_id=task['task_id'],path=filename,sha256=hashlib.sha256((task_root/filename).read_bytes()).hexdigest()))
    save(task_root/'protocol.json',dict(protocol='claim-prospective-tasks-v1',tasks=selection,
        use='prospective_synthetic_engineering_acceptance',claim_outputs_generated=False,semantic_references_frozen=False,
        rule='Collect every selected task including failed/looping agent runs. Freeze evidence-defined references before running claim extraction. Never use for tuning before the acceptance comparison.',
        limitations=['Not CodeTraceBench','New fixtures are not collected agent trajectories','No independent acceptance outcome exists yet']))
    (out/'case_bank.jsonl').write_bytes(b''.join(core.encoded(row)+b'\n' for row in bank))
    save(out/'task_provenance.json',dict(manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        mapping_rule='source group -> existing purpose selection -> trajectory ID -> full manifest task name -> lowercase SHA256',
        train_tasks=[dict(source_group=g,canonical_group=digest(names[g]),task=names[g]) for g in sorted(train_groups)],
        dev_tasks=[dict(source_group=g,canonical_group=digest(names[g]),task=names[g]) for g in sorted(dev_groups)],known_development_sessions=mapping))
    summary=dict(existing_train_rows=len(splits['train']),existing_dev_rows=len(splits['dev']),
        train_groups=len(train_groups),dev_groups=len(dev_groups),case_bank=len(bank),
        reused_train_facts=len(facts),known_development_cases=len(bank)-len(facts),
        new_training_examples=0,prospective_tasks=len(tasks),fixture_preflight=checks,
        training_started=False,independent_acceptance_collected=False,
        blockers=['Prospective native agent runs not collected','Evidence-defined acceptance references not frozen',
                  'Training bank expansion reuses existing reference facts; not additional supervision'],
        next='Collect new prospective native logs, freeze semantic references, then compare frozen 7B before any retraining.')
    save(out/'summary.json',summary)
    print(summary)


if __name__=='__main__':
    main()
