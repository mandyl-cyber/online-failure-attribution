import json
import unittest
from pathlib import Path
from segmenter import stage_claim_summary as core
from segmenter import stage_claim_facts as v1
from segmenter import stage_claim_facts_v2 as v2


def source(action,obs):
    return dict(evidence=[dict(record_id=f'r:1:{f}',source_step_id=1,field=f,text=t)
                          for f,t in [('action',action),('observation',obs)]])


class SidecarTests(unittest.TestCase):
    def test_inner_failure_outer_zero(self):
        src=source('python -m pytest','ModuleNotFoundError: missing attr\nEXIT:1\n[Command finished with exit code 0]')
        fact=v2.extract(src)[0]
        self.assertEqual(fact['type'],'verification_failed')
        self.assertEqual((fact['outer_exit_code'],fact['reported_inner_exit_code']),(0,1))
        self.assertNotEqual(fact['verification_result'],'pass_in_shown_scope')
        self.assertEqual(v1.extract(src)[0]['type'],'verification_exit_zero')

    def test_failure_beats_positive_text(self):
        fact=v2.extract(source('pytest -q','2 passed\nEXIT:1\n[Command finished with exit code 0]'))[0]
        self.assertEqual(fact['type'],'mixed_outcomes')

    def test_clean_scoped_pass_and_path(self):
        fact=v2.extract(source('python tests/runtests.py tests',"Testing against Django installed in '/testbed/django'\nRan 2 tests\nOK\nEXIT:0\n[Command finished with exit code 0]"))[0]
        self.assertEqual(fact['verification_result'],'pass_in_shown_scope')
        self.assertEqual(fact['reported_execution_paths'],['/testbed/django'])

    def test_source_text_is_not_executed_exception(self):
        fact=v2.extract(source('cat tests/example.txt','ValueError: sample\nEXIT:1\n[Command finished with exit code 0]'))[0]
        self.assertEqual(fact['type'],'read_only')
        self.assertEqual(fact.get('reported_errors',[]),[])

    def test_expected_caught_error_not_reported_as_failure(self):
        fact=v2.extract(source('python demo.py','Expected error: ValueError imaginary\n[Command finished with exit code 0]'))[0]
        self.assertEqual(fact['type'],'script_exit_zero')

    def test_generic_reported_error_is_not_uncaught_exception_claim(self):
        fact=v2.extract(source('node repro.js','Error: Cannot read property\n[Command finished with exit code 0]'))[0]
        self.assertEqual(fact['type'],'reported_error')
        self.assertEqual(fact['outer_exit_code'],0)

    def test_zero_tests_remains_nonpass(self):
        fact=v2.extract(source('python -m unittest','Ran 0 tests\nOK\n[Command finished with exit code 0]'))[0]
        self.assertEqual(fact['verification_result'],'zero_tests')

    def test_sidecar_replacement_preserves_evidence_and_labels(self):
        from scripts.prepare_claim_sidecar_ablation import replace_sidecar
        rows=[json.loads(x) for x in Path('pilot/claim_sft_reviewed_v2/dev.jsonl').read_text(encoding='utf-8').splitlines()]
        row=rows[0];before=json.dumps(row,sort_keys=True)
        new=replace_sidecar(row)
        a=json.loads(row['prompt'][1]['content']);b=json.loads(new['prompt'][1]['content'])
        self.assertEqual(json.dumps(row,sort_keys=True),before)
        self.assertEqual(new['completion'],row['completion'])
        self.assertEqual(new['prompt'][0],row['prompt'][0])
        self.assertEqual({k:v for k,v in a.items() if not k.startswith('observed_facts')},
                         {k:v for k,v in b.items() if not k.startswith('observed_facts')})

    def test_reported_codes_preserve_sequence(self):
        fact=v2.extract(source('pytest -q','EXIT:1\nEXIT:0\n2 passed\n[Command finished with exit code 0]'))[0]
        self.assertEqual(fact['reported_inner_exit_codes'],[1,0])
        self.assertEqual(fact['type'],'mixed_outcomes')

    def test_real_training_diagnostics(self):
        path=Path('pilot/claim_sft_reviewed_v2/train.jsonl')
        rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        for index,step,kind,inner in [(80,20,'verification_failed',1),(81,23,'verification_failed',2),(79,18,'command_warning',None)]:
            q=json.loads(rows[index]['prompt'][1]['content'])
            unit=next(u for u in q['units'] if u['step_id']==step)
            src=dict(evidence=[dict(record_id=r['record_id'],source_step_id=step,field=f,text=''.join(c['text'] for c in r['chunks'])) for f in ('action','observation') for r in unit[f]])
            fact=v2.extract(src)[0]
            self.assertEqual((fact['type'],fact['reported_inner_exit_code']),(kind,inner))


if __name__=='__main__':unittest.main()
