from copy import deepcopy
import unittest
from pathlib import Path

from segmenter import stage_claim_summary as core
from scripts.prepare_claim_evidence_standard import select_ids,validate_target,build


def source_row():
    text='prefix TARGET suffix'
    evidence=[dict(record_id='o',source_step_id=1,field='observation',text=text,
                   content_sha256=core.digest(text.encode()),availability_seq=1)]
    return core.build(dict(job_id='j',run_id='r',stage=dict(stage_id=1,start_step_id=1,end_step_id=1),
                           detected_at_seq=1,evidence=evidence,incomplete=False,task_context=None),chunk_chars=10)


class EvidenceStandardTests(unittest.TestCase):
    def test_anchor_crosses_chunks_without_changing_original_offsets(self):
        row=source_row()
        before=deepcopy(row)
        ids=select_ids(row,list(row['evidence_map']),1,'observation',['TARGET'])
        self.assertEqual(len(ids),2)
        self.assertEqual(row,before)
        refs=[row['evidence_map'][eid] for eid in ids]
        original=row['source_snapshot']['evidence'][0]['text']
        self.assertTrue(all(original[r['start']:r['end']]==r['quote'] for r in refs))

    def test_hidden_anchor_cannot_be_added_to_reference(self):
        row=source_row()
        with self.assertRaisesRegex(ValueError,'outside the visible page'):
            select_ids(row,[next(iter(row['evidence_map']))],1,'observation',['suffix'])

    def test_reference_cannot_cite_a_known_but_invisible_id(self):
        row=source_row()
        ids=list(row['evidence_map'])
        response=dict(claims=[dict(text='Recorded text.',source='tool_observation',evidence_ids=[ids[-1]])])
        with self.assertRaisesRegex(ValueError,'not visible'):
            validate_target(response,row,ids[:1])

    @unittest.skipUnless(Path('neural_ledger_acceptance_v4/config.json').exists(),'Requires local frozen v4 return')
    def test_real_candidates_merge_corrections_preserve_controls_and_remain_development_only(self):
        cases,rows,summary=build(Path('neural_ledger_acceptance_v4'))
        self.assertEqual((len(cases),len(rows)),(10,11))
        self.assertFalse(summary['training_eligible'])
        self.assertFalse(summary['training_started'])
        self.assertEqual(len({r['source']['session'] for r in rows}),3)
        self.assertEqual(sum(r['unchanged_control'] for r in rows),3)
        self.assertTrue(all(r['usage']=='development_only' and not r['training_eligible'] for r in rows))
        merged=next(r for r in rows if (r['source']['session'],r['source']['job'],r['source']['part_index'])
                    ==('terminal','00000002',1))
        self.assertEqual(len(merged['case_ids']),2)
        self.assertEqual(len(core.strict_json(merged['completion'][0]['content'])['claims']),4)


if __name__=='__main__':
    unittest.main()
