import unittest
from copy import deepcopy
from segmenter import memory_repair_probe as r


class RepairTests(unittest.TestCase):
    def test_short_ids_round_trip_and_reject_unknown(self):
        job=dict(task='support',case=dict(session='x',prefix_step=1,assertion='passed?',questions=[]),
                 units=[dict(type='claims',claims=[dict(text='failed',evidence_ids=['abc'])])],
                 short_ids=True,limited=True)
        original=deepcopy(job)
        messages,mapping=r.prompt(job)
        self.assertEqual(job,original)
        self.assertEqual(mapping,{'E1':'abc'})
        text='{"latest_run_status":"failed","assertion_support":"not_established","evidence_ids":["E1"],"limitations":"Failed suite."}'
        self.assertEqual(r.parse(text,job,mapping)['evidence_ids'],['abc'])
        self.assertFalse(r.parse(text.replace('E1','E99'),job,mapping)['valid'])

    def test_natural_retrieval_pairs_really_add_target(self):
        jobs,labels,_=r.load()
        targets={g['job_id']:g for g in labels}
        pairs={}
        for j in jobs:
            if j['group']=='retrieval': pairs.setdefault(j['query']['query_id'],{})[j['arm']]=j
        self.assertEqual(len(pairs),32)
        for pair in pairs.values():
            before=pair['claims_only_for_target']; after=pair['target_raw_restored']
            target=targets[before['job_id']]['record_id']
            raw=lambda j:{x['evidence_id'] for u in j['units'] for x in u.get('records',[])}
            self.assertNotIn(target,raw(before)); self.assertIn(target,raw(after))
            self.assertLessEqual(before['query']['target_step'],before['query']['prefix_step']-2)
            self.assertEqual(after['units'][:-1],before['units'])

    def test_source_variants_keep_original_raw_and_basis(self):
        jobs,_,_=r.load(); variants=[j for j in jobs if j['group']=='provenance']
        self.assertEqual(len(variants),3)
        raw=lambda j:[u for u in j['units'] if u.get('type')=='raw']
        self.assertTrue(all(raw(j)==raw(variants[0]) for j in variants))
        for j in variants:
            claim=[c for u in j['units'] for c in u.get('claims',[]) if c.get('source_steps')==[15]][0]
            self.assertEqual(claim['basis'],'agent_assertion')


if __name__=='__main__': unittest.main()
