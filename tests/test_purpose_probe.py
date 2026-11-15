import hashlib
import json
import unittest
from segmenter.purpose_probe import audited_mapping, requests, strict_mapping, token_ids
from pathlib import Path
from tempfile import TemporaryDirectory


class Tests(unittest.TestCase):
    def test_prefix_isolation(self):
        ts = [dict(trajectory_id='t',task='task',steps=[dict(idx=1,action='a'),dict(idx=2,action='SECRET')])]
        first = list(requests(ts))[0][1]
        self.assertNotIn('SECRET', json.dumps(first))
        self.assertEqual(first['current_step_id'], 1)

    def test_token_return_variants(self):
        class Tokenizer:
            def __init__(self, value): self.value = value
            def apply_chat_template(self, *args, **kwargs): return self.value
        for value in ([1,2,3], {'input_ids':[1,2,3], 'attention_mask':[1,1,1]}, {'input_ids':[[1,2,3]]}):
            self.assertEqual(token_ids(Tokenizer(value), []), [1,2,3])

    def test_mapping_rejects_changed_content(self):
        s = dict(idx=1,action='a',observation='b')
        digest = hashlib.sha256(json.dumps(['a','b'],ensure_ascii=False).encode()).hexdigest()
        r = dict(content_sha256=digest,ids=[1])
        t = dict(trajectory_id='t',steps=[s])
        self.assertEqual(strict_mapping([t],[r])[0]['status'], 'exact')
        s['observation']='different'
        self.assertEqual(strict_mapping([t],[r])[0]['status'], 'unresolved')

    def test_features_causal_and_missing_retained(self):
        from segmenter.evaluate_purpose_probe import feature_rows
        t=dict(trajectory_id='t',steps=[dict(idx=1,tool='run',action='ls',observation='a'),
                                      dict(idx=2,tool='run',action='pwd',observation='b')])
        r={'t_s00001':dict(status='scored',purpose='ACQUIRE',scores=[1,0,0,0,0,0,0]),
           't_s00002':dict(status='context_overflow',purpose=None,scores=None)}
        rows=feature_rows(t,r)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[1]['purpose']['purpose_lag0:available'],0)
        r['t_s00002']=dict(status='scored',purpose='CHECK',scores=[0,0,0,0,1,0,0])
        self.assertEqual(feature_rows(t,r)[0],rows[0])

    def test_packed_mapping_requires_all_actions_and_unique_source(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            action='uniqueactionfingerprint'
            observation='uniqueobservationfingerprint'
            steps=[dict(idx=i,action=action+str(i),observation=observation+str(i)) for i in range(1,9)]
            ids=list(range(1,9))
            t=dict(trajectory_id='T01',steps=steps)
            records=[dict(traj_id='original',source='SWE',ids=ids,content_sha256='different')]
            def write(texts):
                (root/'index.json').write_text(json.dumps([dict(traj_id='original',idx=i) for i in ids]))
                (root/'packed_texts.json').write_text(json.dumps(texts))
            write([f'{action}{i} {observation}{i}' for i in ids])
            self.assertEqual(audited_mapping([t],records,root)[0]['status'],'packed_step_verified')
            write(['unrelated text']+[f'{action}{i} {observation}{i}' for i in ids[1:]])
            self.assertEqual(audited_mapping([t],records,root)[0]['status'],'unresolved')

    def test_human_audit_rejects_unfilled_labels(self):
        from segmenter.purpose_human_audit import load_labels
        with TemporaryDirectory() as directory:
            path=Path(directory)/'labels.json'
            path.write_text(json.dumps(dict(annotator='A',labels={'r':None})))
            with self.assertRaisesRegex(ValueError,'Missing/invalid'):
                load_labels(path,['r'],'A')

    def test_assisted_report_requires_explicit_review(self):
        from segmenter.ledger import encoded, sha
        from segmenter.purpose_human_audit import score_assisted
        with TemporaryDirectory() as directory:
            root=Path(directory);probe=root/'probe';out=root/'audit'
            (probe/'responses').mkdir(parents=True);out.mkdir()
            rid='T01_s00001';packet=dict(trajectory_id='T01',task='task',current_step_id=1,
                                         steps=[dict(idx=1,action='inspect',observation='found')])
            packets=[dict(request_id=rid,source='SWE',packet=packet)]
            (out/'blind_packets.json').write_text(json.dumps(packets))
            (probe/'manifest.json').write_text(json.dumps(dict(identity='probe-v1')))
            (probe/'responses'/f'{rid}.json').write_text(json.dumps(dict(
                status='scored',input_sha256=sha(encoded(packet)),purpose='ACQUIRE')))
            (out/'selection.json').write_text(json.dumps(dict(
                blind_packets_sha256=sha((out/'blind_packets.json').read_bytes()),probe_identity='probe-v1')))
            draft=dict(suggestions={rid:dict(label='ACQUIRE')})
            (out/'ai_draft.json').write_text(json.dumps(draft))
            review=dict(reviewer_id='human-1',draft_sha256=sha((out/'ai_draft.json').read_bytes()),
                        labels={rid:dict(suggested_label='ACQUIRE',final_label='ACQUIRE',reviewed=False)})
            (out/'ai_review.json').write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError,'Unreviewed'):
                score_assisted(probe,out)
            self.assertFalse((out/'assisted_agreement_report.json').exists())
            review['labels'][rid]['reviewed']=True
            (out/'ai_review.json').write_text(json.dumps(review))
            score_assisted(probe,out)
            report=json.loads((out/'assisted_agreement_report.json').read_text())
            self.assertEqual(report['overall']['qwen_agrees'],1)
            self.assertIn('not independent human gold',report['reference_provenance'])


if __name__ == '__main__': unittest.main()
