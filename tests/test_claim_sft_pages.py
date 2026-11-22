from copy import deepcopy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import tempfile

from segmenter.claim_sft_backend import messages_for, partition_paged_requests, convert_partial
from segmenter import stage_claim_summary as core
from segmenter.claim_evidence_review import result_scope_issues, review
from test_claim_evidence_review import row_for
from segmenter.claim_sft_backend import ClaimSFTBackend, convert_response
from segmenter.claim_extraction import StageProcessor
from segmenter.reviewed_claim_processor import ReviewedClaimProcessor
from segmenter.ledger import atomic


class PageTests(unittest.TestCase):
    def request(self):
        source=deepcopy(row_for('python work.py','A'*1400+'\n'+'B'*1400+'\n'+'C'*1400)['source_snapshot'])
        source['_claim_policy']='v4'
        return source

    def test_pages_preserve_every_original_chunk_and_offset(self):
        request=self.request()
        def fits(child):
            packet=json.loads(messages_for(child)[1][1]['content'])
            return sum(len(c['text']) for u in packet['units'] for f in ('action','observation')
                       for r in u[f] for c in r['chunks'])<=1500
        parts,gaps=partition_paged_requests(request,fits)
        original=messages_for(request)[0]
        self.assertEqual(gaps,[])
        self.assertGreater(len(parts),1)
        ids=[i for p in parts for i in p['_claim_page_ids']]
        self.assertEqual(ids,list(original['evidence_map']))
        for p in parts:
            row,messages=messages_for(p)
            core.verify_row(row)
            self.assertEqual(row['evidence_map'],original['evidence_map'])
            packet=json.loads(messages[1]['content'])
            self.assertEqual(packet['observed_facts'],[])
            shown=[c['id'] for u in packet['units'] for f in ('action','observation') for r in u[f] for c in r['chunks']]
            self.assertEqual(shown,p['_claim_page_ids'])
            self.assertEqual(packet['input_cutoff_seq'],request['detected_at_seq'])

    def test_known_but_unseen_chunk_is_rejected(self):
        request=self.request()
        row,_=messages_for(request)
        observed=[i for i,r in row['evidence_map'].items() if r['field']=='observation']
        response=dict(claims=[dict(text='Reported output.',source='tool_observation',evidence_ids=[observed[1]])])
        accepted,rejected=convert_partial(response,row,allowed_ids=[observed[0]])
        self.assertEqual(accepted['claims'],[])
        self.assertIn('not visible',rejected[0]['error'])

    def test_minimum_page_failure_is_an_explicit_gap(self):
        parts,gaps=partition_paged_requests(self.request(),lambda _:False)
        self.assertFalse(parts)
        original=messages_for(self.request())[0]['evidence_map']
        self.assertEqual([i for g in gaps for i in g['evidence_ids']],list(original))
        self.assertTrue(all(g['step_id']==1 for g in gaps))

    def test_download_does_not_support_install_completion(self):
        self.assertTrue(result_scope_issues('The command installs pandas.', 'Downloading pandas 40%'))
        self.assertFalse(result_scope_issues('The command installed pandas.', 'Successfully installed pandas-2.3.3'))
        self.assertFalse(result_scope_issues('pandas was not installed.', 'error: failed'))
        self.assertTrue(result_scope_issues('The source was viewed.', '',has_observation=False))
        self.assertFalse(result_scope_issues('The agent requested a view.', '',has_observation=False))

    def test_guard_is_opt_in_and_does_not_rewrite_rejected_claim(self):
        row=row_for('pip install pandas','Downloading pandas 40%')
        claim=dict(text='The command installs pandas.',source='tool_observation',evidence_ids=['S1.observation.1.1'])
        self.assertEqual(len(convert_partial(dict(claims=[claim]),row)[0]['claims']),1)
        accepted,rejected=convert_partial(dict(claims=[claim]),row,enforce_result_scope=True)
        self.assertFalse(accepted['claims'])
        self.assertEqual(rejected[0]['raw_claim'],claim)

    def test_failed_setup_is_retained_without_generated_claims(self):
        text='error: externally-managed-environment\nThe virtual environment was not created successfully because ensurepip is not\navailable. Install python3-venv.'
        row=row_for('python3 -m venv venv',text)
        result=review(dict(claims=[]),row)
        self.assertEqual({s['kind'] for s in result['observation_signals']},
                         {'externally_managed_environment','venv_creation_failed'})
        for signal in result['observation_signals']:
            self.assertEqual(text[signal['start']:signal['end']],signal['quote'])
        self.assertEqual(len(result['uncited_signal_ids']),2)
        listing=row_for('cat fixture.txt',text)
        self.assertFalse(review(dict(claims=[]),listing)['observation_signals'])

    def test_page_claim_offsets_survive_native_publication_and_recovery(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                packet=json.loads(messages[1]['content'])
                size=sum(len(c['text']) for u in packet['units'] for f in ('action','observation')
                         for r in u[f] for c in r['chunks'])
                return [1]*size
        backend=ClaimSFTBackend.__new__(ClaimSFTBackend)
        backend.identity='mocked-v4-page-publication'
        backend.tokenizer=Tokenizer()
        backend.max_length=1501
        backend.max_new_tokens=1
        calls=[]
        def generate(child):
            calls.append(child)
            row,_=messages_for(child)
            visible=child.get('_claim_page_ids',row['evidence_map'])
            observed=[i for i in visible if row['evidence_map'][i]['field']=='observation']
            backend.part_rejections=[]
            return convert_response(dict(claims=[dict(text='The page contains recorded output.',
                source='tool_observation',evidence_ids=[observed[-1]])] if observed else []),row)
        backend._generate=generate
        job=self.request()
        job.pop('_claim_policy')
        job['candidates']=[]
        for evidence in job['evidence']:
            evidence['run_id']=job['run_id']
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules',{'transformers':SimpleNamespace(AutoTokenizer=None)}):
            root=Path(directory)
            atomic(root/'job.json',job)
            processor=ReviewedClaimProcessor(StageProcessor(backend))
            batch=processor(root/'job.json',root/'claim_batches')
            self.assertEqual(batch['status'],'ready',batch.get('extraction_error',batch.get('extraction_details')))
            self.assertTrue(batch['extraction_details']['input_complete'])
            self.assertTrue(any(ref['start']>0 for c in batch['claims'] for ref in c['evidence_refs']))
            original={e['record_id']:e for e in job['evidence']}
            for claim in batch['claims']:
                for ref in claim['evidence_refs']:
                    self.assertEqual(original[ref['evidence_id']]['text'][ref['start']:ref['end']],ref['quote'])
            count=len(calls)
            before=(root/'claim_batches/job.json').read_bytes()
            processor(root/'job.json',root/'claim_batches')
            self.assertEqual(len(calls),count)
            self.assertEqual((root/'claim_batches/job.json').read_bytes(),before)


if __name__=='__main__':
    unittest.main()
