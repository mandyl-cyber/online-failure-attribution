import unittest
from copy import deepcopy
from segmenter.claim_extraction import SCHEMA
from remote_pilots.openhands_smoke.run_ledger_sensitivity import validate


class RemoteContractTests(unittest.TestCase):
    def setUp(self):
        self.request={'response_schema':SCHEMA,'evidence':[{'record_id':'e','text':'测试 ok','field':'observation'}]}
        self.response={'claims':[dict(local_id='c1',kind='verification',text='observed text',basis='tool_observation',
            objects=[],conditions=[],unknowns=[],evidence_refs=[dict(evidence_id='e',start=0,end=2,quote='测试')])]}

    def test_valid_unicode_and_empty_claims(self):
        validate(self.response,self.request);validate({'claims':[]},self.request)

    def test_bad_span_and_boolean_index(self):
        for start in (-1,True,9):
            x=deepcopy(self.response);x['claims'][0]['evidence_refs'][0]['start']=start
            with self.assertRaises(ValueError):validate(x,self.request)

    def test_extra_field_and_quote_mismatch(self):
        x=deepcopy(self.response);x['claims'][0]['extra']='bad'
        with self.assertRaises(ValueError):validate(x,self.request)
        x=deepcopy(self.response);x['claims'][0]['evidence_refs'][0]['quote']='bad'
        with self.assertRaises(ValueError):validate(x,self.request)

    def test_duplicate_and_foreign_evidence(self):
        x=deepcopy(self.response);x['claims'].append(deepcopy(x['claims'][0]))
        with self.assertRaises(ValueError):validate(x,self.request)
        x=deepcopy(self.response);x['claims'][0]['evidence_refs'][0]['evidence_id']='foreign'
        with self.assertRaises(KeyError):validate(x,self.request)

    def test_observation_basis_requires_observation(self):
        self.request['evidence'][0]['field']='action'
        with self.assertRaises(ValueError):validate(self.response,self.request)


if __name__=='__main__':unittest.main()
