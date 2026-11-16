import unittest
from segmenter.relation_features import extend_history, relation_features
from segmenter.relation_evidence import RelationEvidence


class FeatureTests(unittest.TestCase):
    def test_long_history_only_uses_prior_rows(self):
        rows=[{'idx':i,'features':{'lag0:tool':str(i)},'text':''} for i in range(12)]
        result=extend_history(rows,9)
        self.assertEqual(result['features']['lag8:tool'],'1')
        rows[10]['features']['lag0:tool']='future'
        self.assertEqual(result,extend_history(rows,9))
        self.assertEqual(rows[9]['features'],{'lag0:tool':'9'})

    def test_relation_counts_exclude_old_failure(self):
        builder=RelationEvidence('')
        for i in range(1,11):
            e=builder.feed(dict(idx=i,tool='run',tool_class='EXEC',action=f'python x{i}.py',
                                observation='',exit_code=1 if i==1 else 0))
        f=relation_features(e)
        self.assertEqual(f['lag0:relation:prior_failure_count'],0)
        self.assertEqual(f['lag0:relation:prior_exit_zero'],8)
        self.assertEqual(len(f),10)


if __name__=='__main__':unittest.main()
