"""No downloaded models required: budgets, leakage and causal context contracts."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from segmenter.step_encoder import pack_step
from segmenter.encoder_data import step_text
from segmenter.schema import Step


class Tokenizer:
    def encode(self,text,**kwargs):
        return list(text.encode('utf-8'))
    def num_special_tokens_to_add(self,pair):
        return 2
    def build_inputs_with_special_tokens(self,ids):
        return [1000]+ids+[1001]


def test_budget_and_leakage():
    row = step_text(Step(1,tool='run',action='pytest '*300,observation='long output\n'*3000,thought='secret future'))
    for policy in ('headtail','keylines'):
        ids,audit=pack_step(row,Tokenizer(),policy)
        assert len(ids)<=256 and audit['action_truncated'] and audit['observation_truncated']
        poisoned=dict(row,gold_stage=5,incorrect=True,solved=False,thought='future')
        assert pack_step(poisoned,Tokenizer(),policy)==(ids,audit)
    assert 'thought' not in row
    short=step_text(Step(2,tool='read',action='',observation=''))
    assert len(pack_step(short,Tokenizer())[0])<=256


def test_causal_history():
    import json
    import tempfile
    import numpy as np
    from segmenter.train_encoder_boundary import load_data
    with tempfile.TemporaryDirectory() as root:
        p=Path(root)
        def save(name,value):
            (p/name).write_text(json.dumps(value),encoding='utf-8')
        save('records.json',[dict(traj_id='a',ids=[1,2,3],fold=0),dict(traj_id='b',ids=[1,2],fold=1)])
        save('labels.json',{'a':{'boundaries':[2]},'b':{'boundaries':[]}})
        rows=[dict(traj_id=t,rows=[dict(idx=i,features={'f':i}) for i in ids]) for t,ids in [('a',[1,2,3]),('b',[1,2])]]
        (p/'features.jsonl').write_text('\n'.join(map(json.dumps,rows)),encoding='utf-8')
        save('index.json',[dict(traj_id=t,idx=i) for t,ids in [('a',[1,2,3]),('b',[1,2])] for i in ids])
        save('packed_texts.json',['a1','a2','future','b1','b2'])
        vectors=np.arange(10,dtype=np.float32).reshape(5,2)
        np.save(p/'embeddings.npy',vectors)
        before=load_data(p,p)
        # Change only a future step. Earlier a2 and independent b2 must not change.
        vectors[2]=999
        np.save(p/'embeddings.npy',vectors)
        after=load_data(p,p)
        assert np.array_equal(before[3][0],after[3][0])
        assert np.array_equal(before[3][2],after[3][2])
        assert not np.array_equal(before[3][1],after[3][1])
        assert before[6][0]=='a2 a1' and before[6][2]=='b2 b1'
        assert before[3][0,-4:].tolist()==[1.,1.,0.,0.]


def test_bootstrap_timeliness():
    from segmenter.train_boundary import bootstrap_difference
    records=[{'traj_id':'a','group':'g'}]
    labels={'a':{'boundaries':[3]}}
    late={'a':[dict(boundary_step_id=4,detected_at_step_id=4)]}
    baseline={'a':[]}
    assert bootstrap_difference(records,late,baseline,labels)['ci95']==[0.,0.]
    assert bootstrap_difference(records,late,baseline,labels,delay=None)['ci95']==[100.,100.]


if __name__=='__main__':
    test_budget_and_leakage()
    test_causal_history()
    test_bootstrap_timeliness()
    print('encoder contract tests passed')
