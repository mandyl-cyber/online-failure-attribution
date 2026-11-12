import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from segmenter.confirmation import ConfirmationBuffer,confirmation_rows


def test_prefix_and_delay():
    rows=[dict(idx=i,features={'lag0:tool':str(i),'lag1:tool':str(i-1)},gold_stage=999) for i in range(1,7)]
    for delay in (0,1,2):
        all_rows=confirmation_rows(rows,delay)
        for n in range(1,len(rows)+1):
            prefix=confirmation_rows(rows[:n],delay)
            assert prefix==[x for x in all_rows if x['detected_at_step_id']<=n]
        assert all_rows[0]['boundary_step_id']==2
        assert all_rows[0]['detected_at_step_id']==2+delay
        assert len(all_rows)==5-delay
        assert 'gold_stage' not in all_rows[0]['features']
        if delay:
            assert all_rows[0]['features'][f'confirm{delay}:tool']==str(2+delay)
        changed=[dict(x) for x in rows]
        changed[-1]=dict(idx=6,features={'lag0:tool':'poison'})
        assert confirmation_rows(changed,delay)[:-1]==all_rows[:-1]
    buffer=ConfirmationBuffer(1)
    buffer.feed(rows[0])
    try:buffer.feed(rows[0])
    except ValueError:pass
    else:raise AssertionError('Duplicate accepted')


if __name__=='__main__':
    test_prefix_and_delay()
    print('confirmation prefix, delay, tail and label isolation tests passed')
