"""Causality, task leakage, candidate state and delayed scoring regression tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segmenter.boundary_features import BoundaryFeatures
from segmenter.eval_online import score
from segmenter.stage_dataset import assign_folds
from segmenter.schema import Step
from segmenter.segment import Segmenter
from segmenter.rules import Rules
from segmenter.events import E, Event


def test_prefix():
    steps = [Step(i, action=f'cat f{i}', tool_class='READ', observation='ok') for i in range(1,9)]
    def run(xs):
        f = BoundaryFeatures()
        return [f.feed(s) for s in xs]
    full = run(steps)
    for n in range(1,len(steps)):
        assert run(steps[:n]) == full[:n]
    steps[0].thought = 'gold future secret'
    assert run(steps) == full
    assert all('secret' not in r['text'] for r in full)


def test_groups():
    rows = [dict(task='Task A', content_sha256='x', source='SWE'),
            dict(task='task a', content_sha256='y', source='SWE'),
            dict(task='Task B', content_sha256='y', source='SWE')]
    assign_folds(rows)
    assert len({r['fold'] for r in rows}) == 1
    assert len({r['group'] for r in rows}) == 1


def test_delay():
    late = [dict(boundary_step_id=5,detected_at_step_id=8)]
    assert score(late,[5],delay=0)['counts'] == [0,1,1]
    assert score(late,[5],delay=2)['f1'] == 0
    assert score(late,[5],delay=3)['f1'] == 1
    assert score(late,[5])['f1'] == 1
    assert score(late+late,[5])['counts'] == [1,2,1]


def test_candidate_switch():
    rules = Rules()
    seg = Segmenter(rules,'phase')
    def feed(i,tc,events):
        return seg.update(Step(i,tool_class=tc),[Event(e,i) for e in events],{'node':'n1','closed':[]})
    feed(1,'EDIT',[E.STATE_CHANGE])
    feed(2,'READ',[])  # INSPECT candidate; not yet confirmed
    result = feed(3,'VERIFY',[E.VERIFY_RUN])  # VERIFY begins HERE, not at step 2
    assert result[0]['end_step_id'] == 2


if __name__ == '__main__':
    test_prefix(); test_groups(); test_delay(); test_candidate_switch()
    print('boundary tests passed')
