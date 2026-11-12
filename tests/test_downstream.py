import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segmenter.prepare_downstream import validate_ranges
from segmenter.eval_segmenter import cmd_downstream


def test_downstream():
    validate_ranges([dict(start_step_id=0,end_step_id=1)], [0,1])
    for stages in ([dict(start_step_id=0,end_step_id=0)],
                   [dict(start_step_id=0,end_step_id=1),dict(start_step_id=1,end_step_id=2)]):
        try:
            validate_ranges(stages,[0,1,2])
        except ValueError:
            pass
        else:
            raise AssertionError('Gap/overlap accepted')
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        manifest = p/'manifest.jsonl'
        manifest.write_text(''.join(json.dumps(dict(traj_id=t,agent='OpenHands',incorrect_stages=[
            dict(incorrect_step_ids=[1])]))+'\n' for t in ('a','b')),encoding='utf-8')
        for arm in ('gold','rule'):
            (p/arm/'a').mkdir(parents=True)
            (p/arm/'a/codetracer_labels.json').write_text('[{"incorrect_step_ids":[1]}]')
        args = SimpleNamespace(manifest=str(manifest),gold_run=str(p/'gold'),rule_run=str(p/'rule'),
                               agent='OpenHands',out=str(p/'report.json'))
        cmd_downstream(args)
        r = json.loads((p/'report.json').read_text())
        assert r['paired']==1 and r['expected']==2 and r['gate_pass'] is None
        for arm in ('gold','rule'):
            (p/arm/'b').mkdir(parents=True)
            (p/arm/'b/codetracer_labels.json').write_text('[{"incorrect_step_ids":[1]}]')
        cmd_downstream(args)
        r = json.loads((p/'report.json').read_text())
        assert r['gate_pass'] is True and r['macro_f1_difference_points']==0


if __name__ == '__main__':
    test_downstream()
    print('downstream tests passed')
