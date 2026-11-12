import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segmenter.context import ExecutionContext
from segmenter.schema import Step
from segmenter.run_online import run_steps
from segmenter.rules import Rules


def test_context():
    c = ExecutionContext()
    c.apply(Step(1, tool='run', observation='Password:'))
    s = Step(2, tool='run', action='rm -rf /app', tool_class='EXEC_WRITE',
             attrs={'irreversible': 'rm -rf /app', 'cmd_classes': ['state']})
    c.apply(s)
    assert s.tool_class == 'OTHER' and 'irreversible' not in s.attrs
    s = Step(3, tool='run', action='pytest', tool_class='VERIFY', exit_code=0)
    c.apply(s)
    assert s.tool_class == 'VERIFY'
    s = Step(4, tool='run', action='python generate.py', observation='Distribution saved to /app/dist.npy',
             exit_code=0, tool_class='EXEC', attrs={'cmd_classes': ['probe']})
    c.apply(s)
    assert s.attrs['reported_writes'][0]['path'] == '/app/dist.npy'
    assert not s.written_paths
    r = Rules(overrides={'execution_context.enabled': True})
    rows = run_steps([Step(1, tool_class='EDIT'), s, Step(5, tool_class='READ')], r)['rows']
    assert 'VERIFY_RUN' not in rows[1]['events'] and 'VERIFY_PASS' not in rows[2]['events']
    c = ExecutionContext()
    c.apply(Step(1, tool='run', observation='Password:', attrs={'terminal_session': 'a'}))
    s = Step(2, tool='run', tool_class='VERIFY', attrs={'terminal_session': 'b'})
    c.apply(s)
    assert s.tool_class == 'VERIFY'


if __name__ == '__main__':
    test_context()
    print('context tests passed')
