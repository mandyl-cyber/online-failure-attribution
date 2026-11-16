#!/usr/bin/env python3
"""Run on the Linux Docker host. No OpenHands/PyTorch host dependency."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

APP = 'ghcr.io/all-hands-ai/openhands@sha256:00968de77a7b36546413e1f3b5c6b4e2c387d19ade54de7926e10f5b3d265fa6'
RUNTIME = 'ghcr.io/all-hands-ai/runtime@sha256:6791db9f6344908f642ca422bd2328c1a39b667566d95c09c6c64ee22d202ac1'
TASK = '''Fix /workspace/ranges.py: merge_ranges must merge overlapping or touching
half-open intervals, return sorted intervals, handle empty input, and not mutate
the input list. Inspect the implementation and tests, reproduce the failure,
make a minimal fix, and run python -m unittest -v. Do not change test_ranges.py.
Use the finish tool when done. Work autonomously; no dependency installation is
needed. This is a small local code task, not a request for an example solution.'''
CODE = '''def merge_ranges(intervals):
    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged
'''
TESTS = '''import unittest
from ranges import merge_ranges

class MergeTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(merge_ranges([]), [])
    def test_touching(self):
        self.assertEqual(merge_ranges([(1, 3), (3, 5)]), [(1, 5)])
    def test_nested(self):
        self.assertEqual(merge_ranges([(1, 8), (2, 4)]), [(1, 8)])
    def test_sorted_disjoint(self):
        self.assertEqual(merge_ranges([(7, 9), (1, 2)]), [(1, 2), (7, 9)])
    def test_immutable(self):
        source = [(7, 9), (1, 2)]
        before = list(source)
        merge_ranges(source)
        self.assertEqual(source, before)

if __name__ == '__main__':
    unittest.main()
'''


def logged(command, path):
    with path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def main():
    if sys.platform != 'linux':
        raise SystemExit('Run this script on the remote Linux Docker host.')
    subprocess.run(['docker', 'ps'], check=True)
    for image in (APP, RUNTIME):
        subprocess.run(['docker', 'image', 'inspect', image], check=True,
                       stdout=subprocess.DEVNULL)
    with urllib.request.urlopen('http://127.0.0.1:30001/v1/models', timeout=10) as response:
        models = json.load(response)
    if not any(m['id'] == 'qwen-local' for m in models['data']):
        raise SystemExit('qwen-local is not available')
    base = Path('out/openhands_live_smoke').resolve()
    base.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='run-', dir=base))
    workspace = run / 'workspace'
    state = run / 'state'
    checks = run / 'checks'
    for directory in (workspace, state, checks):
        directory.mkdir()
    (workspace / 'ranges.py').write_text(CODE, encoding='utf-8')
    (workspace / 'test_ranges.py').write_text(TESTS, encoding='utf-8')
    (checks / 'test_ranges.py').write_text(TESTS, encoding='utf-8')
    (state / 'task.txt').write_text(TASK, encoding='utf-8')
    (run / 'manifest.json').write_text(json.dumps({
        'app_image': APP, 'runtime_image': RUNTIME, 'host_workspace': str(workspace),
        'max_iterations': 20, 'model': 'qwen-local', 'purpose': 'synthetic transport smoke',
    }, indent=2), encoding='utf-8')
    print(f'RUN_DIR={run}', flush=True)
    source = Path(__file__).resolve().parent
    command = ['docker', 'run', '--rm', '--network', 'host',
               '-v', '/var/run/docker.sock:/var/run/docker.sock',
               '-v', f'{source}:/pilot:ro', '-v', f'{state}:/state',
               '-e', f'PILOT_WORKSPACE={workspace}',
               '-e', f'PILOT_UID={os.getuid()}', '-e', f'PILOT_RUNTIME={RUNTIME}',
               '--entrypoint', 'python', APP, '/pilot/controller.py']
    code = logged(command, run / 'controller.log')
    # Check the saved implementation against an immutable copy, inside a container.
    verify = ['docker', 'run', '--rm', '--network', 'none',
              '-v', f'{workspace}:/workspace:ro', '-v', f'{checks}:/checks:ro',
              '--entrypoint', 'python', APP, '-c',
              "import sys,unittest; sys.path.insert(0,'/workspace'); "
              "suite=unittest.defaultTestLoader.discover('/checks'); "
              "r=unittest.TextTestRunner(verbosity=2).run(suite); "
              "sys.exit(not r.wasSuccessful())"]
    verified = logged(verify, run / 'verification.log')
    events = [str(p) for p in state.rglob('events') if p.is_dir()]
    result = {'controller_exit_code': code, 'verification_exit_code': verified,
              'event_directories': events, 'run_directory': str(run)}
    (run / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    # Success here means task checks + saved events, not segmentation accuracy.
    return 0 if code == 0 and verified == 0 and events else 1


if __name__ == '__main__':
    sys.exit(main())
