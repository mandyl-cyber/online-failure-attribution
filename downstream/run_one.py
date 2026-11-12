"""Run one official CodeTracer diagnosis in an isolated POSIX container.

--dry-run only assembles the real prompt; it never constructs an API client.
Only mount the single input trajectory and an empty output directory.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--vendor', default='vendor/CodeTracer')
    ap.add_argument('--model')
    ap.add_argument('--cost-limit', type=float)
    ap.add_argument('--step-limit', type=int, default=60)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if not a.dry_run:
        if os.name != 'posix' or not Path('/.dockerenv').exists():
            raise SystemExit('Run inference in the provided isolated Linux Docker container.')
        if not a.model or a.cost_limit is None or a.cost_limit <= 0 or a.step_limit <= 0:
            raise SystemExit('Explicit model and positive cost/step limits required.')
        if not all(os.environ.get(k) for k in ('CODETRACER_API_BASE','CODETRACER_API_KEY')):
            raise SystemExit('Configure CODETRACER_API_BASE and CODETRACER_API_KEY in the environment.')
    vendor, inp, out = Path(a.vendor), Path(a.input), Path(a.out)
    sys.path[:0] = [str((vendor/'src').resolve()), str((vendor.parent/'python_deps').resolve())]
    from codetracer.agents.context import ContextAssembler
    from codetracer.query.config import load_config
    from codetracer.state.output_profile import load_profile
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Output directory must be empty; never resume across arms.')
    out.mkdir(parents=True, exist_ok=True)
    files = ('steps.json','task.md','tree.md','stage_ranges.json')
    hashes = {}
    for name in files:
        shutil.copyfile(inp/name, out/name)
        hashes[name] = hashlib.sha256((out/name).read_bytes()).hexdigest()
    config = load_config()
    config['memory'] = {'enabled':False}
    config['trace'].update(step_limit=a.step_limit, cost_limit=a.cost_limit or 1.0)
    profile = load_profile('tracebench', config)
    assembler = ContextAssembler(config, None)
    messages = assembler.build_trace_messages(out, None, profile=profile, memory_text='')
    (out/'initial_prompt.json').write_text(json.dumps(messages,ensure_ascii=False,indent=2),encoding='utf-8')
    record = dict(model=a.model, cost_limit=a.cost_limit, step_limit=a.step_limit,
                  input_hashes=hashes, memory=False, dry_run=a.dry_run, completed=False)
    if not a.dry_run:
        from codetracer.agents.trace_agent import TraceAgent
        from codetracer.llm.client import LLMClient
        from codetracer.services.cost_tracker import CostTracker
        client = LLMClient(api_base=os.environ['CODETRACER_API_BASE'],
                           api_key=os.environ['CODETRACER_API_KEY'], model_name=a.model,
                           azure_ad_resource='', model_kwargs={})
        tracker = CostTracker(budget_limit_usd=a.cost_limit)
        agent = TraceAgent(client, assembler, out, out/profile.output_file, config,
                           profile=profile, cost_tracker=tracker)
        terminal = None
        try:
            for event in agent.run_iter(None, memory_text=''):
                if event.type in ('complete','limits_exceeded','aborted'):
                    terminal = event.type
            agent.save_trajectory(out/'diagnosis.traj.json')
            record.update(terminal=terminal, estimated_cost=tracker.total_cost, api_calls=client.n_calls)
            unchanged = all(hashlib.sha256((out/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
            if terminal != 'complete' or not unchanged:
                raise RuntimeError('Incomplete run or modified input; labels are not evaluable')
            labels = json.loads((out/profile.output_file).read_text(encoding='utf-8'))
            stages = {s['stage_id']:s for s in json.loads((out/'stage_ranges.json').read_text(encoding='utf-8'))}
            if not isinstance(labels,list):
                raise ValueError('Expected tracebench list')
            for label in labels:
                st = stages[label['stage_id']]
                for key in ('incorrect_step_ids','unuseful_step_ids'):
                    if any(type(i) is not int or not st['start_step_id'] <= i <= st['end_step_id'] for i in label[key]):
                        raise ValueError('Label outside its stage')
            record['completed'] = True
        finally:
            (out/'run_status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
            if not record['completed'] and (out/profile.output_file).exists():
                (out/profile.output_file).rename(out/'incomplete_labels.json')
    else:
        (out/'run_status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
