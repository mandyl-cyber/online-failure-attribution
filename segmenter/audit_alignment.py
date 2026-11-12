"""Audit all previously misaligned trajectories without relabeling or truncating."""
import json
from collections import Counter
from pathlib import Path
from .bench import iter_bench
from .adapters.openhands import convert_events, load_trajectory


def main():
    selected = {r['traj_id'] for r in json.loads(Path('eval/m1_audit/skipped.json').read_text(encoding='utf-8'))
                if r['reason'] == 'misaligned'}
    report = []
    for e in iter_bench('codetrace_bench/bench_manifest.verified.jsonl', 'codetrace_bench', traj_ids=selected):
        p = Path(e['traj_path']) / 'events.jsonl'
        events = [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
        normal, _ = convert_events(events)
        folded, _ = convert_events(events, fold_agent_messages=True)
        cmds = [x for x in events if x.get('source') == 'agent' and x.get('action') in ('run','run_ipython')]
        actions = Counter(x.get('action') for x in events if isinstance(x.get('action'),str) and x.get('source')=='agent')
        expected = e['step_count']
        report.append(dict(traj_id=e['traj_id'], expected=expected, default=len(normal), folded=len(folded),
            run_actions=len(cmds), actions=actions,
            variants={'drop_finish':sum(s.tool != 'finish' for s in folded),
                      'drop_finish_think_tracker':sum(s.tool not in ('finish','think','task_tracking') for s in folded)},
            action_ids=[x.get('id') for x in cmds],
            tail=[dict(idx=s.idx, tool=s.tool, action=s.action[:180], raw_ref=s.raw_ref) for s in folded[-8:]],
            note='Count equality alone does not establish semantic alignment; original step annotations required.'))
    out = Path('eval/alignment_audit')
    out.mkdir(parents=True, exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in report:
        print(r['traj_id'],r['expected'],r['default'],r['folded'],r['run_actions'],r['variants'],dict(r['actions']))


if __name__ == '__main__':
    main()
