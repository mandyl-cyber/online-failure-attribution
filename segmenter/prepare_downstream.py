"""Prepare paired inputs; verify official CodeTracer normalization and prompt injection.

No API calls. Only stage boundaries differ. Labels stay outside the agent inputs.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .adapters.openhands import load_trajectory
from .bench import iter_bench, read_manifest
from .rules import Rules
from .run_online import run_steps


def validate_ranges(stages, ids):
    covered = []
    for s in stages:
        if s['end_step_id'] < s['start_step_id']:
            raise ValueError('Empty/reversed stage')
        covered.extend(range(s['start_step_id'], s['end_step_id'] + 1))
    if covered != ids:
        raise ValueError('Stages must cover each annotated step exactly once in order')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='eval/downstream_pair')
    ap.add_argument('--vendor', default='vendor/CodeTracer')
    a = ap.parse_args()
    out, vendor = Path(a.out), Path(a.vendor)
    if out.exists():
        raise SystemExit('Choose a fresh output directory; results are not overwritten.')
    sys.path.insert(0, str((vendor/'src').resolve()))
    sys.path.insert(0, str((vendor.parent/'python_deps').resolve()))
    from codetracer.query.normalizer import _load_pre_normalized
    from codetracer.query.tree_builder import TreeBuilder
    from codetracer.services.classification import ClassificationStore
    from codetracer.agents.context import ContextAssembler
    from codetracer.query.config import load_config
    from codetracer.state.output_profile import load_profile
    class BlankSafeStore(ClassificationStore):
        @staticmethod
        def _normalize_key(action):
            lines = action.strip().splitlines()
            return lines[0][:200] if lines else ''
    config = load_config()
    config['memory'] = {'enabled': False}
    config['trace']['step_limit'] = 60
    profile = load_profile('tracebench', config)
    assembler = ContextAssembler(config, None)
    selected = json.loads(Path('pilot/m2_codetrace/reviewer_only/selection.json').read_text(encoding='utf-8'))
    tids = {r['traj_id'] for r in selected}
    manifest = 'codetrace_bench/bench_manifest.verified.jsonl'
    raw = {r['traj_id']:r for r in read_manifest(manifest)}
    rules = Rules()
    pairs, gold_rows = [], []
    def write(p, obj):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    for e in iter_bench(manifest, 'codetrace_bench', traj_ids=tids):
        if e.get('error'):
            raise ValueError(e['error'])
        steps, meta = load_trajectory(e['traj_path'], rules, base=e['base'], expect_steps=e['step_count'])
        if meta.get('n_core',len(steps)) != e['step_count']:
            raise ValueError('Alignment changed: ' + e['traj_id'])
        # Unannotated extra finish is excluded identically from both arms.
        steps = steps[:e['step_count']]
        ids = [s.idx for s in steps]
        res = run_steps(steps, rules, modes=['phase'])
        stages = {'gold':e['stages'], 'rule':res['stages']['phase']}
        normalized = [dict(step_id=s.idx, action=s.action, observation=s.observation,
                           thinking=s.thought, tool_type=s.tool) for s in steps]
        prompts = {}
        for arm in ('gold','rule'):
            d = out/'inputs'/arm/e['traj_id']
            write(d/'steps.json', normalized)
            (d/'task.md').write_text(meta.get('task',''), encoding='utf-8')
            clean = [{k:s[k] for k in ('stage_id','start_step_id','end_step_id')} for s in stages[arm]]
            validate_ranges(clean, ids)
            write(d/'stage_ranges.json', clean)
            traj = _load_pre_normalized(d)
            assert [s.step_id for s in traj.steps] == ids
            builder = TreeBuilder()
            builder._store = BlankSafeStore(store_path=out/'absent_classifications.jsonl')
            (d/'tree.md').write_text(builder.build(traj), encoding='utf-8')
            messages = assembler.build_trace_messages(d, None, profile=profile, memory_text='')
            assert (d/'stage_ranges.json').read_text(encoding='utf-8') in messages[1]['content']
            prompts[arm] = messages
            write(out/'prompt_checks'/arm/(e['traj_id']+'.json'), messages)
        g, r = (out/'inputs'/arm/e['traj_id'] for arm in ('gold','rule'))
        hashes = {}
        for name in ('steps.json','task.md','tree.md'):
            assert (g/name).read_bytes() == (r/name).read_bytes(), name
            hashes[name] = hashlib.sha256((g/name).read_bytes()).hexdigest()
        def scrub(arm, d):
            return json.dumps(prompts[arm], ensure_ascii=False).replace(
                json.dumps((d/'stage_ranges.json').read_text(encoding='utf-8'),ensure_ascii=False)[1:-1], '<STAGES>').replace(
                json.dumps(str(d),ensure_ascii=False)[1:-1], '<RUN_DIR>')
        assert scrub('gold',g) == scrub('rule',r)
        pairs.append(dict(traj_id=e['traj_id'], n_steps=len(steps), hashes=hashes,
                          gold_stages=len(stages['gold']), rule_stages=len(stages['rule'])))
        gold_rows.append(raw[e['traj_id']])
    assert len(pairs) == 40
    (out/'evaluation_manifest.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in gold_rows),encoding='utf-8')
    write(out/'pairs.json', pairs)
    write(out/'segmenter_rules.json',rules.raw)
    commit = subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()
    write(out/'status.json',dict(n_pairs=len(pairs), official_commit=commit,
         normalizer_validated=True, prompt_stage_injection_validated=True,
         identical_nonstage_inputs=True, inference_completed=False,
         compatibility_adapter='BlankSafeStore guards upstream whitespace-only action IndexError; raw actions unchanged',
         selection='existing stratified 40-task M2 CodeTraceBench subset',
         limitation='trajectory-only diagnostic ablation, no original executable task sandbox',
         missing=['model/API configuration','budget','POSIX isolated execution environment']))
    print(f'Validated {len(pairs)} pairs with official CodeTracer {commit}')


if __name__ == '__main__':
    main()
