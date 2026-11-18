"""Run the v4 observed-fact claim prompt on frozen adjacent-claim source jobs.

This is a development compatibility experiment. It reuses the exact sealed
source snapshots from the earlier v3 run and does not change the online default.
"""
import argparse
import json
import os
from pathlib import Path

from remote_pilots.openhands_smoke import adjacent_claim_pair as old
from remote_pilots.openhands_smoke import claims_qwen32 as qwen
from segmenter import stage_claim_summary as v3
from segmenter import stage_claim_summary_v4 as v4


def prepare(root):
    protocol, groups, original = old.inputs(root)
    rows = [v4.build(row['source_snapshot'], row['chunk_chars']) for row in original]
    mapping = {before['job_id']: after['job_id'] for before, after in zip(original, rows)}
    for row in rows:
        v4.verify_row(row)
    return protocol, groups, rows, mapping


def body(row):
    return dict(model=old.MODEL_NAME, temperature=0, max_tokens=2048,
                messages=[dict(role='system', content=v4.INSTRUCTION),
                          dict(role='user', content=v3.encoded(row['request']).decode('utf-8'))])


def preflight(rows, groups, mapping, tokenizer, protocol):
    checks = {}
    for row in rows:
        request = body(row)
        ids = qwen.token_ids(tokenizer.apply_chat_template(
            request['messages'], add_generation_prompt=True,
            tokenize=True, return_dict=False))
        checks[row['job_id']] = dict(input_tokens=len(ids), output_budget=2048,
            fits=len(ids)+2048 <= protocol['limits']['context_tokens']
            and len(v3.encoded(request)) <= protocol['limits']['max_request_bytes'])
    group_checks = {}
    for group in groups:
        jobs = [mapping[j] for arm in ('original', 'adjacent') for j in group['arms'][arm]]
        group_checks[group['group_id']] = dict(eligible=all(checks[j]['fits'] for j in jobs),
                                               jobs=jobs)
    return dict(requests=checks, groups=group_checks,
                eligible_groups=sum(g['eligible'] for g in group_checks.values()),
                rule='Exclude both arms if any request in the group exceeds the fixed budget.')


def generate(row, endpoint, limit):
    import time
    import urllib.request
    request_body = body(row)
    result = dict(job_id=row['job_id'], source_job_id=row['base_row']['job_id'],
                  status='failed', input_sha256=row['input_sha256'],
                  request_sha256=v3.digest(v3.encoded(request_body)),
                  semantic_quality='not_evaluated')
    started = time.perf_counter()
    try:
        request = urllib.request.Request(endpoint+'/chat/completions',
            data=v3.encoded(request_body), headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=240) as response:
            raw = response.read(limit+1)
        if len(raw) > limit:
            raise ValueError('Response byte limit exceeded')
        result['raw_response_text'] = raw.decode('utf-8', errors='replace')
        reply = v3.strict_json(result['raw_response_text'])
        result['raw_response'] = reply
        if reply['choices'][0]['finish_reason'] != 'stop':
            raise ValueError('Incomplete completion')
        parsed = v3.strict_json(reply['choices'][0]['message']['content'])
        checked = v4.validate(parsed, row)
        result['claims'] = checked['claims']
        result['missing_critical_fact_ids'] = checked['missing_critical_fact_ids']
        result['status'] = 'ready' if checked['claims'] else 'ready_empty'
        result['usage'] = reply.get('usage')
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['seconds'] = time.perf_counter()-started
    return result


def run(args):
    if os.name != 'posix':
        raise ValueError('Qwen32 run requires the remote Linux host')
    protocol, groups, rows, mapping = prepare(args.inputs)
    model, vllm = args.model_path.resolve(), args.vllm.resolve()
    if not model.is_dir() or not vllm.is_file():
        raise FileNotFoundError('Existing model or vLLM executable missing')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    config = dict(protocol_sha256=old.hash_file(args.inputs/'protocol.json'),
                  requests_sha256=old.hash_file(args.inputs/'requests.jsonl'),
                  runner_sha256=old.hash_file(Path(__file__)),
                  v3_sha256=old.hash_file(Path(v3.__file__)),
                  v4_sha256=old.hash_file(Path(v4.__file__)),
                  fact_sha256=old.hash_file(Path(__import__('segmenter.stage_claim_facts', fromlist=['']).__file__)),
                  model_revision=qwen.REVISION, model_path=str(model),
                  vllm_version=__import__('subprocess').check_output([str(vllm),'--version'], text=True, timeout=60).strip(),
                  max_model_len=8192, temperature=0, max_tokens=2048)
    if (out/'config.json').exists() and old.read(out/'config.json') != config:
        raise ValueError('Run configuration drift; use a new output directory')
    old.save(out/'config.json', config)
    old.save(out/'mapping.json', mapping)
    old.save(out/'weight_sha256.json', qwen.verify_weights(model))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    checks = preflight(rows, groups, mapping, tokenizer, protocol)
    old.save(out/'preflight.json', checks)
    print(f"Preflight: {checks['eligible_groups']}/{len(groups)} paired groups fit", flush=True)
    if checks['eligible_groups'] == 0:
        raise ValueError('No complete paired group fits; no model calls')
    selected = {j for group in checks['groups'].values() if group['eligible'] for j in group['jobs']}
    old.save(out/'gpu_before.json', qwen.gpu_snapshot())
    process = log = None
    try:
        process, log, endpoint, command = old.launch(model, vllm, args.port, out)
        old.save(out/'server_command.json', command)
        old.save(out/'gpu_ready.json', qwen.gpu_snapshot())
        for i, row in enumerate(rows, 1):
            if row['job_id'] not in selected:
                continue
            target = out/(row['job_id']+'.json')
            if target.exists():
                prior = old.read(target)
                if prior['input_sha256'] != row['input_sha256']:
                    raise ValueError('Resumed result input drift')
                continue
            result = generate(row, endpoint, protocol['limits']['max_response_bytes'])
            old.save(target, result)
            print(f'{i}/{len(rows)} {result["status"]}', flush=True)
        old.save(out/'gpu_after_requests.json', qwen.gpu_snapshot())
    finally:
        if process is not None:
            old.stop(process, log)
        old.save(out/'gpu_after_stop.json', qwen.gpu_snapshot())
    results = [old.read(out/(j+'.json')) for j in selected]
    summary = dict(scope='Same frozen v3 source snapshots; v4 fact prompt and guard. Development compatibility only.',
                   groups=len(groups), eligible_groups=checks['eligible_groups'],
                   attempted=len(results), ready=sum(r['status'] in ('ready','ready_empty') for r in results),
                   failed=sum(r['status']=='failed' for r in results),
                   missing_critical=sum(bool(r.get('missing_critical_fact_ids')) for r in results),
                   semantic_quality='not_reviewed')
    old.save(out/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary['failed'] == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check','run'])
    parser.add_argument('--inputs', type=Path, default=old.ROOT)
    parser.add_argument('--out', type=Path, default=Path('out/stage_claim_facts_v1'))
    parser.add_argument('--model-path', type=Path, default=Path('vendor/models/Qwen2.5-32B-Instruct-AWQ'))
    parser.add_argument('--vllm', type=Path, default=Path('vendor/.venv-vllm/bin/vllm'))
    parser.add_argument('--port', type=int, default=30032)
    args = parser.parse_args()
    protocol, groups, rows, mapping = prepare(args.inputs)
    if args.command == 'check':
        print(json.dumps(dict(groups=len(groups), calls=len(rows), mapped=len(mapping),
                              protocol=v4.PROTOCOL), ensure_ascii=False))
        return 0
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
