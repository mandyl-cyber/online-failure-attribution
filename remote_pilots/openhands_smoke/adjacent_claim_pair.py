"""Run the frozen adjacent-stage claim pairs on the existing remote 32B AWQ model.

One model, one prompt and one decoding budget for both arms. Oversize groups are
excluded symmetrically before generation. The runner never grades semantics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request

from remote_pilots.openhands_smoke import claims_qwen32 as qwen
from segmenter import stage_claim_summary as core


ROOT = Path('pilot/adjacent_claim_pair_v1')
MODEL_NAME = 'qwen32-awq-local'


def read(path):
    return core.strict_json(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(core.encoded(value) + b'\n')
    temp.replace(path)


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(root):
    protocol = read(root/'protocol.json')
    groups = read(root/'groups.json')
    rows = [core.strict_json(line) for line in (root/'requests.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if hash_file(root/'requests.jsonl') != protocol['request_sha256'] or hash_file(root/'groups.json') != protocol['groups_sha256']:
        raise ValueError('Frozen input drift')
    if hash_file(root/'reference_points.json') != protocol['reference_points_sha256']:
        raise ValueError('Frozen pre-output AI reference drift')
    if core.digest(core.INSTRUCTION.encode()) != protocol['instruction_sha256'] or core.digest(core.encoded(core.SCHEMA)) != protocol['schema_sha256']:
        raise ValueError('Claim contract drift')
    if len(groups) != protocol['groups'] or len(rows) != protocol['calls']:
        raise ValueError('Group/request count drift')
    by_id = {row['job_id']:row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {
            job for group in groups for arm in ('original','adjacent') for job in group['arms'][arm]}:
        raise ValueError('Missing/duplicate paired job')
    for row in rows:
        core.verify_row(row)
    return protocol, groups, rows


def request_body(row):
    return dict(model=MODEL_NAME, temperature=0, max_tokens=2048,
                messages=[dict(role='system', content=core.INSTRUCTION),
                          dict(role='user', content=core.encoded(row['request']).decode('utf-8'))])


def preflight(root, tokenizer):
    protocol, groups, rows = inputs(root)
    by_id = {row['job_id']:row for row in rows}
    checks = {}
    for row in rows:
        body = request_body(row)
        tokens = qwen.token_ids(tokenizer.apply_chat_template(
            body['messages'], add_generation_prompt=True, tokenize=True, return_dict=False))
        n = len(tokens)
        checks[row['job_id']] = dict(input_tokens=n, output_budget=2048,
                                     request_bytes=len(core.encoded(body)),
                                     fits=n+2048 <= protocol['limits']['context_tokens']
                                     and len(core.encoded(body)) <= protocol['limits']['max_request_bytes'])
    group_status = {}
    for group in groups:
        jobs = [j for arm in ('original','adjacent') for j in group['arms'][arm]]
        group_status[group['group_id']] = dict(eligible=all(checks[j]['fits'] for j in jobs),
                                               input_tokens={j:checks[j]['input_tokens'] for j in jobs})
    return dict(requests=checks, groups=group_status,
                eligible_groups=sum(x['eligible'] for x in group_status.values()),
                excluded_groups=sum(not x['eligible'] for x in group_status.values()),
                rule='If any arm exceeds the context or request-byte budget, exclude the whole paired group.')


def launch(model, vllm, port, out):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
    command = [str(vllm), 'serve', str(model), '--served-model-name', MODEL_NAME,
               '--host','127.0.0.1','--port',str(port),'--dtype','float16',
               '--max-model-len','8192','--gpu-memory-utilization','0.92','--max-num-seqs','1']
    log = (out/'server.log').open('w', encoding='utf-8')
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    base = f'http://127.0.0.1:{port}/v1'
    deadline = time.monotonic()+1200
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('vLLM exited; inspect server.log')
            try:
                with urllib.request.urlopen(base+'/models', timeout=3) as response:
                    data = json.load(response)
                if MODEL_NAME in [m['id'] for m in data.get('data',[])]:
                    return process, log, base, command
            except (OSError, ValueError):
                pass
            time.sleep(5)
        raise TimeoutError('vLLM startup exceeded 20 minutes')
    except BaseException:
        stop(process, log)
        raise


def stop(process, log):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
    log.close()


def generate(row, base, limit):
    body = request_body(row)
    started = time.perf_counter()
    result = dict(job_id=row['job_id'], status='failed',
                  request_sha256=core.digest(core.encoded(body)),
                  input_sha256=row['input_sha256'], claims=[],
                  semantic_quality='not_evaluated')
    try:
        request = urllib.request.Request(base+'/chat/completions',
            data=core.encoded(body), headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=240) as response:
            raw = response.read(limit+1)
        result['raw_response_text'] = raw.decode('utf-8', errors='replace')
        if len(raw) > limit:
            raise ValueError('Response byte limit exceeded')
        reply = core.strict_json(raw.decode('utf-8'))
        result['raw_response'] = reply
        choice = reply['choices'][0]
        if choice['finish_reason'] != 'stop':
            raise ValueError('Incomplete completion')
        result['claims'] = core.validate(core.strict_json(choice['message']['content']), row)
        result['status'] = 'ready' if result['claims'] else 'ready_empty'
        result['usage'] = reply.get('usage')
    except urllib.error.HTTPError as exc:
        result['error'] = f'HTTP {exc.code}'
        result['http_error_body'] = exc.read(limit).decode('utf-8', errors='replace')
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['seconds'] = time.perf_counter()-started
    return result


def summarize(groups, rows, pre, out):
    summary = {'groups':[], 'arms':{arm:dict(attempted=0, ready=0, failed=0,
                                      claims=0, seconds=0., prompt_tokens=0, completion_tokens=0)
                                  for arm in ('original','adjacent')}}
    for group in groups:
        eligible = pre['groups'][group['group_id']]['eligible']
        item = dict(group_id=group['group_id'], task=group['task'], eligible=eligible,
                    arms={})
        for arm in ('original','adjacent'):
            results = [read(out/(job+'.json')) for job in group['arms'][arm]] if eligible else []
            statuses = [r['status'] for r in results]
            item['arms'][arm] = dict(statuses=statuses,
                                    claims=sum(len(r['claims']) for r in results),
                                    seconds=sum(r['seconds'] for r in results))
            target = summary['arms'][arm]
            target['attempted'] += len(results)
            target['ready'] += sum(s in ('ready','ready_empty') for s in statuses)
            target['failed'] += sum(s == 'failed' for s in statuses)
            target['claims'] += item['arms'][arm]['claims']
            target['seconds'] += item['arms'][arm]['seconds']
            target['prompt_tokens'] += sum((r.get('usage') or {}).get('prompt_tokens',0) for r in results)
            target['completion_tokens'] += sum((r.get('usage') or {}).get('completion_tokens',0) for r in results)
        summary['groups'].append(item)
    summary['semantic_quality'] = 'not_evaluated'
    summary['warning'] = 'Claims counts and schema success do not measure factual support or coverage.'
    save(out/'summary.json', summary)
    return summary


def prepare_review(groups, pre, out):
    """Build an unscored packet; a reviewer must inspect support and coverage."""
    refs = {r['group_id']:r for r in read(out/'reference_points.json')}
    packet = []
    for group in groups:
        eligible = pre['groups'][group['group_id']]['eligible']
        arms = {}
        for arm in ('original','adjacent'):
            arms[arm] = []
            if eligible:
                for job in group['arms'][arm]:
                    result = read(out/(job+'.json'))
                    arms[arm].append(dict(job_id=job, status=result['status'],
                                          claims=result.get('claims',[]),
                                          request_seconds=result.get('seconds'),
                                          usage=result.get('usage')))
        packet.append(dict(group_id=group['group_id'], task=group['task'],
                           envelope=group['envelope'], eligible=eligible,
                           reference_points=refs[group['group_id']]['points'],
                           forbidden_inferences=refs[group['group_id']]['forbidden_inferences'],
                           arms=arms, reviewed=False, reviewer_type=None,
                           point_coverage={}, claim_support={}, severe_errors=[],
                           winner=None, review_note=''))
    path = out/'paired_review.json'
    if not path.exists():
        save(path, packet)


def run(args):
    if os.name != 'posix':
        raise ValueError('Managed 32B run requires the remote Linux host')
    protocol, groups, rows = inputs(args.inputs)
    model, vllm = args.model_path.resolve(), args.vllm.resolve()
    if not model.is_dir() or not vllm.is_file():
        raise FileNotFoundError('Existing model or vLLM executable missing')
    vllm_version = subprocess.check_output([str(vllm),'--version'], text=True, timeout=60).strip()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    config = dict(protocol_sha256=hash_file(args.inputs/'protocol.json'),
                  requests_sha256=hash_file(args.inputs/'requests.jsonl'),
                  groups_sha256=hash_file(args.inputs/'groups.json'),
                  runner_sha256=hash_file(Path(__file__)),
                  core_sha256=hash_file(Path(core.__file__)),
                  model_path=str(model), model=MODEL_NAME,
                  pinned_model_revision=qwen.REVISION, vllm_version=vllm_version, port=args.port,
                  max_model_len=8192, max_num_seqs=1, temperature=0, max_tokens=2048)
    if (out/'config.json').exists() and read(out/'config.json') != config:
        raise ValueError('Run configuration changed; use a new --out')
    save(out/'config.json', config)
    for name in ('protocol.json','groups.json','requests.jsonl','reference_points.json'):
        target = out/name
        source = args.inputs/name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError('Saved input pack drift: '+name)
        if not target.exists():
            target.write_bytes(source.read_bytes())
    weights = qwen.verify_weights(model)
    save(out/'weight_sha256.json', weights)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    pre = preflight(args.inputs, tokenizer)
    save(out/'preflight.json', pre)
    print(f"Preflight: {pre['eligible_groups']}/{len(groups)} paired groups fit", flush=True)
    if pre['eligible_groups'] == 0:
        raise ValueError('No paired group fits 8192-token budget; no model calls')
    process = log = None
    save(out/'gpu_before.json', qwen.gpu_snapshot())
    try:
        process, log, base, command = launch(model, vllm, args.port, out)
        save(out/'server_command.json', command)
        save(out/'gpu_ready.json', qwen.gpu_snapshot())
        selected = {job for group in groups if pre['groups'][group['group_id']]['eligible']
                    for arm in ('original','adjacent') for job in group['arms'][arm]}
        for i, row in enumerate(rows, 1):
            if row['job_id'] not in selected:
                continue
            target = out/(row['job_id']+'.json')
            if target.exists():
                if read(target)['status'] not in ('ready','ready_empty','failed'):
                    raise ValueError('Incomplete prior result: '+str(target))
                continue
            result = generate(row, base, protocol['limits']['max_response_bytes'])
            save(target, result)
            print(f'{i}/{len(rows)} {result["status"]}', flush=True)
        save(out/'gpu_after_requests.json', qwen.gpu_snapshot())
    finally:
        if process is not None:
            stop(process, log)
        save(out/'gpu_after_stop.json', qwen.gpu_snapshot())
    summary = summarize(groups, rows, pre, out)
    prepare_review(groups, pre, out)
    print('Saved',out/'summary.json',flush=True)
    return 0 if all(a['failed']==0 for a in summary['arms'].values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check')
    check.add_argument('--inputs',type=Path,default=ROOT)
    launch_cmd = sub.add_parser('run')
    launch_cmd.add_argument('--inputs',type=Path,default=ROOT)
    launch_cmd.add_argument('--out',type=Path,default=Path('out/adjacent_claim_pair_v1'))
    launch_cmd.add_argument('--model-path',type=Path,default=Path('vendor/models/Qwen2.5-32B-Instruct-AWQ'))
    launch_cmd.add_argument('--vllm',type=Path,default=Path('vendor/.venv-vllm/bin/vllm'))
    launch_cmd.add_argument('--port',type=int,default=30032)
    args=parser.parse_args()
    if args.command=='check':
        protocol,groups,rows=inputs(args.inputs)
        print(json.dumps(dict(groups=len(groups),requests=len(rows),model=protocol['model']),ensure_ascii=False))
        return 0
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
