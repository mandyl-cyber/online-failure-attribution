"""Run the frozen 6+7 claim requests using the existing local 32B AWQ installation."""
import argparse
from collections.abc import Mapping
from numbers import Integral
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time
import urllib.request

from remote_pilots.openhands_smoke import claims_v3 as pilot
from remote_pilots.openhands_smoke import claims_short_diagnostic as short
from segmenter import stage_claim_summary as core
from scripts.download_qwen32_awq import REVISION, WEIGHT_SHA256

PACKS = {'stages': pilot.ROOT, 'short': short.ROOT}
BASELINE = Path('pilot/swe_claims_qwen32_comparison_v1')


def prepare_baseline(out):
    records = []
    for arm, root in PACKS.items():
        returned = Path('swe_claims_v3' if arm == 'stages' else 'swe_claims_short_v1')
        rows = pilot.rows_at(root / 'development.jsonl')
        if rows != pilot.rows_at(returned / 'development.jsonl'):
            raise ValueError('7B returned input mismatch')
        for row in rows:
            result = pilot.read(returned / (row['job_id'] + '.json'))
            assert core.strict_json(result['request']['messages'][1]['content']) == row['request']
            assert result['request']['messages'][0]['content'] == core.INSTRUCTION
            records.append(dict(arm=arm, job_id=row['job_id'], result=result))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'baseline_7b.jsonl').write_bytes(b''.join(core.encoded(r) + b'\n' for r in records))
    pilot.save(out / 'protocol.json', dict(
        experiment='qwen32-awq-stage-claims-v1', baseline_sha256=core.digest(core.encoded(records)),
        pack_sha256={arm:core.digest(core.encoded(pilot.rows_at(root / 'development.jsonl')))
                     for arm, root in PACKS.items()},
        baseline_config={arm:pilot.read(Path('swe_claims_v3' if arm=='stages' else 'swe_claims_short_v1')/'config.json')
                         for arm in PACKS},
        calls=13, unchanged=['prompt', 'evidence', 'schema', 'temperature=0', 'max_tokens=2048'],
        changes=['model weights/size', 'AWQ quantization', 'vLLM backend and precision'],
        scope='Historical development comparison; semantic review required; not isolated parameter-count effect.'))
    print(f'Frozen {len(records)} original 7B results at {out}')


def check_inputs():
    protocol = pilot.read(BASELINE / 'protocol.json')
    baseline = pilot.rows_at(BASELINE / 'baseline_7b.jsonl')
    if core.digest(core.encoded(baseline)) != protocol['baseline_sha256']:
        raise ValueError('7B baseline drift')
    for arm, root in PACKS.items():
        rows = pilot.rows_at(root / 'development.jsonl')
        if core.digest(core.encoded(rows)) != protocol['pack_sha256'][arm]:
            raise ValueError('frozen evidence drift: ' + arm)
        pack = pilot.read(root / 'protocol.json')
        if pack['instruction_sha256'] != core.digest(core.INSTRUCTION.encode()):
            raise ValueError('prompt drift')
        for row in rows: core.verify_row(row)
    return protocol, baseline


def token_ids(value):
    """Normalize one sequence, including Transformers BatchEncoding and tensors."""
    if isinstance(value, Mapping):
        if 'input_ids' not in value:
            raise ValueError('tokenizer mapping lacks input_ids')
        value = value['input_ids']
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)) or not value or any(
            isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in value):
        raise ValueError('unexpected tokenizer input_ids: expected one nonempty integer sequence')
    return list(value)


def token_preflight(tokenizer, context_limit=8192):
    checks = []
    for arm, root in PACKS.items():
        output_budget = pilot.read(root / 'protocol.json')['limits']['max_tokens']
        for row in pilot.rows_at(root / 'development.jsonl'):
            messages = [dict(role='system', content=core.INSTRUCTION),
                        dict(role='user', content=core.encoded(row['request']).decode('utf-8'))]
            ids = token_ids(tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=False))
            checks.append(dict(arm=arm, job_id=row['job_id'], input_tokens=len(ids),
                               output_budget=output_budget, fits=len(ids)+output_budget<=context_limit))
    return checks


def verify_weights(model):
    checked = {}
    for name, expected in WEIGHT_SHA256.items():
        digest = hashlib.sha256()
        with (model / name).open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''): digest.update(block)
        if digest.hexdigest() != expected: raise ValueError('weight hash mismatch: ' + name)
        checked[name] = expected
        print('Weight verified:', name, flush=True)
    return checked


def gpu_snapshot():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version',
                                 '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        return dict(returncode=result.returncode, output=result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(error=type(exc).__name__)


def compare(out, baseline):
    pairs, groups = [], {}
    for old in baseline:
        path = out / old['arm'] / (old['job_id'] + '.json')
        new = pilot.read(path) if path.exists() else dict(status='missing')
        pairs.append(dict(arm=old['arm'], job_id=old['job_id'],
            baseline_status=old['result']['status'], candidate_status=new['status'],
            baseline_seconds=old['result'].get('seconds'), candidate_seconds=new.get('seconds'),
            baseline_raw_response=old['result'].get('raw_response'),
            candidate_raw_response=new.get('raw_response'),
            baseline_claims=old['result'].get('claims', []), candidate_claims=new.get('claims', []),
            candidate_error=new.get('error'), reviewed=False, reviewer_type=None,
            citation_support=None, key_point_coverage=None, severe_errors=[]))
    for arm in PACKS:
        part = [p for p in pairs if p['arm']==arm]
        groups[arm] = {}
        for name in ('baseline','candidate'):
            times = [p[name+'_seconds'] for p in part if p[name+'_seconds'] is not None]
            groups[arm][name] = dict(
                attempted_or_present=sum(p[name+'_status']!='missing' for p in part),
                schema_ready=sum(p[name+'_status'] in ('ready','ready_empty') for p in part),
                total_seconds=sum(times), median_seconds=statistics.median(times) if times else None,
                accepted_claims=sum(len(p[name+'_claims']) for p in part))
    pilot.save(out / 'comparison.json', dict(groups=groups, semantic_quality='not_evaluated',
        warning='Different serving engines; request latency excludes startup. Do not infer semantic accuracy from readiness.'))
    review = out / 'paired_review.json'
    if not review.exists():
        review.write_text(json.dumps(pairs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return pairs


def run(args):
    if os.name != 'posix': raise ValueError('Managed server launch requires the remote Linux host')
    protocol, baseline = check_inputs()
    model, vllm = args.model_path.resolve(), args.vllm.resolve()
    if not model.is_dir() or not vllm.is_file():
        raise ValueError('Existing model/vLLM environment missing; no automatic install or download')
    # Refuse to attach to or terminate an unrelated existing service.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', args.port))
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output([str(vllm), '--version'], text=True, timeout=60).strip()
    config = dict(protocol_sha256=core.digest(core.encoded(protocol)), model_path=str(model),
        pinned_weight_revision=REVISION, port=args.port, vllm_version=version,
        context_limit=8192, max_num_seqs=1, gpu_memory_utilization=0.92, dtype='float16',
        wrapper_sha256=core.digest(Path(__file__).read_bytes().replace(b'\r\n',b'\n')))
    if (out/'experiment_config.json').exists() and pilot.read(out/'experiment_config.json') != config:
        raise ValueError('experiment configuration changed; choose a new --out')
    pilot.save(out/'experiment_config.json', config)
    weights = verify_weights(model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    checks = token_preflight(tokenizer)
    pilot.save(out/'preflight.json', dict(context_limit=8192, requests=checks))
    metadata = short.model_files(model); metadata['verified_weight_sha256'] = weights
    pilot.save(out/'model_metadata.json', metadata)
    if not all(c['fits'] for c in checks):
        raise ValueError('input + 2048 output exceeds 8192; inspect preflight, no truncation or model calls')
    print('Token preflight: 13/13 fit; starting existing vLLM environment.', flush=True)
    command = [str(vllm), 'serve', str(model), '--served-model-name', 'qwen32-awq-local',
        '--host','127.0.0.1','--port',str(args.port),'--dtype','float16',
        '--max-model-len','8192','--gpu-memory-utilization','0.92','--max-num-seqs','1']
    stamp = str(time.time_ns()); pilot.save(out/f'server_launch_{stamp}.json',dict(argv=command,started_at=pilot.now()))
    server = None; codes = {}; gpu = dict(before_launch=gpu_snapshot())
    def interrupted(signum, frame): raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        with (out/f'server_{stamp}.log').open('w', encoding='utf-8') as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            base = f'http://127.0.0.1:{args.port}/v1'; deadline=time.monotonic()+1200
            while True:
                if server.poll() is not None: raise RuntimeError('vLLM exited; inspect saved server log')
                try:
                    with urllib.request.urlopen(base+'/models',timeout=3) as response: models=json.load(response)
                    if any(m['id']=='qwen32-awq-local' for m in models.get('data',[])): break
                except (OSError, ValueError): pass
                if time.monotonic()>deadline: raise TimeoutError('server startup timed out')
                time.sleep(5)
            pilot.save(out/'server_models.json', models)
            gpu['server_ready'] = gpu_snapshot()
            for arm, root in PACKS.items():
                print('Running', arm, flush=True)
                # Schema failures return 1, but do not prevent the second arm running.
                codes[arm] = pilot.run(root, out/arm, base, 'qwen32-awq-local')
            compare(out, baseline)
            gpu['after_requests'] = gpu_snapshot()
    finally:
        if server is not None:
            # Only the process group launched by this wrapper; no pkill or shared-service stop.
            try: os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try: os.killpg(server.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                server.wait(timeout=10)
        signal.signal(signal.SIGTERM, previous)
        gpu['after_stop'] = gpu_snapshot()
        pilot.save(out/f'gpu_snapshots_{stamp}.json', gpu)
        pilot.save(out/f'completion_{stamp}.json',dict(completed_at=pilot.now(),arm_exit_codes=codes,
                    server_exit_code=server.poll() if server is not None else None))
    print('Return entire directory:', out, flush=True)
    return 0 if codes and all(c==0 for c in codes.values()) else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare-baseline');prep.add_argument('--out',type=Path,default=BASELINE)
    launch=sub.add_parser('run')
    launch.add_argument('--model-path',type=Path,default=Path('vendor/models/Qwen2.5-32B-Instruct-AWQ'))
    launch.add_argument('--vllm',type=Path,default=Path('vendor/.venv-vllm/bin/vllm'))
    launch.add_argument('--port',type=int,default=30032)
    launch.add_argument('--out',type=Path,default=Path('out/swe_claims_qwen32_v1'))
    args=parser.parse_args()
    if args.command=='prepare-baseline': prepare_baseline(args.out);return 0
    return run(args)


if __name__=='__main__': raise SystemExit(main())
