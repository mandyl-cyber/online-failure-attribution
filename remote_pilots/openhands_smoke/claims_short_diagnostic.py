"""Short complete step diagnostics; same v3 prompt, schema and generation budget."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import subprocess
import urllib.request
from urllib.parse import urlsplit

from remote_pilots.openhands_smoke import claims_v3 as pilot
from segmenter import stage_claim_summary as core

ROOT = Path('pilot/swe_claims_short_v1')
CASES = [
    ('104c381337', 12, 'edit_not_verified', '工具确认源码修改；不能说功能正确，也不能把要求复核写成已复核。'),
    ('ae1e47e246', 27, 'review_instruction', '工具确认 MethodDefinition 编辑；末尾 Review the changes 是请求，不是完成记录。'),
    ('d49ee42e57', 17, 'dependency_absent', '检查输出 absl present? False，依赖未被该检查发现。'),
    ('d49ee42e57', 18, 'import_failed', 'tensorflow 导入失败；不能把脚本中的成功打印语句当已运行输出。'),
    ('ae1e47e246', 28, 'example_failed', 'shell错误和TypeError，exit 1；不能说示例成功。'),
    ('ae1e47e246', 31, 'example_succeeded', 'babylon 示例输出了带装饰器的 OrderLine，exit 0；不能推广为全测试通过，也不能从单步推断修复过程。'),
    ('c3c276d0ad', 35, 'commit_not_verification', '工具报告提交 f2ded26 和变更计数；提交消息是描述，不是功能验证。'),
]


def prepare(out):
    parents = pilot.rows_at(pilot.ROOT / 'development.jsonl')
    rows, refs, selection = [], [], []
    for prefix, sid, category, expectation in CASES:
        parent, = [r for r in parents if r['source_job_id'].startswith(prefix)]
        source = deepcopy(parent['source_snapshot'])
        source['job_id'] += f':diagnostic-step-{sid}'
        source['stage'] = dict(stage_id=f'diagnostic-{sid}', start_step_id=sid, end_step_id=sid)
        source['detected_at_seq'] = sid
        source['evidence'] = [e for e in source['evidence'] if e['source_step_id'] == sid]
        source['incomplete'] = False
        if {e['field'] for e in source['evidence']} != {'action', 'observation'}:
            raise ValueError('diagnostic must contain complete normalized step')
        row = core.build(source); rows.append(row)
        refs.append(dict(job_id=row['job_id'], reviewer_type='AI', reviewed=True,
                         reviewed_before_output=True, category=category, expected=expectation,
                         evidence_ids=list(row['evidence_map']), semantic_reviewed=False))
        selection.append(dict(job_id=row['job_id'], parent_job_id=parent['job_id'],
                              source_step_id=sid, category=category,
                              same_evidence_as_parent=source['evidence']))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'development.jsonl').write_bytes(b''.join(core.encoded(r) + b'\n' for r in rows))
    pilot.save(out / 'reference_points.json', refs)
    pilot.save(out / 'selection.json', selection)
    pilot.save(out / 'protocol.json', dict(protocol=core.PROTOCOL,
        experiment='short-complete-step-diagnostic-v1', calls=len(rows),
        input_sha256=core.digest(core.encoded(rows)),
        references_sha256=core.digest(core.encoded(refs)),
        instruction_sha256=core.digest(core.INSTRUCTION.encode()),
        schema_sha256=core.digest(core.encoded(core.SCHEMA)),
        limits=dict(max_tokens=2048, max_request_bytes=100000, max_response_bytes=1000000),
        decoder='prompt_only',
        unchanged=['v3 system instruction', 'schema', 'chunk size', 'temperature', 'output token budget'],
        caveats=['Post-error selected development diagnostics; not independent evaluation.',
                 'Single complete step windows are NOT gold stages or a segmentation change.',
                 'Shortening changes context and selection difficulty together; no causal isolation claim.',
                 'Commit case is unchanged evidence control; deterministic repetition is not independent data.',
                 'No synthetic positive claims: reference facts are withheld from model.']))
    print(f'Prepared {len(rows)} short diagnostics: {out}')


def allowed_args(argv):
    """Only allowlisted values; never log complete argv, environment or credentials."""
    result = {}
    for i, arg in enumerate(argv):
        for key in ('--model-path', '--model-name', '--port', '--context-limit'):
            if arg == key and i + 1 < len(argv):
                result[key[2:]] = argv[i + 1]
            elif arg.startswith(key + '='):
                result[key[2:]] = arg.split('=', 1)[1]
    return result


def model_files(path):
    path = Path(path).expanduser().resolve()
    result = dict(path=str(path), config_files={}, weight_files=[], weights_hashed=False)
    for name in ('config.json', 'generation_config.json', 'tokenizer_config.json',
                 'tokenizer.json', 'chat_template.jinja', 'model.safetensors.index.json'):
        file = path / name
        if not file.is_file():
            continue
        entry = dict(bytes=file.stat().st_size)
        if entry['bytes'] <= 32 * 1024 * 1024:
            raw = file.read_bytes(); entry['sha256'] = core.digest(raw)
            if name in ('config.json', 'generation_config.json'):
                obj = json.loads(raw)
                keys = ('model_type', 'architectures', 'hidden_size', 'num_hidden_layers',
                        'num_attention_heads', 'num_key_value_heads', 'intermediate_size',
                        'max_position_embeddings', 'torch_dtype', 'dtype', 'vocab_size',
                        'quantization_config', 'eos_token_id', 'bos_token_id', 'pad_token_id',
                        'do_sample', 'temperature', 'top_p', 'top_k')
                entry['selected_fields'] = {k: obj[k] for k in keys if k in obj}
        result['config_files'][name] = entry
    for pattern in ('*.safetensors', 'pytorch_model*.bin'):
        result['weight_files'].extend(dict(name=f.name, bytes=f.stat().st_size)
                                       for f in sorted(path.glob(pattern)))
    return result


def collect_metadata(base_url, model_path=None):
    report = dict(platform=platform.platform(), collected_at=pilot.now(), server_candidates=[],
                  limitations=['Candidate processes are not proven to own the endpoint socket.',
                               'Config/files do not prove actual in-memory weights or runtime dtype.',
                               'Weight contents not hashed; no full checkpoint identity guarantee.'])
    try:
        with urllib.request.urlopen(base_url.rstrip('/') + '/models', timeout=10) as response:
            report['models_response'] = json.load(response)
    except Exception as exc:
        report['models_error'] = type(exc).__name__  # URLs may contain credentials
    port = urlsplit(base_url).port
    proc = Path('/proc')
    for directory in sorted(proc.glob('[0-9]*')) if proc.is_dir() else []:
        try:
            argv = (directory / 'cmdline').read_bytes().decode('utf-8').split('\0')
            # Recognize only our known server invocation, never arbitrary processes.
            scripts = [a for a in argv if a.endswith('qwen_tool_server/server.py') or a.endswith('serve_local_qwen.py')]
            module = 'remote_pilots.qwen_tool_server.server' in argv
            if not scripts and not module:
                continue
            args = allowed_args(argv)
            if port is not None and str(args.get('port', '30001')) != str(port):
                continue
            candidate = dict(pid=int(directory.name), launch_args=args)
            cwd = (directory / 'cwd').resolve()
            if scripts:
                script = Path(scripts[0]); script = script if script.is_absolute() else cwd / script
            else:
                script = cwd / 'remote_pilots/qwen_tool_server/server.py'
            if script.is_file():
                candidate['server_source_sha256'] = core.digest(script.read_bytes().replace(b'\r\n', b'\n'))
            candidate['python_executable'] = str((directory / 'exe').resolve())
            # Inspect distributions using the server interpreter, not the caller's conda env.
            code = 'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ["torch","transformers"]}))'
            completed = subprocess.run([candidate['python_executable'], '-c', code],
                                       capture_output=True, text=True, timeout=20)
            if completed.returncode == 0:
                candidate['server_packages'] = json.loads(completed.stdout)
            else:
                candidate['server_packages_error'] = 'interpreter query failed'
            if 'model-path' in args:
                path = Path(args['model-path']).expanduser()
                candidate['model_files'] = model_files(path if path.is_absolute() else cwd / path)
            report['server_candidates'].append(candidate)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            report.setdefault('collection_errors', []).append(dict(pid=directory.name, error=type(exc).__name__))
    if model_path:
        report['user_supplied_model_files'] = model_files(model_path)
    if not report['server_candidates']:
        report['process_discovery'] = 'unavailable_or_unrecognized; use --model-path if needed'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--out', type=Path, default=ROOT)
    run = sub.add_parser('run'); run.add_argument('--inputs', type=Path, default=ROOT)
    run.add_argument('--out', type=Path, default=Path('out/swe_claims_short_v1'))
    run.add_argument('--base-url', default='http://127.0.0.1:30001/v1')
    run.add_argument('--model', default='qwen-local'); run.add_argument('--model-path', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out); return 0
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = collect_metadata(args.base_url, args.model_path)
    stamp = pilot.now().replace(':', '-').replace('+', '_')
    pilot.save(args.out / f'model_metadata_{stamp}.json', metadata)
    for name in ('selection.json',):
        dest = args.out / name
        if not dest.exists(): dest.write_bytes((args.inputs / name).read_bytes())
    print('Metadata saved; starting seven development diagnostics.', flush=True)
    return pilot.run(args.inputs, args.out, args.base_url, args.model)


if __name__ == '__main__':
    raise SystemExit(main())
