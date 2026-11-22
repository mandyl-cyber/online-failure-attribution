"""Generate frozen adaptive-sealing pairs with the existing 7B adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.run_stage_claim_pipeline import read, save
from segmenter import stage_claim_summary as core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check-inputs', 'preflight', 'run'))
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--training-run', type=Path, default=Path('out/claim_sft_7b_reviewed_v2'))
    parser.add_argument('--inputs', type=Path, default=Path('pilot/adaptive_seal_claim_v1'))
    parser.add_argument('--out', type=Path, default=Path('out/adaptive_seal_claim_v1'))
    parser.add_argument('--reuse-results', type=Path, help='Reuse identical requests from a completed v1 result folder')
    parser.add_argument('--reuse-inputs', type=Path, default=Path('pilot/adaptive_seal_claim_v1'))
    args = parser.parse_args()
    protocol = read(args.inputs / 'protocol.json')
    source = args.inputs / 'requests.jsonl'
    if hashlib.sha256(source.read_bytes()).hexdigest() != protocol['requests_sha256']:
        raise ValueError('Request pack changed')
    if hashlib.sha256(core.INSTRUCTION.encode()).hexdigest() != protocol['instruction_sha256']:
        raise ValueError('Instruction changed')
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    groups = read(args.inputs / 'groups.json')
    for name, key in [('groups.json', 'groups_sha256'), ('fact_review.json', 'fact_review_sha256'),
                      ('critical_facts.json', 'critical_facts_sha256')]:
        if hashlib.sha256((args.inputs / name).read_bytes()).hexdigest() != protocol[key]:
            raise ValueError('Frozen review/group pack changed: '+name)
    ids = [j for g in groups for arm in ('immediate', 'adaptive') for j in g['arms'][arm]]
    if len(ids) != len(set(ids)) or set(ids) != {r['job_id'] for r in rows}:
        raise ValueError('Duplicate or missing paired request')
    for row in rows:
        core.verify_row(row)
    if args.command == 'check-inputs':
        print(dict(groups=len(groups), requests=len(rows), status='input_hashes_and_provenance_valid'))
        return
    if args.model_path is None:
        parser.error('--model-path is required for preflight/run')
    from transformers import AutoTokenizer, set_seed
    from remote_pilots.openhands_smoke.claim_sft_train import chat_ids
    from segmenter import stage_claim_facts as facts
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    adapter = args.training_run / 'adapter'
    if not (adapter / 'adapter_config.json').exists() or not list(adapter.glob('adapter_model.*')):
        raise FileNotFoundError('Completed training adapter required at '+str(adapter))
    training = read(args.training_run / 'config.json')
    if Path(training['model_path']).resolve() != args.model_path.resolve():
        raise ValueError('Use original training base model')
    config = dict(protocol=protocol, adapter_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(adapter.glob('adapter_*')) if p.is_file()}, seed=training['seed'], facts_version=facts.VERSION)
    if (args.out / 'config.json').exists() and read(args.out / 'config.json') != config:
        raise ValueError('Configuration drift. Choose a new output directory.')
    save(args.out / 'config.json', config)
    encoded = {}
    for row in rows:
        request = dict(row['request'])
        request['observed_facts'] = facts.extract(row['source_snapshot'], row['evidence_map'])
        request['observed_facts_version'] = facts.VERSION
        messages = [dict(role='system', content=core.INSTRUCTION), dict(role='user', content=core.encoded(request).decode())]
        encoded[row['job_id']] = chat_ids(tokenizer, messages, True)
    eligibility = {g['group_id']: all(len(encoded[j])+protocol['max_new_tokens'] <= protocol['max_length']
        for ids in g['arms'].values() for j in ids) for g in groups}
    save(args.out / 'preflight.json', dict(groups=eligibility, prompt_tokens={k: len(v) for k,v in encoded.items()}))
    selected = {j for g in groups if eligibility[g['group_id']] for ids in g['arms'].values() for j in ids}
    print(dict(eligible_groups=sum(eligibility.values()), total_groups=len(groups), requests=len(selected)), flush=True)
    if args.command == 'preflight':
        return
    if not selected:
        raise ValueError('No complete paired group fits the context budget')
    if args.reuse_results is not None:
        old_config = read(args.reuse_results/'config.json')
        if old_config['adapter_sha256'] != config['adapter_sha256'] or old_config['seed'] != config['seed'] or old_config['facts_version'] != config['facts_version']:
            raise ValueError('Reuse model/adapter/fact configuration mismatch')
        for key in ('instruction_sha256','max_length','max_new_tokens'):
            if old_config['protocol'][key] != protocol[key]:
                raise ValueError('Reuse prompt or generation budget mismatch')
        old_source = args.reuse_inputs/'requests.jsonl'
        if hashlib.sha256(old_source.read_bytes()).hexdigest() != old_config['protocol']['requests_sha256']:
            raise ValueError('Reuse input pack drift')
        old_rows = {r['job_id']:r for r in map(core.strict_json,old_source.read_text(encoding='utf-8').splitlines())}
        by_id = {r['job_id']:r for r in rows}
        for job_id in sorted(selected):
            old_path = args.reuse_results/(job_id+'.json')
            target = args.out/(job_id+'.json')
            if target.exists() or not old_path.exists():
                continue
            if core.encoded(old_rows[job_id]) != core.encoded(by_id[job_id]):
                raise ValueError('Reuse request content mismatch')
            previous = read(old_path)
            if previous['job_id'] != job_id or previous['status'] not in ('ready','failed'):
                raise ValueError('Invalid prior result identity/status')
            if previous['status']=='ready' and core.validate(core.strict_json(previous['text']),by_id[job_id]) != previous['claims']:
                raise ValueError('Prior result provenance mismatch')
            save(target,dict(previous,reused_from=str(old_path),
                             reused_result_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest()))
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    set_seed(training['seed'])
    model = AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16), device_map={'':0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True).eval()
    model.config.use_cache = True
    # Alternate arm order by group to reduce systematic ordering effects.
    order = [j for i,g in enumerate(groups) if eligibility[g['group_id']]
             for arm in (('immediate','adaptive') if i % 2 == 0 else ('adaptive','immediate')) for j in g['arms'][arm]]
    by_id = {r['job_id']: r for r in rows}
    for number, job_id in enumerate(order, 1):
        target = args.out / (job_id + '.json')
        if target.exists():
            continue
        tensor = torch.tensor([encoded[job_id]], device=model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor), do_sample=False,
                max_new_tokens=protocol['max_new_tokens'], pad_token_id=tokenizer.pad_token_id)
        tokens = generated[0, tensor.shape[-1]:].tolist()
        result = dict(job_id=job_id, text=tokenizer.decode(tokens, skip_special_tokens=True),
            input_sha256=by_id[job_id]['input_sha256'], prompt_tokens=tensor.shape[-1], generated_tokens=len(tokens),
            seconds=time.perf_counter()-started, token_limit=len(tokens) >= protocol['max_new_tokens'], status='failed')
        try:
            if result['token_limit']:
                raise ValueError('Generation token limit reached')
            result['claims'] = core.validate(core.strict_json(result['text']), by_id[job_id])
            result['status'] = 'ready'
        except (ValueError, KeyError, TypeError) as exc:
            result['error'] = str(exc)
        save(target, result)
        print(number, len(order), result['status'], flush=True)
    results = [read(args.out / (j+'.json')) for j in order]
    save(args.out / 'summary.json', dict(requests=len(order), ready=sum(r['status']=='ready' for r in results),
        semantic_review='pending', eligible_groups=sum(eligibility.values()), total_groups=len(groups),
        reused_requests=sum('reused_from' in r for r in results),
        timing_scope='Request times may mix reused v1 runs and current generation. Record this when comparing costs.'))


if __name__ == '__main__':
    main()
