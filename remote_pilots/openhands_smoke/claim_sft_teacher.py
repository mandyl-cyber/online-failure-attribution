"""Generate review candidates for existing task-disjoint claim SFT inputs."""
import argparse
import json
from pathlib import Path
from remote_pilots.openhands_smoke import stage_claim_facts_v1 as runner
from segmenter import stage_claim_summary as v3
from segmenter import stage_claim_summary_v5 as v5


def rows(root):
    manifest = runner.old.read(root/"manifest.json")
    groups = {m["job_id"]:m for m in manifest}
    result = []
    for split in ("train","dev"):
        for line in (root/f"{split}_inputs.jsonl").read_text(encoding="utf-8").splitlines():
            old = v3.strict_json(line)
            v3.verify_row(old)
            meta = groups[old["job_id"]]
            if meta["split"] != split:
                raise ValueError("Split mismatch")
            row = v5.build(old["source_snapshot"],old["chunk_chars"])
            result.append((meta,row))
    if {m["group"] for m,r in result if m["split"]=="train"} & {
            m["group"] for m,r in result if m["split"]=="dev"}:
        raise ValueError("Task leakage")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["check","run"])
    parser.add_argument("--inputs",type=Path,default=Path("pilot/swe_claim_sft_v1"))
    parser.add_argument("--out",type=Path,default=Path("out/claim_sft_teacher_v1"))
    parser.add_argument("--model-path",type=Path,default=Path("vendor/models/Qwen2.5-32B-Instruct-AWQ"))
    parser.add_argument("--vllm",type=Path,default=Path("vendor/.venv-vllm/bin/vllm"))
    parser.add_argument("--port",type=int,default=30032)
    parser.add_argument("--partition-long",action="store_true")
    args = parser.parse_args()
    inputs = rows(args.inputs)
    print(json.dumps({s:sum(m["split"]==s for m,r in inputs) for s in ("train","dev")}),flush=True)
    if args.command=="check":
        return
    from transformers import AutoTokenizer
    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    config = dict(input_hash=v3.digest(v3.encoded(inputs)),
        instruction_hash=v3.digest(v5.INSTRUCTION.encode()),
        facts_hash=runner.old.hash_file(Path(__import__("segmenter.stage_claim_facts",fromlist=[""]).__file__)),
        validator_hash=runner.old.hash_file(Path(__import__("segmenter.stage_claim_summary_v4",fromlist=[""]).__file__)),
        core_hash=runner.old.hash_file(Path(v5.__file__)),
        runner_hash=runner.old.hash_file(Path(__file__)),
        model_path=str(args.model_path.resolve()),temperature=0,max_tokens=2048,context=8192,
        partition_long=args.partition_long)
    if (out/"config.json").exists() and runner.old.read(out/"config.json") != config:
        raise ValueError("Configuration changed; use a new output directory")
    runner.old.save(out/"config.json",config)
    runner.old.save(out/"weights.json",runner.qwen.verify_weights(args.model_path.resolve()))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path),local_files_only=True)
    runner.v4 = v5
    eligible = []
    checks = []
    partitions = []
    def fits_row(candidate):
        tokens = runner.qwen.token_ids(tokenizer.apply_chat_template(
            runner.body(candidate)["messages"],tokenize=True,add_generation_prompt=True,return_dict=False))
        return len(tokens)+2048 <= 8192
    for meta,row in inputs:
        if meta.get("known_input_gap"):
            checks.append(dict(job_id=row["job_id"],input_tokens=None,eligible=False,
                               split=meta["split"],reason="known_source_input_gap"))
            continue
        ids = runner.qwen.token_ids(tokenizer.apply_chat_template(
            runner.body(row)["messages"],tokenize=True,add_generation_prompt=True,return_dict=False))
        fits = len(ids)+2048 <= 8192
        checks.append(dict(job_id=row["job_id"],input_tokens=len(ids),eligible=fits,split=meta["split"]))
        if fits:
            eligible.append((meta,row))
        elif args.partition_long:
            from segmenter.claim_input_partition import partition
            parts,oversized = partition(row,fits_row)
            for part in parts:
                part_meta = dict(meta,parent_job_id=row["job_id"],parent_stage=part["parent_stage"])
                eligible.append((part_meta,part["row"]))
            partitions.append(dict(parent_job_id=row["job_id"],
                part_ids=[p["row"]["job_id"] for p in parts],oversized_steps=oversized,
                complete=not oversized))
    runner.old.save(out/"preflight.json",checks)
    runner.old.save(out/"partitions.json",partitions)
    (out/"effective_inputs.jsonl").write_bytes(b"".join(
        v3.encoded(dict(metadata=meta,row=row))+b"\n" for meta,row in eligible))
    if not eligible:
        raise ValueError("No input fits the context budget")
    process,log,endpoint,command = runner.old.launch(args.model_path.resolve(),args.vllm.resolve(),args.port,out)
    runner.old.save(out/"server_command.json",command)
    try:
        for index,(meta,row) in enumerate(eligible,1):
            target = out/(row["job_id"]+".json")
            if target.exists():
                prior=runner.old.read(target)
                if prior["input_sha256"]!=row["input_sha256"]:
                    raise ValueError("Result input drift")
                continue
            result = runner.generate(row,endpoint,1048576)
            result.update(split=meta["split"],group=meta["group"],original_job_id=meta["job_id"])
            result["parent_job_id"] = meta.get("parent_job_id",row["job_id"])
            runner.old.save(target,result)
            print(f'{index}/{len(eligible)} {result["status"]}',flush=True)
    finally:
        runner.old.stop(process,log)
    reviews = []
    for meta,row in eligible:
        value = runner.old.read(out/(row["job_id"]+".json"))
        reviews.append(dict(job_id=row["job_id"],original_job_id=meta["job_id"],
            split=meta["split"],group=meta["group"],status=value["status"],
            parent_job_id=meta.get("parent_job_id",row["job_id"]),
            candidate_claims=value.get("claims",[]),reviewed=False,
            final_response=None,review_note=""))
    review_path=out/"review.json"
    if not review_path.exists():
        runner.old.save(review_path,reviews)
    runner.old.save(out/"summary.json",dict(total=len(inputs),eligible=len(eligible),
        ready=sum(r["status"] in ("ready","ready_empty") for r in reviews),
        failed=sum(r["status"]=="failed" for r in reviews),reviewed=0))


if __name__=="__main__":
    main()
