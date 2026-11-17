#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MODEL="$ROOT/vendor/models/Qwen2.5-32B-Instruct-AWQ"
OUT="$ROOT/eval/swe_purpose_qwen32_awq_v1"
CLIENT=/home/sjtu/miniconda3/envs/agentforesight/bin/python
VLLM="$ROOT/vendor/.venv-vllm/bin/vllm"
COMPARE="$ROOT/vendor/.venv-compare/bin/python"
PORT=30032
mkdir -p "$OUT"

"$VLLM" serve "$MODEL" \
  --served-model-name qwen32-awq-local \
  --host 127.0.0.1 --port "$PORT" \
  --dtype float16 --max-model-len 8192 \
  --gpu-memory-utilization 0.92 --max-num-seqs 1 \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" > "$OUT/server_models.json" 2>/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -80 "$OUT/server.log"
    exit 1
  fi
  sleep 5
done
if [[ "$ready" != 1 ]]; then
  tail -80 "$OUT/server.log"
  exit 1
fi

"$VLLM" --version > "$OUT/vllm_version.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$OUT/gpu.txt"

for split in train dev; do
  package="remote_pilots/swe_purpose_v2/$split"
  "$CLIENT" "$package/remote.py" preflight --package "$package" \
    --out "$OUT/$split" --tokenizer "$MODEL" --context-tokens 8192 \
    > "$OUT/${split}_preflight.log" 2>&1
done

"$CLIENT" remote_pilots/swe_purpose_v2/train/remote.py run \
  --package remote_pilots/swe_purpose_v2/train --out "$OUT/pilot_train5" \
  --tokenizer "$MODEL" --context-tokens 8192 \
  --model qwen32-awq-local --base-url "http://127.0.0.1:$PORT/v1" \
  --limit 5 > "$OUT/pilot_train5.log" 2>&1

"$CLIENT" - <<'PY'
import json
from pathlib import Path
root=Path('eval/swe_purpose_qwen32_awq_v1/pilot_train5/responses')
expected=[f'TRAIN_001_s{i:05d}' for i in range(1,6)]
missing=[rid for rid in expected if not (root/(rid+'.json')).exists()]
if missing:
    raise SystemExit('Five-step trial has invalid/missing responses: '+str(missing))
print('Five-step trial passed: 5/5',flush=True)
PY

for split in train dev; do
  package="remote_pilots/swe_purpose_v2/$split"
  "$CLIENT" "$package/remote.py" run --package "$package" \
    --out "$OUT/pilot_${split}_representative" \
    --tokenizer "$MODEL" --context-tokens 8192 \
    --model qwen32-awq-local --base-url "http://127.0.0.1:$PORT/v1" \
    --request-ids-file "pilot/swe_purpose_qwen_v2/$split/representative_ids.txt" \
    > "$OUT/pilot_${split}_representative.log" 2>&1
done
"$CLIENT" - <<'PY'
from pathlib import Path
base=Path('pilot/swe_purpose_qwen_v2')
out=Path('eval/swe_purpose_qwen32_awq_v1')
for split in ('train','dev'):
    ids=(base/split/'representative_ids.txt').read_text().splitlines()
    responses=out/f'pilot_{split}_representative'/'responses'
    valid=sum((responses/(rid+'.json')).exists() for rid in ids)
    print(f'{split} representative trial: {valid}/{len(ids)}',flush=True)
    if valid < len(ids)*0.9:
        raise SystemExit('Representative trial below 90% valid; inspect raw errors before full run')
PY

# Prioritize the frozen AI audit sample, then resume the same run on every step.
"$CLIENT" - <<'PY'
import json
from pathlib import Path
base=Path('pilot/swe_purpose_qwen_v2')
out=Path('eval/swe_purpose_qwen32_awq_v1')
for split in ('train','dev'):
    sample=json.loads((base/split/'codex_blind_audit_v1'/'selection.json').read_text())
    ids=[item['request_id'] for item in sample['items']]
    (out/f'{split}_audit_ids.txt').write_text('\n'.join(ids)+'\n')
PY
for split in train dev; do
  package="remote_pilots/swe_purpose_v2/$split"
  "$CLIENT" "$package/remote.py" run --package "$package" \
    --out "$OUT/$split" --tokenizer "$MODEL" --context-tokens 8192 \
    --model qwen32-awq-local --base-url "http://127.0.0.1:$PORT/v1" \
    --request-ids-file "$OUT/${split}_audit_ids.txt" \
    > "$OUT/${split}_audit_sample.log" 2>&1
done
"$COMPARE" -m segmenter.compare_qwen_purpose_runs \
  --qwen32 "$OUT" --out "$OUT/comparison_partial.json" \
  > "$OUT/comparison_partial.log" 2>&1

for split in train dev; do
  package="remote_pilots/swe_purpose_v2/$split"
  "$CLIENT" "$package/remote.py" run --package "$package" \
    --out "$OUT/$split" --tokenizer "$MODEL" --context-tokens 8192 \
    --model qwen32-awq-local --base-url "http://127.0.0.1:$PORT/v1" \
    > "$OUT/${split}_full.log" 2>&1
  "$CLIENT" "$package/remote.py" validate --package "$package" \
    --out "$OUT/$split" > "$OUT/${split}_validate.log" 2>&1
done

"$COMPARE" -m segmenter.compare_qwen_purpose_runs \
  --qwen32 "$OUT" --out "$OUT/comparison.json" \
  > "$OUT/comparison.log" 2>&1
cat "$OUT/comparison.log"
