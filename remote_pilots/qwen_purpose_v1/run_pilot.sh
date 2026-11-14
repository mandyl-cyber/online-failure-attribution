#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${MODEL_ID:?Set MODEL_ID}"
: "${TOKENIZER_DIR:?Set TOKENIZER_DIR}"
: "${CONTEXT_TOKENS:?Set CONTEXT_TOKENS}"
command_name="${1:-preflight}"
if [ "$#" -gt 0 ]; then shift; fi
pilot_args=("$command_name" --package . --out "${RESULTS_DIR:-results}" --model "$MODEL_ID" --tokenizer "$TOKENIZER_DIR" --context-tokens "$CONTEXT_TOKENS" --base-url "${BASE_URL:-http://127.0.0.1:8000/v1}")
if [ -n "${CHAT_TEMPLATE_KWARGS:-}" ]; then pilot_args+=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS"); fi
"${PYTHON_BIN:-python}" remote.py "${pilot_args[@]}" "$@"
