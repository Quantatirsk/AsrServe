#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export NEMOTRON_MODEL_PATH="${NEMOTRON_MODEL_PATH:-$PWD/models/nemotron-3-diarization}"
export HF_HOME="${HF_HOME:-$PWD/models/huggingface}"
UV_ARGS=(--frozen)
if [ "$(uname -s)" = Linux ]; then
  if [ "${DEVICE:-cuda:0}" = cpu ] || [ "${DEVICE:-cuda:0}" = npu:0 ]; then
    UV_ARGS+=(--extra cpu)
  else
    UV_ARGS+=(--extra cuda)
  fi
fi
exec uv run "${UV_ARGS[@]}" python -m app.utils.download_models "$@"
