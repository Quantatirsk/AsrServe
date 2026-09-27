#!/usr/bin/env bash
# GPU: ./build.sh -> quantatrisk/qwen3-asr:gpu
# CPU for this host's architecture: TARGET=cpu ./build.sh -> quantatrisk/qwen3-asr:cpu
# Extra arguments go to buildx, e.g. TARGET=cpu ./build.sh --platform linux/arm64
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ "${TARGET:-gpu}" = cpu ]; then
  exec docker buildx build --load \
    -f Dockerfile.cpu -t quantatrisk/qwen3-asr:cpu "$@" .
fi
exec docker buildx build --platform linux/amd64 --load \
  -f Dockerfile.gpu -t quantatrisk/qwen3-asr:gpu "$@" .
