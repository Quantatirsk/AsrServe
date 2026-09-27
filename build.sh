#!/usr/bin/env bash
# CUDA: ./build.sh    CPU for this host's architecture: TARGET=cpu ./build.sh
# Extra arguments go to buildx, e.g. TARGET=cpu ./build.sh --platform linux/arm64
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ "${TARGET:-cuda}" = cpu ]; then
  exec docker buildx build --load \
    -f Dockerfile.cpu -t quantatrisk/qwen3-asr:latest "$@" .
fi
exec docker buildx build --platform linux/amd64 --load \
  -f Dockerfile.gpu -t quantatrisk/qwen3-asr:latest "$@" .
