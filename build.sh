#!/usr/bin/env bash
# GPU: ./build.sh -> quantatrisk/qwen3-asr:gpu
# CPU for this host's architecture: TARGET=cpu ./build.sh -> quantatrisk/qwen3-asr:cpu
# Ascend 910B on the NPU host: TARGET=ascend ./build.sh -> quantatrisk/qwen3-asr:ascend
# Extra arguments go to buildx, e.g. TARGET=cpu ./build.sh --platform linux/arm64
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ "${TARGET:-gpu}" = ascend ]; then
  exec docker buildx build --load \
    -f Dockerfile.ascend -t quantatrisk/qwen3-asr:ascend "$@" .
fi
if [ "${TARGET:-gpu}" = cpu ]; then
  exec docker buildx build --load \
    -f Dockerfile.cpu -t quantatrisk/qwen3-asr:cpu "$@" .
fi
exec docker buildx build --platform linux/amd64 --load \
  -f Dockerfile.gpu -t quantatrisk/qwen3-asr:gpu "$@" .
