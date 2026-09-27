#!/usr/bin/env bash
# GPU: ./build.sh -> quantatrisk/asrserve:gpu
# CPU for this host's architecture: TARGET=cpu ./build.sh -> quantatrisk/asrserve:cpu
# Ascend 910B on the NPU host: TARGET=ascend ./build.sh -> quantatrisk/asrserve:ascend
# Extra arguments go to buildx, e.g. TARGET=cpu ./build.sh --platform linux/arm64
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ "${TARGET:-gpu}" = ascend ]; then
  exec docker buildx build --load \
    -f Dockerfile.ascend -t quantatrisk/asrserve:ascend "$@" .
fi
if [ "${TARGET:-gpu}" = cpu ]; then
  exec docker buildx build --load \
    -f Dockerfile.cpu -t quantatrisk/asrserve:cpu "$@" .
fi
exec docker buildx build --platform linux/amd64 --load \
  -f Dockerfile.gpu -t quantatrisk/asrserve:gpu "$@" .
