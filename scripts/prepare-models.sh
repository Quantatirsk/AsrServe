#!/usr/bin/env bash
# Download the pinned models into ./models with an image built by build.sh.
# No GPU is needed. CPU image: IMAGE=quantatrisk/qwen3-asr:cpu ./scripts/prepare-models.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p models
exec docker run --rm -e HF_ENDPOINT -v "$PWD/models:/app/models" \
  "${IMAGE:-quantatrisk/qwen3-asr:gpu}" --download-models
