#!/usr/bin/env bash
# Download the pinned models into ./models with the service image.
# No GPU is needed. CPU image: IMAGE=quantatrisk/asrserve:cpu ./scripts/prepare-models.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p models
exec docker run --rm -e HF_ENDPOINT -v "$PWD/models:/app/models" \
  "${IMAGE:-quantatrisk/asrserve:gpu}" --download-models
