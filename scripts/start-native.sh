#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export DEVICE=cpu
export PORT=17003
exec "$ROOT_DIR/.venv/bin/python" -u start.py "$@"
