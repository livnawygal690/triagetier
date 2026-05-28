#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-main}"
shift || true

python -m triagetier.toolchain.run "experiment=${EXPERIMENT}" "$@"
