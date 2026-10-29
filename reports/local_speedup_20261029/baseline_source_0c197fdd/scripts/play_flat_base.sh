#!/usr/bin/env bash
set -euo pipefail
PLAY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PLAY_PYTHON="${FLAT_PLAY_PYTHON:-/home/noir/miniconda3/envs/isaacsim6/bin/python}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
cd -- "$PLAY_ROOT"
exec "$PLAY_PYTHON" "$PLAY_ROOT/scripts/play_flat_base.py" "$@"
