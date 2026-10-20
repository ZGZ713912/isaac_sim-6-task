#!/usr/bin/env bash
set -euo pipefail
PLAY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PLAY_PYTHON="${FLAT_PLAY_PYTHON:-/home/noir/miniconda3/envs/isaacsim6/bin/python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
cd -- "$PLAY_ROOT"
exec "$PLAY_PYTHON" "$PLAY_ROOT/scripts/play_flat_speed_preview.py" "$@"
