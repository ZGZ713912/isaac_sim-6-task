#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_root"
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1
task_python="${WHEELED_RL_PYTHON:-/home/noir/miniconda3/envs/isaacsim6/bin/python}"
exec "$task_python" "$task_root/scripts/play_prone_motion.py" "$@"
