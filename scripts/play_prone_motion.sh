#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_root"
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1
source "$task_root/scripts/resolve_isaac_python.sh"
wheeled_resolve_python "$@"
exec "$task_python" "$task_root/scripts/play_prone_motion.py" "$@"
