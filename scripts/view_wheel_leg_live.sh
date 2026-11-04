#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_root"
export OMNI_KIT_ACCEPT_EULA=YES
export OPENBLAS_NUM_THREADS=1
source "$task_root/scripts/resolve_isaac_python.sh"
wheeled_resolve_python "$@"
# The pure viewport needs cached renderer extensions. Some local Isaac 6
# training environments contain only core packages and cannot open a GUI.
task_explicit=false
if [[ -n "${WHEELED_RL_PYTHON:-}" ]]; then task_explicit=true; fi
for task_argument in "$@"; do
    if [[ "$task_argument" == --python || "$task_argument" == --python=* ]]; then task_explicit=true; fi
done
if [[ "$task_explicit" == false ]]; then
    task_render_probe='from importlib.util import find_spec; from pathlib import Path; import sys; p=Path(find_spec("isaacsim").origin).parent; sys.exit(0 if any(any((p/d).glob("omni.hydra.rtx*")) for d in ("extscache","exts","kit/exts")) else 1)'
    if ! "$task_python" -c "$task_render_probe" >/dev/null 2>&1; then
        for task_candidate in "$HOME/miniconda3/envs/isaaclab/bin/python" "$HOME/miniforge3/envs/isaaclab/bin/python"; do
            if [[ -x "$task_candidate" ]] && "$task_candidate" -c "$task_render_probe" >/dev/null 2>&1; then
                task_python="$task_candidate"
                echo "Mirror GUI Python (cached renderer): $task_python" >&2
                break
            fi
        done
    fi
fi
exec "$task_python" "$task_root/scripts/view_wheel_leg_live.py" "$@"
