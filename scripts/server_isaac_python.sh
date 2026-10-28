#!/usr/bin/env bash
# Python launcher for the server image's bundled Isaac Sim and Isaac Lab venv.
set -euo pipefail
task_sim_root="${WHEELED_ISAACSIM_ROOT:-/isaac-sim}"
task_sim_python="${WHEELED_ISAAC_PYTHON:-/opt/isaaclab-venv/bin/python}"
if [[ ! -f "$task_sim_root/setup_python_env.sh" || ! -x "$task_sim_python" ]]; then
    echo "Set WHEELED_ISAACSIM_ROOT and WHEELED_ISAAC_PYTHON to the server installation." >&2
    exit 127
fi
export CARB_APP_PATH="$task_sim_root/kit"
export ISAAC_PATH="$task_sim_root"
export EXP_PATH="$task_sim_root/apps"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source "$task_sim_root/setup_python_env.sh"
# The image puts the system Python 3.12.3 library before bundled Python 3.12.14.
export LD_LIBRARY_PATH="$task_sim_root/kit/python/lib:$LD_LIBRARY_PATH"
# Kit's pip bundles must not shadow the repaired training environment packages.
task_sim_site="$("$task_sim_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export PYTHONPATH="$task_sim_site${PYTHONPATH:+:$PYTHONPATH}"
# Torch lazily dlopens libtorch_cuda_linalg.so by name for the inertia eigensolver.
export LD_LIBRARY_PATH="$task_sim_site/torch/lib:$LD_LIBRARY_PATH"
# Match NVIDIA's python.sh bootstrap and retain any caller-provided preloads.
export LD_PRELOAD="$task_sim_root/kit/libcarb.so${LD_PRELOAD:+:$LD_PRELOAD}"
exec "$task_sim_python" "$@"
