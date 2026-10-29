#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_root"
export OMNI_KIT_ACCEPT_EULA=YES
export OPENBLAS_NUM_THREADS=1
# Keep the local default; remote installations select their own interpreter.
task_python="${WHEELED_RL_PYTHON:-/home/noir/miniconda3/envs/isaacsim6/bin/python}"
if ! command -v -- "$task_python" >/dev/null 2>&1; then
    echo "Training Python is unavailable: $task_python" >&2
    echo "Set WHEELED_RL_PYTHON to the Python executable in your Isaac environment." >&2
    exit 127
fi
# A normal launch runs a separate0-PPO native check first. Each output directory
# is new; a failed probe never falls through to PPO or retries a stopped run.
task_output=""
task_probe_only=false
task_native_only=false
task_args=("$@")
for ((task_i=0; task_i<${#task_args[@]}; task_i++)); do
    case "${task_args[task_i]}" in
        --output) task_output="${task_args[task_i+1]:-}" ;;
        --output=*) task_output="${task_args[task_i]#--output=}" ;;
        --native-probe|--native_probe) task_probe_only=true; task_native_only=true ;;
        --preflight-only|--preflight_only|--print-output|--help|-h) task_probe_only=true ;;
    esac
done
if [[ "$task_probe_only" == false || "$task_native_only" == true ]]; then
    task_output="$("$task_python" "$task_root/scripts/train_prone_motion.py" "$@" --print-output)"
    if [[ -e "$task_output" || ( "$task_native_only" == false && -e "${task_output}.native_probe" ) ]]; then
        echo "Output or native-probe directory already exists: $task_output" >&2
        echo "Choose a new --run_name or --output. Existing runs are never overwritten or resumed." >&2
        exit 2
    fi
fi
verify_probe() {
    "$task_python" - "$task_root/scripts" "$1" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from train_prone_motion import verify_native_probe
verify_native_probe(Path(sys.argv[2]))
PY
}
if [[ "$task_probe_only" == false && -n "$task_output" ]]; then
    "$task_python" "$task_root/scripts/train_prone_motion.py" \
        "$@" --output "${task_output}.native_probe" --native-probe
    verify_probe "${task_output}.native_probe"
fi
if [[ "$task_native_only" == true ]]; then
    "$task_python" "$task_root/scripts/train_prone_motion.py" "$@" --output "$task_output"
    verify_probe "$task_output"
    exit 0
fi
if [[ "$task_probe_only" == false ]]; then
    exec "$task_python" "$task_root/scripts/train_prone_motion.py" "$@" --output "$task_output"
fi
exec "$task_python" "$task_root/scripts/train_prone_motion.py" "$@"
