#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_root"
export OMNI_KIT_ACCEPT_EULA=YES
export OPENBLAS_NUM_THREADS=1
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
        --native-probe) task_probe_only=true; task_native_only=true ;;
        --preflight-only|--help|-h) task_probe_only=true ;;
    esac
done
verify_probe() {
    /home/noir/miniconda3/envs/isaacsim6/bin/python - "$task_root/scripts" "$1" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from train_prone_motion import verify_native_probe
verify_native_probe(Path(sys.argv[2]))
PY
}
if [[ "$task_probe_only" == false && -n "$task_output" ]]; then
    /home/noir/miniconda3/envs/isaacsim6/bin/python "$task_root/scripts/train_prone_motion.py" \
        "$@" --output "${task_output}.native_probe" --native-probe
    verify_probe "${task_output}.native_probe"
fi
if [[ "$task_native_only" == true ]]; then
    /home/noir/miniconda3/envs/isaacsim6/bin/python "$task_root/scripts/train_prone_motion.py" "$@"
    verify_probe "$task_output"
    exit 0
fi
exec /home/noir/miniconda3/envs/isaacsim6/bin/python "$task_root/scripts/train_prone_motion.py" "$@"
