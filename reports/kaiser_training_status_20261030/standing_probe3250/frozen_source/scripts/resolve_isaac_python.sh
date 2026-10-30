#!/usr/bin/env bash
# Shared bootstrap for train/play; probe package locations without starting Kit.
wheeled_prepare_native_loader() {
    local task_wsl_lib="${WHEELED_RL_WSL_LIB_DIR:-/usr/lib/wsl/lib}"
    local task_runtime_lib="${WHEELED_RL_RUNTIME_LIB_DIR:-${HOME}/.local/share/wheeled-rl/runtime-libs/usr/lib/x86_64-linux-gnu}"
    local -a task_library_dirs=()
    local task_library_dir
    # WSL's ldconfig cache exposes libcuda.so.1, whereas PhysX dlopens
    # libcuda.so. Torch CUDA can work while PhysX silently falls back to CPU.
    if [[ -f "$task_wsl_lib/libcuda.so" ]]; then
        task_library_dirs+=("$task_wsl_lib")
        if [[ -x "$task_wsl_lib/nvidia-smi" ]]; then
            case ":${PATH:-}:" in
                *":$task_wsl_lib:"*) ;;
                *) export PATH="$task_wsl_lib${PATH:+:$PATH}" ;;
            esac
        fi
    fi
    # Optional Ubuntu runtime packages extracted without sudo; never include a
    # CUDA toolkit stubs directory or replace the actual WSL driver library.
    if [[ -d "$task_runtime_lib" ]]; then task_library_dirs+=("$task_runtime_lib"); fi
    for task_library_dir in "${task_library_dirs[@]}"; do
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$task_library_dir:"*) ;;
            *) export LD_LIBRARY_PATH="$task_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    done
}

wheeled_resolve_python() {
    wheeled_prepare_native_loader
    local task_requested="${WHEELED_RL_PYTHON:-}" task_explicit=false task_argument
    local task_cli_python=false task_expect_python=false
    if [[ -n "$task_requested" ]]; then task_explicit=true; fi
    for task_argument in "$@"; do
        if [[ "$task_expect_python" == true ]]; then
            if [[ -z "$task_argument" || "$task_argument" == --* ]]; then
                echo "--python requires a Python executable or launcher path." >&2
                return 2
            fi
            task_requested="$task_argument"
            task_expect_python=false
            task_cli_python=true
        else
            case "$task_argument" in
                --python) task_expect_python=true ;;
                --python=*) task_requested="${task_argument#--python=}"; task_cli_python=true ;;
            esac
        fi
    done
    if [[ "$task_expect_python" == true || ( "$task_cli_python" == true && -z "$task_requested" ) ]]; then
        echo "--python requires a Python executable or launcher path." >&2
        return 2
    fi
    if [[ "$task_explicit" == true || "$task_cli_python" == true ]]; then
        if task_python="$(command -v -- "$task_requested")" && [[ -f "$task_python" && -x "$task_python" ]]; then
            return 0
        fi
        echo "Isaac Python is unavailable: $task_requested" >&2
        echo "Use --python /path/to/isaac/python (or set WHEELED_RL_PYTHON)." >&2
        return 127
    fi

    local -a task_candidates=()
    local task_candidate task_resolved task_env_root task_env_name task_image_python=""
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then task_candidates+=("$VIRTUAL_ENV/bin/python"); fi
    if [[ -n "${CONDA_PREFIX:-}" ]]; then task_candidates+=("$CONDA_PREFIX/bin/python"); fi
    # The container image needs its loader bootstrap before its venv Python.
    if [[ -f "${WHEELED_ISAACSIM_ROOT:-/isaac-sim}/setup_python_env.sh" &&
          -x "${WHEELED_ISAAC_PYTHON:-/opt/isaaclab-venv/bin/python}" ]]; then
        task_image_python="${WHEELED_ISAAC_PYTHON:-/opt/isaaclab-venv/bin/python}"
        task_candidates+=("$task_root/scripts/server_isaac_python.sh")
    fi
    task_candidates+=(python python3 "${HOME}/robot-rl-sim60/env/bin/python")
    for task_env_root in "${HOME}/miniconda3" "${HOME}/miniforge3" "${HOME}/anaconda3"; do
        for task_env_name in isaacsim6 isaaclab; do
            task_candidates+=("$task_env_root/envs/$task_env_name/bin/python")
        done
    done
    local -A task_seen=()
    for task_candidate in "${task_candidates[@]}"; do
        if ! task_resolved="$(command -v -- "$task_candidate")"; then continue; fi
        if [[ -n "$task_image_python" && "$task_resolved" == "$task_image_python" ]]; then
            task_resolved="$task_root/scripts/server_isaac_python.sh"
        fi
        if [[ ! -f "$task_resolved" || ! -x "$task_resolved" || -n "${task_seen[$task_resolved]:-}" ]]; then continue; fi
        task_seen["$task_resolved"]=1
        if "$task_resolved" -c 'from importlib.util import find_spec; import sys; sys.exit(0 if all(find_spec(name) is not None for name in ("isaaclab", "isaacsim", "torch", "rsl_rl")) else 1)' >/dev/null 2>&1; then
            task_python="$task_resolved"
            echo "Isaac Python: $task_python" >&2
            return 0
        fi
    done
    echo "No Isaac Python found in the active environment, PATH or known installation directories." >&2
    echo "Activate your Isaac environment, use --python /path/to/isaac/python, or set WHEELED_RL_PYTHON." >&2
    return 127
}
