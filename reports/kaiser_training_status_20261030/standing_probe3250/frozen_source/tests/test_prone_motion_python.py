"""Interpreter bootstrap portability without starting Isaac, CUDA or a learner."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESOLVER = ROOT / "scripts/resolve_isaac_python.sh"
BASH = "/usr/bin/bash"


def executable(path, *, available=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/usr/bin/bash\nexit {0 if available else 1}\n")
    path.chmod(0o755)
    return path


def environment(tmp_path):
    env = dict(os.environ)
    for key in ("WHEELED_RL_PYTHON", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(key, None)
    env.update(HOME=str(tmp_path / "home"), PATH=str(tmp_path / "bin"),
               WHEELED_ISAACSIM_ROOT=str(tmp_path / "absent-sim"),
               WHEELED_RL_WSL_LIB_DIR=str(tmp_path / "absent-wsl"),
               WHEELED_RL_RUNTIME_LIB_DIR=str(tmp_path / "absent-runtime"),
               WHEELED_ISAAC_PYTHON=str(tmp_path / "absent-image-python"))
    return env


def resolve(env, *args, root=ROOT):
    return subprocess.run([BASH, "-euc",
        'task_root="$1"; source "$2"; shift 2; wheeled_resolve_python "$@"; printf "%s\\n" "$task_python"',
        "resolver-test", str(root), str(RESOLVER), *map(str, args)],
        env=env, text=True, capture_output=True)


@pytest.mark.parametrize("equal", [False, True])
def test_cli_python_overrides_environment_and_preserves_spaces(tmp_path, equal):
    env = environment(tmp_path)
    env["WHEELED_RL_PYTHON"] = str(tmp_path / "missing")
    python = executable(tmp_path / "Isaac environment/bin/python")
    args = ["--python=" + str(python)] if equal else ["--python", str(python)]
    result = resolve(env, *args, "--run_name", "manual")
    assert result.returncode == 0 and result.stdout.strip() == str(python)


def test_environment_python_is_explicit_and_does_not_require_a_probe(tmp_path):
    env = environment(tmp_path)
    python = executable(tmp_path / "custom launcher", available=False)
    env["WHEELED_RL_PYTHON"] = str(python)
    result = resolve(env)
    assert result.returncode == 0 and result.stdout.strip() == str(python)


@pytest.mark.parametrize("args", [("--python",), ("--python=",), ("--python", "--headless")])
def test_missing_python_value_fails_before_any_launch(tmp_path, args):
    result = resolve(environment(tmp_path), *args)
    assert result.returncode == 2 and "--python requires" in result.stderr


def test_invalid_explicit_python_never_silently_uses_another_environment(tmp_path):
    env = environment(tmp_path)
    executable(Path(env["PATH"]) / "python")
    result = resolve(env, "--python", tmp_path / "missing")
    assert result.returncode == 127 and "unavailable" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("active", ["VIRTUAL_ENV", "CONDA_PREFIX"])
def test_active_isaac_environment_precedes_path(tmp_path, active):
    env = environment(tmp_path)
    env[active] = str(tmp_path / "active")
    python = executable(Path(env[active]) / "bin/python")
    executable(Path(env["PATH"]) / "python")
    result = resolve(env)
    assert result.returncode == 0 and result.stdout.strip() == str(python)


@pytest.mark.parametrize("relative", ["robot-rl-sim60/env/bin/python", "miniconda3/envs/isaacsim6/bin/python"])
def test_known_user_installation_skips_python_without_isaac(tmp_path, relative):
    env = environment(tmp_path)
    executable(Path(env["PATH"]) / "python", available=False)
    executable(Path(env["PATH"]) / "python3", available=False)
    python = executable(Path(env["HOME"]) / relative)
    result = resolve(env)
    assert result.returncode == 0 and result.stdout.strip() == str(python)
    assert result.stderr.strip() == "Isaac Python: " + str(python)


def test_container_venv_uses_its_loader_bootstrap_even_when_activated(tmp_path):
    env = environment(tmp_path)
    sim = tmp_path / "isaac-sim"
    sim.mkdir()
    (sim / "setup_python_env.sh").touch()
    env["WHEELED_ISAACSIM_ROOT"] = str(sim)
    env["VIRTUAL_ENV"] = str(tmp_path / "image-env")
    env["WHEELED_ISAAC_PYTHON"] = str(executable(Path(env["VIRTUAL_ENV"]) / "bin/python"))
    launcher = executable(tmp_path / "checkout/scripts/server_isaac_python.sh")
    result = resolve(env, root=tmp_path / "checkout")
    assert result.returncode == 0 and result.stdout.strip() == str(launcher)


def test_missing_isaac_environment_explains_both_overrides(tmp_path):
    result = resolve(environment(tmp_path))
    assert result.returncode == 127
    assert "--python" in result.stderr and "WHEELED_RL_PYTHON" in result.stderr


@pytest.mark.parametrize("entry", ["train", "play"])
def test_wsl_driver_and_user_runtime_reach_python_without_replacing_caller_paths(tmp_path, entry):
    env = environment(tmp_path)
    env["PATH"] = "/usr/bin:/bin"
    wsl = tmp_path / "WSL libraries"
    wsl.mkdir()
    (wsl / "libcuda.so").touch()
    executable(wsl / "nvidia-smi")
    runtime = tmp_path / "runtime libraries"
    runtime.mkdir()
    env.update(WHEELED_RL_WSL_LIB_DIR=str(wsl), WHEELED_RL_RUNTIME_LIB_DIR=str(runtime),
               LD_LIBRARY_PATH="/existing/lib", PYTHON_CALL_LOG=str(tmp_path / "loader"))
    python = executable(tmp_path / "python")
    python.write_text('#!/usr/bin/bash\nprintf "%s\\n%s\\n" "$LD_LIBRARY_PATH" "$PATH" > "$PYTHON_CALL_LOG"\n')
    result = subprocess.run([BASH, str(ROOT / f"scripts/{entry}_prone_motion.sh"),
        "--python", str(python), "--help"], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "loader").read_text().splitlines() == [
        f"{runtime}:{wsl}:/existing/lib", f"{wsl}:/usr/bin:/bin"]


def test_native_loader_is_idempotent_and_does_not_add_missing_driver(tmp_path):
    env = environment(tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    env.update(WHEELED_RL_RUNTIME_LIB_DIR=str(runtime), LD_LIBRARY_PATH="/existing/lib")
    result = subprocess.run([BASH, "-euc",
        'source "$1"; wheeled_prepare_native_loader; wheeled_prepare_native_loader; printf "%s" "$LD_LIBRARY_PATH"',
        "loader-test", str(RESOLVER)], env=env, text=True, capture_output=True)
    assert result.returncode == 0 and result.stdout == f"{runtime}:/existing/lib"


@pytest.mark.parametrize("entry,flag", [("train", "--print-output"), ("play", "--help")])
def test_train_and_play_forward_cli_and_arguments_to_selected_python(tmp_path, entry, flag):
    env = environment(tmp_path)
    env["PATH"] = "/usr/bin:/bin"
    python = executable(tmp_path / "custom python")
    python.write_text('#!/usr/bin/bash\nprintf "%s\\0" "$@" > "$PYTHON_CALL_LOG"\n')
    log = tmp_path / "args"
    env["PYTHON_CALL_LOG"] = str(log)
    result = subprocess.run([BASH, str(ROOT / f"scripts/{entry}_prone_motion.sh"),
        "--python", str(python), flag], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert log.read_bytes().split(b"\0")[:-1] == [
        str(ROOT / f"scripts/{entry}_prone_motion.py").encode(), b"--python", str(python).encode(), flag.encode()]
