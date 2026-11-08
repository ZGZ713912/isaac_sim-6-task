"""Learning-state checkpoints for a clean, process-exiting RSL-RL boundary.

Cold resume deliberately resets PhysX episodes and observation histories. It
preserves the learner and its clock; it does not claim bitwise physics replay.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile


FORMAT = "wheelbipe-rsl-durable-v1"


def fingerprint(value):
    import numpy as np
    import torch

    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous()
            digest.update(f"tensor:{array.dtype}:{tuple(array.shape)}".encode())
            digest.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(f"numpy:{item.dtype}:{item.shape}".encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(f"{type(item).__name__}:{len(item)}".encode())
            for child in item:
                visit(child)
        else:
            digest.update(f"{type(item).__name__}:{item!r};".encode())

    visit(value)
    return digest.hexdigest()


def rng_state():
    import numpy as np
    import torch

    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    import numpy as np
    import torch

    if len(state["cuda"]) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("CUDA RNG device count differs from the saved learning state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _normalizers(runner):
    return {name: getattr(runner, name).state_dict()
            for name in ("obs_normalizer", "critic_obs_normalizer")
            if hasattr(getattr(runner, name, None), "state_dict")}


def _finite(value):
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        import math
        return math.isfinite(value)
    return True


def capture(runner, env, completed, session_sha256, budget):
    if getattr(runner, "is_distributed", False) or getattr(runner.alg, "rnd", None):
        raise ValueError("Durable control supports single-process PPO without RND")
    if getattr(getattr(runner.alg, "storage", None), "step", 0) != 0:
        raise ValueError("A cold checkpoint requires an empty rollout after a complete PPO update")
    state = dict(format=FORMAT, session_sha256=session_sha256, completed_updates=int(completed),
                 total_updates=int(budget), next_iteration=int(completed),
                 iter=max(0, int(completed) - 1), infos={"managed_learning_checkpoint": True},
                 model_state_dict=runner.alg.policy.state_dict(),
                 optimizer_state_dict=runner.alg.optimizer.state_dict(),
                 learning_rate=float(runner.alg.learning_rate), normalizers=_normalizers(runner),
                 rng=rng_state(), runner_counters={name: getattr(runner, name)
                    for name in ("tot_timesteps", "tot_time") if hasattr(runner, name)},
                 training_clock=dict(completed_updates=int(completed),
                    common_step_counter=int(env.common_step_counter),
                    extrapolated_iteration=int(env._get_training_iteration())),
                 physical_episode_restored=False, observation_history_restored=False,
                 resume_semantics="learning state restored; new physical episodes and observation histories")
    if not 0 <= completed <= budget or state["training_clock"]["extrapolated_iteration"] != completed:
        raise ValueError("PPO budget and curriculum clock disagree at the save boundary")
    if not _finite({key: state[key] for key in ("model_state_dict", "optimizer_state_dict", "normalizers", "learning_rate")}):
        raise ValueError("Nonfinite learner state cannot be committed for resume")
    return state


def state_fingerprints(state):
    return {key: fingerprint(state[key]) for key in
            ("model_state_dict", "optimizer_state_dict", "normalizers", "learning_rate", "rng", "runner_counters", "training_clock")}


def save_checkpoint(path, runner, env, completed, session_sha256, budget):
    import torch
    from .trainctl_shared.training_control import file_pin, write_json

    path = Path(path)
    state = capture(runner, env, completed, session_sha256, budget)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pt", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    metadata = dict(format=FORMAT, completed_updates=completed, total_updates=budget,
                    session_sha256=session_sha256, checkpoint_pins=[file_pin(path)],
                    learning_state_fingerprints=state_fingerprints(state),
                    physical_episode_restored=False, observation_history_restored=False,
                    resume_semantics=state["resume_semantics"])
    write_json(Path(str(path) + ".json"), metadata)
    return path


def load_checkpoint(path, runner, env, session_sha256, budget):
    import torch
    from .trainctl_shared.training_control import read_json, verify_pin

    path = Path(path).resolve(strict=True)
    metadata = read_json(Path(str(path) + ".json"))
    pins = metadata.get("checkpoint_pins", []) if isinstance(metadata, dict) else []
    if len(pins) != 1 or pins[0].get("path") != str(path):
        raise ValueError("Durable checkpoint payload is not pinned by its sidecar")
    verify_pin(pins[0])
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state.get("format") != FORMAT or state.get("session_sha256") != session_sha256
            or state.get("total_updates") != budget or metadata.get("session_sha256") != session_sha256
            or metadata.get("completed_updates") != state.get("completed_updates")
            or not 0 <= state["completed_updates"] < budget
            or state["next_iteration"] != state["completed_updates"]
            or state["training_clock"]["extrapolated_iteration"] != state["completed_updates"]
            or state_fingerprints(state) != metadata.get("learning_state_fingerprints")
            or not _finite({key: state[key] for key in ("model_state_dict", "optimizer_state_dict", "normalizers", "learning_rate")})):
        raise ValueError("Checkpoint format, budget, clock or learning-state integrity differs")
    runner.alg.policy.load_state_dict(state["model_state_dict"], strict=True)
    runner.alg.optimizer.load_state_dict(state["optimizer_state_dict"])
    runner.alg.learning_rate = state["learning_rate"]
    if set(_normalizers(runner)) != set(state["normalizers"]):
        raise ValueError("Normalizer layout differs from the stopped attempt")
    for name, value in state["normalizers"].items():
        getattr(runner, name).load_state_dict(value)
    for name, value in state["runner_counters"].items():
        setattr(runner, name, value)
    completed = state["completed_updates"]
    runner.current_learning_iteration = completed
    env.common_step_counter = state["training_clock"]["common_step_counter"]
    env.set_training_progress(iteration=completed)
    restore_rng(state["rng"])
    actual = capture(runner, env, completed, session_sha256, budget)
    verified = state_fingerprints(actual)
    if verified != metadata["learning_state_fingerprints"]:
        raise ValueError("Loaded learner differs from the exact stopped learning state")
    return dict(completed_updates=completed, checkpoint=str(path),
                learning_state_fingerprints=verified, exact_learning_state_verified=True,
                physical_episode_restored=False, observation_history_restored=False,
                budget_extended=False)
