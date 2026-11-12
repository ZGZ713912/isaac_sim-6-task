"""Explicit cross-course continuation, separate from exact same-course resume."""
from __future__ import annotations

from pathlib import Path

from . import durable_checkpoint as durable
from .trainctl_shared.training_control import file_pin, read_json, verify_pin


SOURCE_SETTINGS = {
    "--task": "flat", "--leg-target-projection-penalty": "1",
    "--training-profile": "foundation-v2", "--physics-hz": "200",
    "--num-envs": "1024", "--leg-kp": "60", "--leg-kd": "2",
    "--leg-armature": ".015795", "--solver-position-iterations": "64",
    "--solver-velocity-iterations": "32", "--randomization-profile": "requested",
    "--randomization-strength": "1",
}


def _option(arguments, key):
    values = [arguments[i+1] for i, value in enumerate(arguments[:-1]) if value == key]
    values += [value.split("=", 1)[1] for value in arguments if value.startswith(key+"=")]
    if len(values) != 1:
        raise ValueError("Continuation requires one explicit source setting: "+key)
    return values[0]


def admission(path, target_arguments):
    """Pin the completed source session and full checkpoint before launch."""
    path = Path(path).resolve(strict=True)
    sidecar = Path(str(path)+".json")
    metadata = read_json(sidecar)
    if (not metadata or metadata.get("format") != durable.FORMAT
            or metadata.get("completed_updates") != 20000
            or metadata.get("total_updates") != 20000
            or metadata.get("checkpoint_pins") != [file_pin(path)]):
        raise ValueError("Continuation requires the completed 20000-update full learning checkpoint")
    parents = [p for p in path.parents if (p/"session.pin.json").is_file()]
    if not parents:
        raise ValueError("Continuation checkpoint has no pinned source session")
    root = parents[0]
    verify_pin(read_json(root/"session.pin.json"))
    source = read_json(root/"session.json")
    if (file_pin(root/"session.json")["sha256"] != metadata["session_sha256"]
            or source.get("updates") != 20000):
        raise ValueError("Continuation source session identity differs from checkpoint")
    for key, expected in SOURCE_SETTINGS.items():
        value = _option(source["training_args"], key)
        if value != expected:
            raise ValueError("Continuation source physics/course differs: "+key)
        target = _option(target_arguments, key)
        if key == "--training-profile":
            expected = "foundation-jump-v1"
        if target != expected:
            raise ValueError("Continuation target changes an unsupported setting: "+key)
    if any("--no-external-forces-every-iteration" not in args or "--gas-feedforward" in args
           for args in (source["training_args"], target_arguments)):
        raise ValueError("Continuation must retain the original external-force integration")
    if _option(target_arguments, "--motion-update-offset") != "20000":
        raise ValueError("Continuation must retain the completed motion curriculum")
    return dict(checkpoint=str(path), source_completed_updates=20000,
        source_session=str(root), source_session_sha256=metadata["session_sha256"],
        pins=[file_pin(path), file_pin(sidecar), file_pin(root/"session.json"),
              file_pin(root/"session.pin.json")])


def initialize(path, runner, env, target_arguments):
    """Keep actor/critic/Adam/LR/normalizers/RNG, start a new trial clock."""
    import torch

    proof = admission(path, target_arguments)
    state = torch.load(path, map_location="cpu", weights_only=False)
    metadata = read_json(Path(str(path)+".json"))
    if (state.get("format") != durable.FORMAT or state.get("completed_updates") != 20000
            or state.get("total_updates") != 20000
            or state.get("session_sha256") != proof["source_session_sha256"]
            or state.get("next_iteration") != 20000
            or state.get("training_clock", {}).get("extrapolated_iteration") != 20000
            or durable.state_fingerprints(state) != metadata["learning_state_fingerprints"]
            or not durable._finite(state)):
        raise ValueError("Continuation source learning-state integrity differs")
    if not state["optimizer_state_dict"]["state"] or runner.alg.optimizer.state:
        raise ValueError("Continuation needs saved Adam and an unmodified target learner")
    if set(durable._normalizers(runner)) != set(state["normalizers"]):
        raise ValueError("Continuation normalizer layout differs")
    runner.alg.policy.load_state_dict(state["model_state_dict"], strict=True)
    runner.alg.optimizer.load_state_dict(state["optimizer_state_dict"])
    runner.alg.learning_rate = state["learning_rate"]
    for name, value in state["normalizers"].items():
        getattr(runner, name).load_state_dict(value)
    durable.restore_rng(state["rng"])
    actual = dict(model_state_dict=runner.alg.policy.state_dict(),
        optimizer_state_dict=runner.alg.optimizer.state_dict(), normalizers=durable._normalizers(runner),
        learning_rate=runner.alg.learning_rate, rng=durable.rng_state())
    verified = {key: durable.fingerprint(value) for key, value in actual.items()}
    if any(verified[key] != metadata["learning_state_fingerprints"][key] for key in verified):
        raise ValueError("Continuation did not restore exact source learning tensors/RNG")
    runner.current_learning_iteration = 0
    runner.tot_timesteps, runner.tot_time = 0, 0.
    env.common_step_counter = 0
    env.set_training_progress(iteration=0)
    proof.update(exact_loaded_components=verified, source_runner_counters=state["runner_counters"],
        trial_completed_updates=0, motion_update_offset=20000, jump_difficulty_update=0,
        physical_episode_restored=False, observation_history_restored=False,
        target_curriculum_rng="new independent seeded stream", same_course_resume=False,
        semantics="explicit cross-course learning continuation; new trial budget and physical episodes")
    return proof
