"""Cross-course transfer preserves learning state without weakening resume."""
from copy import deepcopy

import pytest

from wheeled_algo import durable_checkpoint as checkpoint
from wheeled_algo.learning_transfer import SOURCE_SETTINGS, admission, initialize
from wheeled_algo.trainctl_shared.training_control import file_pin, write_json
from test_durable_training import Clock, learner, update


def source_point(tmp_path):
    root = tmp_path/"baseline"
    root.mkdir()
    arguments = [x for key, value in SOURCE_SETTINGS.items() for x in (key, value)]
    arguments += ["--no-external-forces-every-iteration"]
    write_json(root/"session.json", dict(updates=20000, training_args=arguments))
    write_json(root/"session.pin.json", file_pin(root/"session.json"))
    runner, env = learner(), Clock()
    update(runner, env)
    env.common_step_counter = 20000*24
    point = checkpoint.save_checkpoint(root/"attempts/a/train/checkpoints/final.pt",
        runner, env, 20000, file_pin(root/"session.json")["sha256"], 20000)
    target = ["foundation-jump-v1" if x == "foundation-v2" else x for x in arguments]
    target += ["--motion-update-offset", "20000"]
    return point, runner, target


def test_exact_transfer_then_exact_same_course_resume(tmp_path):
    import torch
    point, original, arguments = source_point(tmp_path)
    state = torch.load(point, weights_only=False)
    fresh, env = learner(), Clock()
    proof = initialize(point, fresh, env, arguments)
    assert proof["source_completed_updates"] == 20000
    assert proof["trial_completed_updates"] == fresh.current_learning_iteration == 0
    assert env.common_step_counter == 0 and fresh.tot_timesteps == 0
    for key in ("model_state_dict", "optimizer_state_dict", "normalizers", "learning_rate", "rng"):
        assert proof["exact_loaded_components"][key] == checkpoint.fingerprint(state[key])
    assert fresh.alg.optimizer.state and fresh.alg.learning_rate == original.alg.learning_rate
    update(fresh, env)
    saved = checkpoint.save_checkpoint(tmp_path/"trial.pt", fresh, env, 1, "new-course", 10)
    restored, other_env = learner(), Clock()
    checkpoint.load_checkpoint(saved, restored, other_env, "new-course", 10)
    assert checkpoint.fingerprint(restored.alg.optimizer.state_dict()) == checkpoint.fingerprint(fresh.alg.optimizer.state_dict())
    assert restored.current_learning_iteration == 1


@pytest.mark.parametrize("setting,value", [("--physics-hz", "1000"),
    ("--training-profile", "foundation-v2"), ("--motion-update-offset", "0"),
    ("--randomization-strength", "0"), ("--num-envs", "512")])
def test_mismatched_transfer_contract_rejected(tmp_path, setting, value):
    point, _, arguments = source_point(tmp_path)
    arguments[arguments.index(setting)+1] = value
    with pytest.raises(ValueError, match="Continuation"):
        admission(point, arguments)


def test_payload_change_and_wrong_state_fingerprint_rejected(tmp_path):
    import torch
    point, _, arguments = source_point(tmp_path)
    data = torch.load(point, weights_only=False)
    data["learning_rate"] *= 2
    torch.save(data, point)
    with pytest.raises(ValueError, match="completed 20000"):
        admission(point, arguments)
    import json
    sidecar = point.with_suffix(".pt.json")
    metadata = json.loads(sidecar.read_text())
    metadata["checkpoint_pins"] = [file_pin(point)]
    write_json(sidecar, metadata, replace=True)
    fresh, env = learner(), Clock()
    with pytest.raises(ValueError, match="integrity"):
        initialize(point, fresh, env, arguments)
    assert not fresh.alg.optimizer.state
