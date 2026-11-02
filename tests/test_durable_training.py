"""Real learning-state/IPC checks; all processes belong to these tests."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from wheeled_algo import durable_checkpoint as checkpoint
from wheeled_algo.managed_training import WheelbipeSession
from wheeled_algo.trainctl_shared.training_control import file_pin, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def learner():
    torch = pytest.importorskip("torch")
    policy = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    optimizer = torch.optim.Adam(policy.parameters(), lr=.003)
    normalizer = torch.nn.BatchNorm1d(3)
    return SimpleNamespace(alg=SimpleNamespace(policy=policy, optimizer=optimizer, learning_rate=.003,
               storage=SimpleNamespace(step=0), rnd=None), device="cpu", is_distributed=False,
               obs_normalizer=normalizer, current_learning_iteration=0, tot_timesteps=0, tot_time=0.)


class Clock:
    common_step_counter = 0

    def _get_training_iteration(self):
        return self.common_step_counter // 24

    def set_training_progress(self, iteration):
        assert self._get_training_iteration() == iteration


def update(runner, env):
    import numpy as np
    import torch

    observation = torch.randn(12, 3)
    target = torch.randn(12, 2) + random.random() + float(np.random.uniform())
    runner.alg.optimizer.zero_grad()
    loss = (runner.alg.policy(runner.obs_normalizer(observation)) - target).square().mean()
    loss.backward()
    runner.alg.optimizer.step()
    env.common_step_counter += 24
    runner.tot_timesteps += 12


def test_exact_learning_resume_with_adam_normalizer_lr_rng_and_clock(tmp_path):
    import numpy as np
    torch = pytest.importorskip("torch")
    torch.manual_seed(1001)
    np.random.seed(1001)
    random.seed(1001)
    runner, env = learner(), Clock()
    for _ in range(3):
        update(runner, env)
    runner.alg.learning_rate = .00125
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = .00125
    saved = checkpoint.save_checkpoint(tmp_path / "point.pt", runner, env, 3, "source", 6)
    expected_before = checkpoint.state_fingerprints(checkpoint.capture(runner, env, 3, "source", 6))
    for _ in range(3):
        update(runner, env)
    expected = checkpoint.state_fingerprints(checkpoint.capture(runner, env, 6, "source", 6))
    restored, fresh_env = learner(), Clock()
    proof = checkpoint.load_checkpoint(saved, restored, fresh_env, "source", 6)
    assert proof["exact_learning_state_verified"] and proof["learning_state_fingerprints"] == expected_before
    assert restored.current_learning_iteration == 3 and fresh_env.common_step_counter == 72
    assert restored.alg.optimizer.state and restored.alg.learning_rate == .00125
    for _ in range(3):
        update(restored, fresh_env)
    assert checkpoint.state_fingerprints(checkpoint.capture(restored, fresh_env, 6, "source", 6)) == expected


def test_real_wheelbipe_rsl301_ppo_cold_resume_matches_uninterrupted_learning(tmp_path):
    import importlib.metadata
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    if importlib.metadata.version("rsl-rl-lib") != "3.0.1":
        pytest.skip("Wheelbipe is explicitly pinned to RSL-RL3.0.1")
    from tensordict import TensorDict
    from rsl_rl.runners import OnPolicyRunner

    class Env(Clock):
        num_envs, num_actions, device, cfg = 4, 2, "cpu", {}

        def get_observations(self):
            return TensorDict(dict(policy=torch.full((4, 3), .1), critic=torch.full((4, 7), .2)), batch_size=[4])

        def step(self, action):
            self.common_step_counter += 1
            return self.get_observations(), -action.square().mean(-1), torch.zeros(4, dtype=torch.long), {}

    config = dict(num_steps_per_env=24, save_interval=500,
        obs_groups={"policy": ["policy"], "critic": ["critic"]}, logger="tensorboard",
        policy=dict(class_name="ActorCritic", init_noise_std=1., noise_std_type="scalar",
            actor_hidden_dims=[8], critic_hidden_dims=[8], activation="elu",
            actor_obs_normalization=True, critic_obs_normalization=True),
        algorithm=dict(class_name="PPO", num_learning_epochs=2, num_mini_batches=2,
            learning_rate=.001, schedule="adaptive", gamma=.99, lam=.95, entropy_coef=.005,
            desired_kl=.01, max_grad_norm=1., value_loss_coef=4., use_clipped_value_loss=True,
            clip_param=.2))
    torch.set_num_threads(2)
    torch.manual_seed(779)
    continuous_env = Env()
    continuous = OnPolicyRunner(continuous_env, deepcopy(config), log_dir=str(tmp_path / "continuous"), device="cpu")
    continuous.learn(6)
    expected_model = checkpoint.fingerprint(continuous.alg.policy.state_dict())
    expected_optimizer = checkpoint.fingerprint(continuous.alg.optimizer.state_dict())
    expected_rng = checkpoint.fingerprint(torch.get_rng_state())
    torch.manual_seed(779)
    first_env = Env()
    first = OnPolicyRunner(first_env, deepcopy(config), log_dir=str(tmp_path / "first"), device="cpu")
    first.learn(3)
    point = checkpoint.save_checkpoint(tmp_path / "ppo.pt", first, first_env, 3, "rsl301", 6)
    second_env = Env()
    second = OnPolicyRunner(second_env, deepcopy(config), log_dir=str(tmp_path / "second"), device="cpu")
    checkpoint.load_checkpoint(point, second, second_env, "rsl301", 6)
    assert second.current_learning_iteration == 3 and second.alg.optimizer.state
    second.learn(3)
    assert checkpoint.fingerprint(second.alg.policy.state_dict()) == expected_model
    assert checkpoint.fingerprint(second.alg.optimizer.state_dict()) == expected_optimizer
    assert checkpoint.fingerprint(torch.get_rng_state()) == expected_rng
    assert second.alg.learning_rate == continuous.alg.learning_rate
    assert second_env.common_step_counter == continuous_env.common_step_counter == 144
    for runner in (continuous, first, second):
        runner.writer.flush()
        runner.writer.close()


def test_nonempty_rollout_cannot_be_saved_as_a_clean_boundary(tmp_path):
    runner, env = learner(), Clock()
    runner.alg.storage.step = 1
    with pytest.raises(ValueError, match="empty rollout"):
        checkpoint.save_checkpoint(tmp_path / "bad.pt", runner, env, 0, "x", 3)
    assert not (tmp_path / "bad.pt").exists()


@pytest.mark.parametrize("change", ["payload", "source", "budget", "clock", "nonfinite"])
def test_tampered_or_incompatible_checkpoint_cannot_restore(tmp_path, change):
    torch = pytest.importorskip("torch")
    runner, env = learner(), Clock()
    update(runner, env)
    path = checkpoint.save_checkpoint(tmp_path / "point.pt", runner, env, 1, "source", 3)
    if change == "payload":
        path.write_bytes(path.read_bytes() + b"changed")
    elif change in ("clock", "nonfinite"):
        state = torch.load(path, weights_only=False)
        if change == "clock":
            state["training_clock"]["extrapolated_iteration"] += 1
        else:
            next(iter(state["model_state_dict"].values())).fill_(float("nan"))
        torch.save(state, path)
        sidecar = read_json(Path(str(path) + ".json"))
        sidecar["checkpoint_pins"] = [file_pin(path)]
        write_json(Path(str(path) + ".json"), sidecar, replace=True)
    with pytest.raises(ValueError):
        checkpoint.load_checkpoint(path, learner(), Clock(), "other" if change == "source" else "source",
                                   4 if change == "budget" else 3)


def _fake_source(tmp_path):
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/train_wheelbipe_port.py").write_text("# source pinned by the test\n")
    return source


def test_init_never_starts_a_process_and_freezes_source(tmp_path):
    source = _fake_source(tmp_path)
    session = WheelbipeSession.create(tmp_path / "session", source, sys.executable,
        ["--iterations", "6", "--num-envs", "4", "--seed", "115"])
    assert session.status()["phase"] == "ready" and session.attempt() is None
    session.verify_sources()
    (source / "scripts/train_wheelbipe_port.py").write_text("# changed\n")
    with pytest.raises(ValueError):
        session.start()
    assert session.attempt() is None


def test_runtime_environment_is_frozen_without_capturing_unrelated_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("UNRELATED_API_TOKEN", "must-not-be-saved")
    session = WheelbipeSession.create(tmp_path / "session", _fake_source(tmp_path), sys.executable,
        ["--iterations", "6", "--num-envs", "4", "--seed", "115"],
        environment={"LD_LIBRARY_PATH": "/explicit/sim/libs"})
    assert session.spec["runtime_environment"]["OMP_NUM_THREADS"] == "4"
    assert session.spec["runtime_environment"]["LD_LIBRARY_PATH"] == "/explicit/sim/libs"
    assert "UNRELATED_API_TOKEN" not in session.spec["runtime_environment"]


@pytest.mark.parametrize("extra", [["--resume", "old.pt"], ["--checkpoint", "old.pt"], ["--output", "old"]])
def test_no_adoption_of_legacy_checkpoint_or_existing_output(tmp_path, extra):
    with pytest.raises(ValueError, match="owned"):
        WheelbipeSession.create(tmp_path / "session", _fake_source(tmp_path), sys.executable,
            ["--iterations", "6", "--num-envs", "4", "--seed", "115", *extra])
    assert not (tmp_path / "session").exists()


def test_unclean_attempt_cannot_replay_from_periodic_checkpoint(tmp_path):
    source = _fake_source(tmp_path)
    session = WheelbipeSession.create(tmp_path / "session", source, sys.executable,
        ["--iterations", "6", "--num-envs", "4", "--seed", "115"])
    attempt = session.root / "attempts/0001-123456abcdef"
    (attempt / "control").mkdir(parents=True)
    from wheeled_algo.trainctl_shared.training_control import boot_id
    write_json(attempt / "request.json", dict(boot_id=boot_id(), token="x", submitted_epoch=0))
    write_json(attempt / "control/request.json", dict(operation="resume", sequence=0))
    write_json(session.root / "current.json", dict(attempt=attempt.name))
    with pytest.raises(ValueError, match="Unclean"):
        session.start()


def test_shared_release_verifies_checkpoint_before_requesting_stop(tmp_path):
    from wheeled_algo.trainctl_shared.training_cli import Release

    requests = []
    session = SimpleNamespace(command=requests.append)
    release = Release(session, probe=lambda: None)
    assert release.step(dict(active=True, phase="running"))[0]
    assert requests == ["pause"]
    release.step(dict(active=True, phase="paused"))
    running, message, level = release.step(dict(active=True, phase="paused", checkpoint=str(tmp_path / "missing.pt")))
    assert not running and level == "error" and requests == ["pause"]


def _wait(predicate, timeout=25.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.03)
    raise AssertionError("owned managed worker did not acknowledge within the test deadline")


def test_owned_subprocess_release_exit_and_new_process_learning_resume(tmp_path):
    pytest.importorskip("torch")
    from wheeled_algo.trainctl_shared.training_cli import _run_release
    from wheeled_algo.trainctl_shared.training_control import process_identity

    source = _fake_source(tmp_path)
    (source / "scripts/train_wheelbipe_port.py").write_text('''
import argparse,time
from pathlib import Path
from types import SimpleNamespace
import torch
from wheeled_algo.managed_training import ManagedWorker,ManagedStop
p=argparse.ArgumentParser()
for key in ('iterations','num-envs','seed'):p.add_argument('--'+key,type=int)
for key in ('output','managed-control-dir','resume'):p.add_argument('--'+key,type=Path)
a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(a.seed)
w=ManagedWorker(a.managed_control_dir,a.output,a.resume);a.output.mkdir()
policy=torch.nn.Linear(3,2)
r=SimpleNamespace(alg=SimpleNamespace(policy=policy,optimizer=torch.optim.Adam(policy.parameters(),lr=.001),
  learning_rate=.001,storage=SimpleNamespace(step=0),rnd=None),device='cpu',is_distributed=False,
  num_steps_per_env=24,current_learning_iteration=0,tot_timesteps=0,tot_time=0.)
class Env:
 common_step_counter=0
 def _get_training_iteration(self):return self.common_step_counter//24
 def set_training_progress(self,iteration):assert self._get_training_iteration()==iteration
e=Env();phase='completed'
try:
 w.attach(r,e,a.resume)
 for count in range(w.completed+1,a.iterations+1):
  r.alg.optimizer.zero_grad();loss=r.alg.policy(torch.randn(8,3)).square().mean()
  loss.backward();r.alg.optimizer.step();e.common_step_counter+=24;r.tot_timesteps+=8
  w.after_update(count);time.sleep(.15)
except ManagedStop:phase='stopped'
finally:w.commit_endpoint(phase);w.close()
''')
    session = WheelbipeSession.create(tmp_path / "session", source, sys.executable,
        ["--iterations", "12", "--num-envs", "4", "--seed", "115"])
    try:
        session.start(paused=True)
        _wait(lambda: session.status()["phase"] == "paused")
        first_identity = session.status()["identity"]
        session.command("resume", wait_seconds=3.)
        _wait(lambda: session.status()["update"] >= 2)
        state, _, level = _run_release(session, 20.)
        assert level == "ok" and state["phase"] == "stopped" and not state["active"]
        assert state["checkpoint_resume"] and 2 <= state["update"] < 12
        assert process_identity(first_identity["pid"]) is None
        previous = read_json(Path(str(state["checkpoint"]) + ".json"))
        completed = state["update"]
        session.start(paused=True)
        _wait(lambda: session.status()["phase"] == "paused")
        resumed = session.status()
        assert resumed["identity"] != first_identity and resumed["update"] == completed
        proof = read_json(session.attempt() / "train/resume_verification.json")
        assert proof["exact_learning_state_verified"]
        assert proof["learning_state_fingerprints"] == previous["learning_state_fingerprints"]
        assert proof["physical_episode_restored"] is False and proof["budget_extended"] is False
        session.command("resume", wait_seconds=3.)
        final = _wait(lambda: (s := session.status())["phase"] == "completed" and not s["active"] and s)
        assert final["update"] == final["target_update"] == 12
        assert final["cumulative_transitions"] == 12 * 4 * 24
    finally:
        if session.status()["active"]:
            session.command("stop", wait_seconds=20.)


def test_installer_adds_shared_console_route_preserving_both_existing_routes(tmp_path):
    spec = importlib.util.spec_from_file_location("disk_installer", ROOT / "scripts/install_trainctl.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    prefix = tmp_path / "install"
    executable = prefix / "bin/trainctl"
    executable.parent.mkdir(parents=True)
    original = b'#!/bin/sh\n# -m transformer_rl.common.training_cli\nprintf "original:%s\\n" "$*"\n'
    executable.write_bytes(original)
    executable.chmod(0o755)
    installer.install(prefix, share_existing=True)
    for _ in range(2):
        result = subprocess.run([str(executable), "status", "other-project"], capture_output=True, text=True)
        assert result.returncode == 0 and result.stdout.strip() == "original:status other-project"
        result = subprocess.run([str(executable), "wheelbipe", "list", str(tmp_path), "--json"], capture_output=True, text=True)
        assert result.returncode == 0 and json.loads(result.stdout) == []
        installer.install(prefix)
    preserved = list((prefix / "share/wheel-leg-trainctl").glob("original-trainctl-*.sh"))
    assert len(preserved) == 1 and preserved[0].read_bytes() == original
