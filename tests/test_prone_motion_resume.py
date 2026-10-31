"""Interrupted-course recovery must preserve learning state and course clocks."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from train_prone_motion import learner_config
from wheeled_algo.prone_motion_resume import (
    authenticate_sources, digest, equal_state, metadata, restore, restore_rng, validate_checkpoint,
)
from wheeled_tasks.chassis.prone_motion import domain_stage, target
from wheeled_tasks.chassis.prone_motion_cli import resolve_plan, train_parser, training_options

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4.json").read_text())


class Env:
    num_envs, num_actions, device, cfg = 20, 6, "cpu", {}

    def get_observations(self):
        return TensorDict(dict(policy=torch.full((20,35), .1), critic=torch.full((20,87), .2)), batch_size=[20])

    def step(self, action):
        return self.get_observations(), -action.square().mean(-1), torch.zeros(20, dtype=torch.long), {}


def runner(plan):
    cfg = learner_config(plan, "cpu")
    cfg["actor"]["hidden_dims"] = cfg["critic"]["hidden_dims"] = [8]
    return OnPolicyRunner(Env(), cfg, device="cpu")


@pytest.fixture
def saved(tmp_path):
    torch.set_num_threads(2)
    plan = resolve_plan(BASE, num_envs=20, num_steps_per_env=2)
    first = runner(plan)
    first.learn(1)
    state = deepcopy(first.alg.save())
    old = tmp_path/"original"
    old.mkdir()
    (old/"pose_bank.json").write_text("{}")
    sources = {"plant.py": "test-source-hash"}
    state.update(iter=4000, infos=dict(schema="prone-motion-checkpoint-v1", training_origin="scratch",
        continuation_forbidden=True, completed_updates=4000, resolved_plan=plan,
        source_hashes=sources, bank_sha256=digest(old/"pose_bank.json"), initialization={"checkpoint_loaded":False},
        torch_rng=torch.get_rng_state(), cuda_rng=[], env_rng=torch.get_rng_state(), domain_rng=torch.get_rng_state()))
    checkpoint = old/"model_04000.pt"
    torch.save(state, checkpoint)
    (old/"curriculum_plan.resolved.json").write_text(json.dumps(plan))
    (old/"source_hashes.json").write_text(json.dumps(sources))
    (old/"run_options.json").write_text(json.dumps(dict(task="WheelLeg-ProneMotion-v4", performance_backend="cuda_graph")))
    (old/"progress.json").write_text(json.dumps(dict(completed_updates=4015,
        last_checkpoint=dict(path=str(checkpoint), sha256=digest(checkpoint), updates=4000))))
    return state, metadata(checkpoint)


def test_full_restore_and_next_native_ppo_keep_adam_norms_std_and_iteration(saved):
    state, bundle = saved
    second = runner(bundle["plan"])
    proof = restore(second, state, bundle)
    assert all(proof["checks"].values()) and proof["optimizer_states"] == 9
    assert second.current_learning_iteration == second.env.actor_updates == 4000
    assert second.logger.tot_timesteps == 4000*20*2
    assert equal_state(second.alg.save(), {k:state[k] for k in second.alg.save()})
    assert proof["remaining_updates"] == 6000 and not proof["curriculum_restarted"]
    before = deepcopy(second.alg.save())
    observed = []
    second.logger.log = lambda **kw: observed.append((kw["it"],kw["start_it"],kw["total_it"]))
    second.learn(1)
    assert observed == [(4000,4000,4001)]
    after = second.alg.save()
    assert not equal_state(before["actor_state_dict"], after["actor_state_dict"])
    assert not equal_state(before["critic_state_dict"], after["critic_state_dict"])
    for key, value in before["optimizer_state_dict"]["state"].items():
        assert float(after["optimizer_state_dict"]["state"][key]["step"]) == float(value["step"])+12


def test_resume_inherits_course_without_scaling_to_remaining_budget(saved, tmp_path):
    _, bundle = saved
    args = train_parser().parse_args(["--resume", str(bundle["path"]), "--output", str(tmp_path/"continued")])
    _, plan, _ = training_options(args, ROOT)
    assert plan == bundle["plan"]
    assert plan["foundation_until_update"] == 1000 and plan["max_updates"] == 10000
    assert target("forward",4000,plan) == [1.2,0.,.305]
    assert target("forward",0,plan) == [0.,0.,.305]
    assert domain_stage(plan,6000) == ("sensors_only",.25)


@pytest.mark.parametrize("flag,value", [("--max_iterations","6000"),("--num_envs","128"),
    ("--seed","123"),("--learning_rate",".001"),("--domain_randomization","off"),
    ("--num_steps_per_env","64"),("--dr_strength",".5")])
def test_course_changes_rejected_before_starting_native(saved, tmp_path, flag, value):
    _, bundle = saved
    args = train_parser().parse_args(["--resume",str(bundle["path"]),"--output",str(tmp_path/"continued"),flag,value])
    with pytest.raises(ValueError, match="changed course"):
        training_options(args, ROOT)


@pytest.mark.parametrize("broken", ["iter","plan","adam","normalizer","nonfinite","rng"])
def test_incomplete_or_incompatible_state_cannot_resume(saved, broken):
    original, bundle = saved
    state = deepcopy(original)
    if broken == "iter": state["iter"] = 0
    if broken == "plan": state["infos"]["resolved_plan"]["max_updates"] = 6000
    if broken == "adam": state["optimizer_state_dict"]["state"] = {}
    if broken == "normalizer": del state["actor_state_dict"]["obs_normalizer._mean"]
    if broken == "nonfinite": next(iter(state["critic_state_dict"].values())).fill_(float("nan"))
    if broken == "rng": del state["infos"]["env_rng"]
    with pytest.raises(ValueError): validate_checkpoint(state,bundle)


def test_original_checkpoint_bytes_and_output_are_protected(saved):
    _, bundle = saved
    args = train_parser().parse_args(["--resume",str(bundle["path"]),"--output",str(bundle["path"].parent)])
    with pytest.raises(ValueError, match="separate"):
        training_options(args,ROOT)
    with bundle["path"].open("ab") as stream: stream.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        metadata(bundle["path"])


def test_source_check_allows_resume_support_but_refuses_plant_changes(tmp_path):
    plant, entry = tmp_path/"plant.py", tmp_path/"scripts/train_prone_motion.py"
    entry.parent.mkdir()
    plant.write_text("old plant")
    entry.write_text("resume support")
    original = {"plant.py":digest(plant), "scripts/train_prone_motion.py":"old entry hash"}
    current = {name:digest(tmp_path/name) for name in original}
    assert authenticate_sources({"sources":original},current,tmp_path)["unchanged_source_count"] == 1
    plant.write_text("changed dynamics")
    current["plant.py"] = digest(plant)
    with pytest.raises(ValueError, match="changed training sources"):
        authenticate_sources({"sources":original},current,tmp_path)


def test_saved_rng_streams_are_restored_without_claiming_physics_replay(saved, monkeypatch):
    from types import SimpleNamespace
    state, _ = saved
    env = SimpleNamespace(generator=torch.Generator(), domain=SimpleNamespace(rng=torch.Generator()))
    cuda = []
    monkeypatch.setattr(torch.cuda,"set_rng_state_all",lambda values:cuda.append(values))
    torch.manual_seed(123)
    proof = restore_rng(state,env)
    assert torch.equal(torch.get_rng_state(),state["infos"]["torch_rng"])
    assert torch.equal(env.generator.get_state(),state["infos"]["env_rng"])
    assert torch.equal(env.domain.rng.get_state(),state["infos"]["domain_rng"])
    assert cuda == [[]] and not proof["exact_physics_trajectory_replay"]
