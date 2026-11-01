"""Interrupted-course recovery must preserve learning state and course clocks."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from train_prone_motion import learner_config
from wheeled_algo.prone_motion_resume import (
    authenticate_sources, digest, equal_state, metadata, restore, restore_rng, restore_curriculum, validate_checkpoint,
)
from wheeled_tasks.chassis.prone_motion import domain_stage, timing
from wheeled_tasks.chassis.prone_motion_cli import resolve_plan, train_parser, training_options, validate_resolved
from wheeled_tasks.chassis.global_curriculum import GlobalCurriculum, SKILLS, assessment_descriptors
from wheeled_tasks.chassis.recovery_mastery import RecoveryMastery, CLASSES
from wheeled_tasks.chassis.multi_pose_recovery import ResetSampler

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
    mastery = RecoveryMastery(plan["multi_pose_recovery"]["mastery_gate"])
    course = GlobalCurriculum(plan, mastery)
    course.review_skill("standing", 0, [dict(r, passed=True) for r in
        assessment_descriptors(course.config, "standing", 0)], 250)
    from evaluate_recovery_mastery import cases_for
    bank=json.loads((ROOT/plan['pose_bank']['file']).read_text())
    mastery.review([dict(row,passed=True) for row in cases_for(
        SimpleNamespace(poses=bank['poses'],num_envs=20),mastery.config)],250)
    # Synthetic mastered state tests restoration, never robot capability.
    course.levels = {skill:len(course.config["skills"][skill]["levels"]) for skill in SKILLS}
    course.retained_levels = dict(course.levels)
    course.complex_passed = True
    course.finish_review({skill:dict(passed=True) for skill in (*SKILLS,"belly","complex")}, 3500)
    bank = json.loads((ROOT/plan["pose_bank"]["file"]).read_text())
    sampler = ResetSampler(bank, plan["multi_pose_recovery"], 42, mastery=mastery)
    for _ in range(41): sampler.sample(4000)
    state.update(iter=4000, infos=dict(schema="prone-motion-checkpoint-v1", training_origin="scratch",
        continuation_forbidden=True, completed_updates=4000, resolved_plan=plan,
        source_hashes=sources, bank_sha256=digest(old/"pose_bank.json"), initialization={"checkpoint_loaded":False},
        recovery_mastery=mastery.state_dict(), global_curriculum=course.state_dict(),
        recovery_sampler=sampler.state_dict(), prone_reset_generation=17,
        torch_rng=torch.get_rng_state(), cuda_rng=[], env_rng=torch.get_rng_state(), domain_rng=torch.get_rng_state()))
    checkpoint = old/"model_04000.pt"
    torch.save(state, checkpoint)
    (old/"curriculum_plan.resolved.json").write_text(json.dumps(plan))
    (old/"source_hashes.json").write_text(json.dumps(sources))
    (old/"run_options.json").write_text(json.dumps(dict(task="WheelLeg-ProneMotion-v4",
        performance_backend="cuda_graph", sampling_backend="packed")))
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
    mastery, course = restore_curriculum(saved[0], plan)
    assert mastery.unlocked and course.recovery is mastery
    assert domain_stage(plan,4000,course) == ("sensors_only",.25)
    assert domain_stage(plan,6000) == ("nominal",0.)  # Update count cannot replace saved gates.
    assert timing(plan)["physics_dt"] == .005
    assert args.performance_backend == "cuda_graph" and args.sampling_backend == "packed"


@pytest.mark.parametrize("flag,value", [("--max_iterations","6000"),("--num_envs","128"),
    ("--seed","123"),("--learning_rate",".001"),("--domain_randomization","off"),
    ("--num_steps_per_env","64"),("--dr_strength",".5"),("--physics-hz","1000"),
    ("--policy-hz","50"),("--hidden-dims","512"),("--ppo-epochs","5"),("--ppo-mini-batches","8")])
def test_course_changes_rejected_before_starting_native(saved, tmp_path, flag, value):
    _, bundle = saved
    args = train_parser().parse_args(["--resume",str(bundle["path"]),"--output",str(tmp_path/"continued"),flag,value])
    with pytest.raises(ValueError, match="changed course"):
        training_options(args, ROOT)


def test_resume_accepts_explicit_original_policy_and_ppo_without_changing_saved_plan(saved, tmp_path):
    _, bundle = saved
    args = train_parser().parse_args(["--resume", str(bundle["path"]), "--output", str(tmp_path/"continued"),
        "--physics-hz", "200", "--policy-hz", "100", "--hidden-dims", "256", "128", "64",
        "--ppo-epochs", "3", "--ppo-mini-batches", "4"])
    _, plan, _ = training_options(args, ROOT)
    assert plan == bundle["plan"]
    assert plan["max_updates"] == 10000 and timing(plan)["policy_dt"] == .01


@pytest.mark.parametrize("broken", ["iter","plan","adam","normalizer","nonfinite","rng",
    "mastery","course","sampler","reset_clock","future_course","divergent_mastery"])
def test_incomplete_or_incompatible_state_cannot_resume(saved, broken):
    original, bundle = saved
    state = deepcopy(original)
    if broken == "iter": state["iter"] = 0
    if broken == "plan": state["infos"]["resolved_plan"]["max_updates"] = 6000
    if broken == "adam": state["optimizer_state_dict"]["state"] = {}
    if broken == "normalizer": del state["actor_state_dict"]["obs_normalizer._mean"]
    if broken == "nonfinite": next(iter(state["critic_state_dict"].values())).fill_(float("nan"))
    if broken == "rng": del state["infos"]["env_rng"]
    if broken == "mastery": del state["infos"]["recovery_mastery"]
    if broken == "course": del state["infos"]["global_curriculum"]
    if broken == "sampler": del state["infos"]["recovery_sampler"]
    if broken == "reset_clock": state["infos"]["prone_reset_generation"] = -1
    if broken == "future_course": state["infos"]["global_curriculum"]["domain_started_at"] = 4001
    if broken == "divergent_mastery": state["infos"]["recovery_mastery"]["pass_streak"] += 1
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


def test_resume_preserves_admitted_skills_replay_domain_and_next_pose_draws(saved, monkeypatch):
    from types import SimpleNamespace
    state, bundle = saved
    mastery, course = restore_curriculum(state, bundle["plan"])
    assert course.state_dict() == state["infos"]["global_curriculum"]
    before = course.training_descriptors(20, 17)
    assert len({row["curriculum_skill"] for row in before}) > 1
    bank = json.loads((ROOT/bundle["plan"]["pose_bank"]["file"]).read_text())
    expected = ResetSampler(bank, bundle["plan"]["multi_pose_recovery"], 1, mastery=mastery)
    expected.load_state_dict(state["infos"]["recovery_sampler"])
    actual = ResetSampler(bank, bundle["plan"]["multi_pose_recovery"], 999, mastery=mastery)
    env = SimpleNamespace(generator=torch.Generator(), domain=SimpleNamespace(rng=torch.Generator()),
        recovery_sampler=actual, prone_reset_generation=0)
    monkeypatch.setattr(torch.cuda,"set_rng_state_all",lambda _:None)
    proof = restore_rng(state, env)
    assert proof["recovery_sampler"] and env.prone_reset_generation == 17
    assert course.training_descriptors(20,env.prone_reset_generation) == before
    assert [actual.sample(4000) for _ in range(200)] == [expected.sample(4000) for _ in range(200)]
    assert not course.recovery.due(4000)  # An admitted belly gate stays admitted.


def test_resume_inherits_original_backends_and_refuses_explicit_changes(saved, tmp_path):
    _, bundle = saved
    options = bundle["path"].parent/"run_options.json"
    values = json.loads(options.read_text())
    values.update(performance_backend="eager", sampling_backend="reference")
    options.write_text(json.dumps(values))
    argv = ["--resume",str(bundle["path"]),"--output",str(tmp_path/"continued")]
    args = train_parser().parse_args(argv)
    training_options(args,ROOT)
    assert (args.performance_backend,args.sampling_backend) == ("eager","reference")
    for flag,value in (("--performance-backend","cuda_graph"),("--sampling-backend","packed")):
        with pytest.raises(ValueError,match="original"):
            training_options(train_parser().parse_args(argv+[flag,value]),ROOT)


@pytest.mark.parametrize("global_course", [True,False])
def test_historical_resolved_plan_wording_stays_readable_without_accepting_course_changes(global_course):
    base = BASE if global_course else json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4_standard_20261031.json").read_text())
    plan = resolve_plan(base)
    plan["training_progression"] = (
        "One fresh learn(10000); URDF nominal standing first; cold per-skill/per-level admission, "
        "persistent replay and measured domain admission; no elapsed-update unlock or resume." if global_course else
        "One continuous learn(10000) call; snapshots every 250 complete updates. Manual checkpoint selection; no resume. "
        "Curriculum update thresholds scale with this run's budget; acceptance limits do not change.")
    assert validate_resolved(plan) == plan
    plan["num_envs"] = 128
    with pytest.raises(ValueError): validate_resolved(plan)
