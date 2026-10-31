"""Run options must reach the native contract without changing the mechanism."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import pytest
import torch
from tensordict import TensorDict

from train_prone_motion import configuration, learner_config, source_hashes
from wheeled_tasks.chassis import prone_motion as task
from wheeled_tasks.chassis.prone_motion_cli import resolve_plan, train_parser, training_options
import play_prone_motion as play

ROOT = Path(__file__).resolve().parents[1]
PLAN_FILE = ROOT/"contracts/wheel_leg_prone_motion_v4.json"
BASE = json.loads(PLAN_FILE.read_text())


def test_explicit_v4_options_preserve_the_accepted_recipe(tmp_path):
    args = train_parser().parse_args(["--task","WheelLeg-ProneMotion-v4","--output", str(tmp_path/"train")])
    path, plan, output = training_options(args, ROOT)
    assert path == PLAN_FILE and output == tmp_path/"train"
    for key, value in BASE.items():
        if key != "training_progression":
            assert plan[key] == value
    assert task.validate(plan) == plan
    assert args.headless and args.device == "cuda:0"


@pytest.mark.parametrize("num_envs", [128, 512])
def test_isaac_style_flags_reach_contract_and_learner(tmp_path, num_envs):
    args = train_parser().parse_args(["--task","WheelLeg-ProneMotion-v4","--num_envs", str(num_envs), "--max_iterations", "5000", "--seed", "103",
        "--run_name", "manual_5000", "--device", "cuda:1", "--headless", "--save_interval", "100",
        "--learning_rate", ".0001", "--domain_randomization", "off"])
    _, plan, output = training_options(args, ROOT)
    assert output == ROOT/"runs/prone_motion/manual_5000"
    assert plan["foundation_until_update"] == 500 and plan["height_full_update"] == 2500
    assert plan["speed_full_update"] == 3000 and plan["spin_full_update"] == 3500
    assert task.domain_stage(plan, 4999) == ("nominal", 0.)
    cfg, settings, _ = configuration(tmp_path, plan, device=args.device)
    agent = learner_config(plan, args.device)
    assert cfg["target_num_envs"] == num_envs and cfg["num_steps_per_env"] == 64
    assert agent["device"] == "cuda:1" and agent["seed"] == 103
    assert agent["max_iterations"] == 5000 and agent["save_interval"] == 100
    assert agent["algorithm"]["learning_rate"] == .0001 and not agent["resume"]
    assert cfg["v5_control"]["gas_spring_compensation"] == BASE["gas_spring_feedforward"]
    assert settings["prone_motion_task"] == plan
    saved = json.loads((tmp_path/"task_contract.json").read_text())
    assert saved["plan"] == plan and saved["policy_hz"] == 100
    assert saved["physics_hz"] == 1000 and saved["rollout_steps"] == 64
    assert plan["evaluation"] == BASE["evaluation"] and plan["roles_per20"] == BASE["roles_per20"]


@pytest.mark.parametrize("mode,strength,expected", [("off",1.,("nominal",0.)), ("on",.5,("requested",.5)),
                                                   ("curriculum",.5,("requested",.5))])
def test_explicit_domains_and_scaled_schedule(mode, strength, expected):
    plan = resolve_plan(BASE, max_iterations=5000, domain_randomization=mode, dr_strength=strength)
    task.validate(plan)
    assert task.domain_stage(plan,4500) == ("nominal",0.)
    assert (plan["domain_sequence"][-1]["profile"],plan["domain_sequence"][-1]["strength"]) == expected
    # The preserved pre-global contract still reproduces its original clock.
    legacy = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4_recovery_20261031.json").read_text())
    legacy = resolve_plan(legacy,max_iterations=5000,domain_randomization=mode,dr_strength=strength)
    assert task.domain_stage(legacy,4500) == expected
    if mode == "on":
        assert task.domain_stage(legacy, 0) == expected
    if mode == "curriculum":
        assert task.domain_stage(legacy, 2999) == ("nominal",0.)
        assert task.domain_stage(legacy, 3000) == ("sensors_only", .125)


@pytest.mark.parametrize("override", [dict(num_envs=19), dict(num_envs=513), dict(max_iterations=0), dict(seed=-1),
    dict(seed=2**32), dict(save_interval=0), dict(learning_rate=float("nan")), dict(dr_strength=1.1),
    dict(domain_randomization="unknown"), dict(num_steps_per_env=0), dict(position_iterations=0),
    dict(position_iterations=256), dict(position_iterations=True), dict(velocity_iterations=-1),
    dict(velocity_iterations=256)])
def test_invalid_run_options_fail_before_output_or_kit(override):
    with pytest.raises(ValueError):
        resolve_plan(BASE, **override)


@pytest.mark.parametrize("key,value", [("forward_speed_m_s", 3.), ("completion_hold_seconds", .1),
    ("timing", {"policy_dt": .005}), ("initialization", "resume"), ("snapshot_interval", 1)])
def test_resolved_plan_rejects_undeclared_changes(key, value):
    plan = resolve_plan(BASE)
    plan[key] = value
    with pytest.raises(ValueError):
        task.validate(plan)


def test_budget_does_not_enable_segmented_or_resumed_training():
    plan = resolve_plan(BASE, max_iterations=5000, num_steps_per_env=32)
    assert task.validate(plan)["rollout_steps"] == 32
    assert plan["review_updates"] == [] and plan["execution"] == "single_learn_call_manual_review"
    assert task.snapshot_name(250, plan) == "model_00250.pt"
    assert task.snapshot_name(5000, plan) == "model_05000.pt"
    with pytest.raises(SystemExit):
        train_parser().parse_args(["--checkpoint", "old.pt"])


def test_both_flag_spellings_and_no_gui():
    args = train_parser().parse_args(["--num-envs", "128", "--max-iterations", "5000", "--run-name", "hyphen",
        "--domain-randomization", "on", "--dr-strength", ".25", "--save-interval", "50", "--no-headless"])
    _, plan, _ = training_options(args, ROOT)
    assert not args.headless and plan["num_envs"] == 128
    assert task.domain_stage(plan, 0) == ("nominal", 0.)
    assert plan["domain_sequence"] == [dict(start_update=0,profile="requested",strength=.25)]


def test_python_override_is_accepted_by_train_and_play_without_changing_the_plan():
    args = train_parser().parse_args(["--task","WheelLeg-ProneMotion-v4","--python", "/custom/isaac/python"])
    assert args.python == "/custom/isaac/python"
    _, plan, _ = training_options(args, ROOT)
    assert plan == resolve_plan(BASE)
    args = play.parser().parse_args(["--python=/custom/isaac/python", "--checkpoint", "unused.pt"])
    assert args.python == "/custom/isaac/python"


@pytest.mark.parametrize("version", [3, 4, 5, "4_standard_20261031"])
def test_play_has_the_same_role_plan_and_restores_actor_only(tmp_path, version, monkeypatch):
    from rsl_rl.utils import resolve_class
    from train_scratch_standing import digest
    original = json.loads((ROOT/f"contracts/wheel_leg_prone_motion_v{version}.json").read_text())
    plan = resolve_plan(original, max_iterations=5000)
    agent = learner_config(plan, "cuda:0")
    obs = TensorDict(dict(policy=torch.zeros(1,35), critic=torch.zeros(1,87 if plan.get("full_vehicle") else 83)), batch_size=[1])
    cls, settings = resolve_class(deepcopy(agent["actor"]))
    actor = cls(obs, agent["obs_groups"], "actor", 6, **settings).eval()
    bank_file = plan.get("pose_bank",{}).get("file","reports/robust_policy_repair_20261024/pose_bank_height_matched073.json")
    (tmp_path/"pose_bank.json").write_bytes((ROOT/bank_file).read_bytes())
    (tmp_path/"agent_config.json").write_text(json.dumps(agent))
    (tmp_path/"curriculum_plan.resolved.json").write_text(json.dumps(plan))
    checkpoint = tmp_path/"model_00250.pt"
    torch.save(dict(actor_state_dict=actor.state_dict(), iter=250, infos=dict(schema="prone-motion-checkpoint-v1",
        training_origin="scratch", completed_updates=250, resolved_plan=plan,
        bank_sha256=digest(tmp_path/"pose_bank.json"), source_revision={"commit":"test-only-untrained"})), checkpoint)
    args = play.parser().parse_args(["--checkpoint", str(checkpoint), "--scene", "forward", "--vx", "1.2", "--height", ".42",
        "--num_envs", "1", "--headless", "--preflight-only"])
    bundle = play.load_bundle(args)
    restored = play.build_actor(obs, bundle, "cpu")
    with torch.inference_mode():
        torch.testing.assert_close(restored(obs), actor(obs), rtol=0., atol=0.)
    cfg, settings, _ = play.prepare(args, bundle, tmp_path/"preview")
    assert cfg["target_num_envs"] == 1 and cfg["scene_groups"][0]["fraction"] == 1.
    assert settings["prone_motion_task"] == plan
    assert settings.get("standing_odometry") == plan.get("standing_odometry")
    env = SimpleNamespace(prone_index=7, height_pose={.305:2},
        poses=json.loads((tmp_path/"pose_bank.json").read_text())["poses"])
    assert play.case_for(args, env, plan)["target"] == [1.2,0.,.42]
    assert bundle["profile"] == "nominal" and not any(k.startswith("optimizer") for k in bundle["state"])
    preview = tmp_path/"main_preview"
    monkeypatch.setattr(sys,"argv",["play_prone_motion.py","--checkpoint",str(checkpoint),
        "--output",str(preview),"--preflight-only","--device","cuda:0"])
    play.main()
    receipt = json.loads((preview/"receipt.json").read_text())
    assert receipt["status"] == "preflight_completed" and receipt["new_PPO"] == 0
    args.scene = "prone"
    if "multi_pose_recovery" in plan:
        args.fallen_family,args.leg_class = "roof_down","front_rear"
    case = play.case_for(args, env, plan)
    assert case["recovery"]
    if "multi_pose_recovery" in plan:
        selected = env.poses[case["pose_index"]]["recovery"]
        assert (selected["family"],selected["leg_class"],selected["split"]) == ("roof_down","front_rear","holdout")
    else:
        assert case["pose_index"] == 7
    # A copied/changed recipe cannot silently replace the checkpoint's mechanism.
    bad = deepcopy(plan)
    bad["gas_spring_backend"] = "other"
    (tmp_path/"curriculum_plan.resolved.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        play.load_bundle(args)


def test_old_resolved_checkpoint_with_eight_runtime_options_stays_readable():
    from wheeled_tasks.chassis.prone_motion_cli import LEGACY_OPTION_KEYS
    legacy = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4_standard_20261031.json").read_text())
    old = resolve_plan(legacy,max_iterations=5000)
    old["runtime_cli"]["options"] = {k:v for k,v in old["runtime_cli"]["options"].items() if k in LEGACY_OPTION_KEYS}
    assert task.validate(old) == old


def test_solver_override_reaches_authenticated_design_and_preserves_other_physics(tmp_path):
    args = train_parser().parse_args(["--position-iterations", "12", "--velocity_iterations", "6"])
    _, plan, _ = training_options(args, ROOT)
    cfg, _, _ = configuration(tmp_path/"reduced", plan)
    original, _, _ = configuration(tmp_path/"original", resolve_plan(BASE))
    assert (original["position_iterations"], original["velocity_iterations"]) == (64,32)
    assert (cfg["position_iterations"], cfg["velocity_iterations"]) == (12,6)
    saved = json.loads((tmp_path/"reduced/task_contract.json").read_text())
    assert saved["solver_iterations"] == plan["solver_iterations"] == dict(position_iterations=12,velocity_iterations=6)
    design = json.loads((tmp_path/"reduced/config/design.json").read_text())
    row = design["stages"]["speed"]["roles"]["train"]["contract"]
    assert (row["position_iterations"],row["velocity_iterations"]) == (12,6)
    for key in ("closure_gap_termination_m", "asset_manifest_sha256", "full_vehicle", "leg_motor_friction"):
        assert cfg[key] == original[key]
    assert plan["timing"] == BASE["timing"] and plan["global_curriculum"] == BASE["global_curriculum"]
    changed = deepcopy(plan)
    changed["solver_iterations"]["velocity_iterations"] = 7
    with pytest.raises(ValueError):task.validate(changed)
    # Zero velocity iterations is an explicit PhysX setting, not an omitted flag.
    zero = resolve_plan(BASE,position_iterations=12,velocity_iterations=0)
    assert zero["solver_iterations"]["velocity_iterations"] == 0


def test_old_eighteen_option_global_checkpoint_keeps_original_solver(tmp_path):
    from wheeled_tasks.chassis.prone_motion_cli import LEGACY_GLOBAL_OPTION_KEYS
    old = resolve_plan(BASE)
    old["runtime_cli"]["options"] = {k:v for k,v in old["runtime_cli"]["options"].items() if k in LEGACY_GLOBAL_OPTION_KEYS}
    assert task.validate(old) == old and "solver_iterations" not in old
    cfg, _, _ = configuration(tmp_path, old)
    assert (cfg["position_iterations"],cfg["velocity_iterations"]) == (64,32)


def test_explicit_v3_training_keeps_legacy_zero_wheel_position_slots(tmp_path):
    args = train_parser().parse_args(["--task", "WheelLeg-ProneMotion-v3"])
    _, plan, _ = training_options(args, ROOT)
    cfg, settings, _ = configuration(tmp_path, plan)
    assert "standing_odometry" not in settings and "quiet_position_tail" not in plan
    assert cfg["actor_frame_dim"] == 35 and not settings["quiet_velocity_feedback"]


def test_all_new_entrypoints_are_part_of_the_source_freeze():
    hashes = source_hashes(PLAN_FILE, resolve_plan(BASE))
    for name in ("scripts/play_prone_motion.py", "scripts/play_prone_motion.sh", "tests/test_prone_motion_cli.py",
                 "scripts/resolve_isaac_python.sh", "tests/test_prone_motion_python.py",
                 "src/wheeled_tasks/chassis/prone_motion_cli.py"):
        assert len(hashes[name]) == 64


@pytest.mark.parametrize("fail_probe", [False, True])
def test_run_name_wrapper_checks_native_before_any_learner(tmp_path, fail_probe):
    interpreter = tmp_path/"fake_python"
    interpreter.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['CLI_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps(args)+'\\n')
if '--print-output' in args:
    print(os.environ['CLI_TEST_OUTPUT'])
elif '--native-probe' in args and os.environ['CLI_TEST_FAIL'] == '1':
    sys.exit(17)
''')
    interpreter.chmod(0o755)
    log = tmp_path/"calls.jsonl"
    output = tmp_path/"run with space"
    env = dict(os.environ, WHEELED_RL_PYTHON=str(interpreter), CLI_TEST_LOG=str(log),
        CLI_TEST_OUTPUT=str(output), CLI_TEST_FAIL=str(int(fail_probe)))
    result = subprocess.run(["bash", str(ROOT/"scripts/train_prone_motion.sh"), "--run_name", "manual",
        "--max_iterations", "5000", "--seed", "103", "--domain_randomization", "off"], env=env, capture_output=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert '--print-output' in calls[0] and '--native-probe' in calls[1]
    assert calls[1][-3:] == ["--output", str(output)+".native_probe", "--native-probe"]
    if fail_probe:
        assert result.returncode == 17 and len(calls) == 2
    else:
        assert result.returncode == 0 and calls[2][0] == "-"  # verification before learner
        assert '--native-probe' not in calls[3] and calls[3][-2:] == ["--output", str(output)]
        assert len(calls) == 4


def test_run_names_cannot_escape_the_run_directory():
    for name in ("../old", ".", "..", "a/b"):
        args = train_parser().parse_args(["--run_name", name])
        with pytest.raises(ValueError):
            training_options(args, ROOT)
