"""Run options must reach the native contract without changing the mechanism."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess

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


def test_default_options_preserve_the_accepted_recipe(tmp_path):
    args = train_parser().parse_args(["--output", str(tmp_path/"train")])
    path, plan, output = training_options(args, ROOT)
    assert path == PLAN_FILE and output == tmp_path/"train"
    for key, value in BASE.items():
        if key != "training_progression":
            assert plan[key] == value
    assert task.validate(plan) == plan
    assert args.headless and args.device == "cuda:0"


def test_isaac_style_flags_reach_contract_and_learner(tmp_path):
    args = train_parser().parse_args(["--num_envs", "128", "--max_iterations", "5000", "--seed", "103",
        "--run_name", "manual_5000", "--device", "cuda:1", "--headless", "--save_interval", "100",
        "--learning_rate", ".0001", "--domain_randomization", "off"])
    _, plan, output = training_options(args, ROOT)
    assert output == ROOT/"runs/prone_motion/manual_5000"
    assert plan["foundation_until_update"] == 500 and plan["height_full_update"] == 2500
    assert plan["speed_full_update"] == 3000 and plan["spin_full_update"] == 3500
    assert task.domain_stage(plan, 4999) == ("nominal", 0.)
    cfg, settings, _ = configuration(tmp_path, plan, device=args.device)
    agent = learner_config(plan, args.device)
    assert cfg["target_num_envs"] == 128 and cfg["num_steps_per_env"] == 64
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
    assert task.domain_stage(plan, 4500) == expected
    if mode == "on":
        assert task.domain_stage(plan, 0) == expected
    if mode == "curriculum":
        assert task.domain_stage(plan, 2999) == ("nominal",0.)
        assert task.domain_stage(plan, 3000) == ("sensors_only", .125)


@pytest.mark.parametrize("override", [dict(num_envs=19), dict(num_envs=512), dict(max_iterations=0), dict(seed=-1),
    dict(seed=2**32), dict(save_interval=0), dict(learning_rate=float("nan")), dict(dr_strength=1.1),
    dict(domain_randomization="unknown"), dict(num_steps_per_env=0)])
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
    assert task.domain_stage(plan, 0) == ("requested", .25)


def test_python_override_is_accepted_by_train_and_play_without_changing_the_plan():
    args = train_parser().parse_args(["--python", "/custom/isaac/python"])
    assert args.python == "/custom/isaac/python"
    _, plan, _ = training_options(args, ROOT)
    assert plan == resolve_plan(BASE)
    args = play.parser().parse_args(["--python=/custom/isaac/python", "--checkpoint", "unused.pt"])
    assert args.python == "/custom/isaac/python"


@pytest.mark.parametrize("version", [3, 4])
def test_play_has_the_same_role_plan_and_restores_actor_only(tmp_path, version):
    from rsl_rl.utils import resolve_class
    from train_scratch_standing import digest
    original = json.loads((ROOT/f"contracts/wheel_leg_prone_motion_v{version}.json").read_text())
    plan = resolve_plan(original, max_iterations=5000)
    agent = learner_config(plan, "cuda:0")
    obs = TensorDict(dict(policy=torch.zeros(1,35), critic=torch.zeros(1,83)), batch_size=[1])
    cls, settings = resolve_class(deepcopy(agent["actor"]))
    actor = cls(obs, agent["obs_groups"], "actor", 6, **settings).eval()
    (tmp_path/"pose_bank.json").write_bytes((ROOT/"reports/robust_policy_repair_20261024/pose_bank_height_matched073.json").read_bytes())
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
    env = SimpleNamespace(prone_index=7, height_pose={.305:2})
    assert play.case_for(args, env, plan)["target"] == [1.2,0.,.42]
    assert bundle["profile"] == "nominal" and not any(k.startswith("optimizer") for k in bundle["state"])
    args.scene = "prone"
    case = play.case_for(args, env, plan)
    assert case["recovery"] and case["pose_index"] == 7
    # A copied/changed recipe cannot silently replace the checkpoint's mechanism.
    bad = deepcopy(plan)
    bad["gas_spring_backend"] = "other"
    (tmp_path/"curriculum_plan.resolved.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        play.load_bundle(args)


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
