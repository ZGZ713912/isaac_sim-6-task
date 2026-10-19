"""Recovered actor trials must use its exact mechanics and deployed action ABI."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.flat_base_continuation import write_bundle
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.paired_leg_targets import paired_leg_targets
from wheeled_tasks.chassis.preflight import design_preflight
from wheeled_tasks.chassis.v5_control import V5Control

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_paired107_matches_independently_compiled_rmcs_reference(dtype):
    fixture = json.loads((ROOT / "tests/fixtures/paired107_cpp_reference.json").read_text())
    wanted = torch.tensor(fixture["desired"], dtype=torch.float64)
    current = torch.tensor(fixture["current"], dtype=torch.float64)
    expected = torch.tensor(fixture["expected"], dtype=torch.float64)
    # C++ calculations are double, with the same explicit final PhysX cast.
    actual = paired_leg_targets(wanted, current).to(dtype)
    torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=0)


def test_paired107_float32_inputs_and_common_turns():
    wanted = torch.tensor([[-.42, .13742282595, .42, -.13741557626]])
    current = wanted + torch.tensor([[20., 20., -30., -30.]])
    result = paired_leg_targets(wanted, current)
    reference = paired_leg_targets(wanted.double(), current.double()).float()
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    for side in (0, 2):
        torch.testing.assert_close(result[:, side + 1] - result[:, side],
                                  wanted[:, side + 1] - wanted[:, side], atol=3e-6, rtol=0)


def test_bad_leg_winding_does_not_silently_select_another_controller():
    manifest = dict(control_joint_names=list(V5Control.ACTIVE))
    with pytest.raises(ValueError, match="Unknown leg_target_winding"):
        V5Control(manifest, {}, {}, {}, {"leg_target_winding": "paired_branhc107_v1"}, "cpu")


@pytest.mark.parametrize("evaluation,num_envs", ((True, 32), (False, 512)))
@pytest.mark.parametrize("plan_file", ("contracts/flat_base_finetune_v1.json", "contracts/flat_base_finetune_v2.json",
                                      "contracts/flat_base_finetune_v3.json", "contracts/flat_base_finetune_v4.json",
                                      "contracts/flat_base_finetune_v5.json", "contracts/flat_base_finetune_v6.json",
                                      "contracts/flat_base_finetune_v7.json", "contracts/flat_base_finetune_v8.json",
                                      "contracts/flat_base_finetune_v9.json", "contracts/flat_base_finetune_v10.json"))
def test_separately_pinned_worker_and_design_agree(tmp_path, evaluation, num_envs, plan_file):
    config, seed = write_bundle(ROOT, tmp_path / "config", num_envs, evaluation=evaluation, plan_file=plan_file)
    control = load_chassis_control(ROOT / config["control_math_source"])
    role = "quick" if evaluation else "train"
    receipt = design_preflight(config, control, ROOT, stage="speed", role=role, num_envs=num_envs, seed=seed)
    assert receipt["status"] == "static_passed"
    assert config["policy_dt"] == .02 and config["physics_dt"] == .005
    assert not config["new_asset_curriculum"]["enabled"]
    assert "task_campaign" not in config
    if plan_file.endswith("v5.json"):
        assert bool(config.get("actor_output_layer_only")) == (not evaluation)
        assert config["critic_dim"] == (81 if evaluation else 83)
    if plan_file.endswith(("v6.json", "v7.json", "v8.json", "v9.json", "v10.json")):
        assert bool(config.get("actor_adaptation")) == (not evaluation)
        assert config["critic_dim"] == (81 if evaluation else 83)
        assert config["learning_rate"] == .0001
    if plan_file.endswith(("v7.json", "v8.json", "v9.json", "v10.json")):
        assert bool(config.get("flat_rmcs_command_training")) == (not evaluation)
    for key, value in (("physics_dt", .001), ("asset_manifest_sha256", "0" * 64),
                       ("flat_base_continuation", {})):
        bad = deepcopy(config)
        bad[key] = value
        with pytest.raises(ValueError, match="Preflight rejected"):
            design_preflight(bad, control, ROOT, stage="speed", role=role, num_envs=num_envs, seed=seed)


def test_evaluation_replica_count_is_fixed(tmp_path):
    with pytest.raises(ValueError, match="four equal replicas"):
        write_bundle(ROOT, tmp_path / "bad", 64, evaluation=True)


def test_birth_sampling_keeps_moving_population_and_each_height_total():
    from collections import Counter
    from wheeled_tasks.chassis.flat_base_continuation import read_plan
    from wheeled_tasks.chassis.task import choose_scene_groups
    old, new = (read_plan(ROOT, f"contracts/flat_base_finetune_v{v}.json") for v in (8, 9))
    counts = []
    for plan in (old, new):
        groups = [dict(name=g["name"], fraction=g["fraction"], terrain=["flat"]) for g in plan["train_groups"]]
        counts.append(Counter(name for name, _ in choose_scene_groups(groups, 512)))
    a, b = counts
    assert {n:c for n,c in a.items() if not n.startswith("stand_")} == {
        n:c for n,c in b.items() if not n.startswith("stand_")}
    assert new["episode_seconds"] == old["episode_seconds"] == 20.
    for height in ("low", "mid", "high"):
        long, birth = "stand_" + height, "stand_" + height + "_birth"
        assert a[long] + a[birth] == b[long] + b[birth]
        ratio = ((b[birth] / new["group_episode_seconds"][birth]) /
                 (a[birth] / old["group_episode_seconds"][birth]))
        assert ratio >= 5.5 and new["group_episode_seconds"][birth] == 1.
    assert new["acceptance"] == old["acceptance"]
    assert new["command_branch_restore"] == old["command_branch_restore"]
