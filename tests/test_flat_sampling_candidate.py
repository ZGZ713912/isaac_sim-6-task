"""Exercise actual command sampling for the multi-height/start sampling continuation."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.skill_commands import SkillCommands
from wheeled_tasks.chassis.task import choose_scene_groups

ROOT = Path(__file__).resolve().parents[1]
OLD = json.loads((ROOT / "contracts/flat_base_finetune_v9.json").read_text())
NEW = json.loads((ROOT / "contracts/flat_base_finetune_v10.json").read_text())


def population(plan, n):
    return Counter(name for name,_ in choose_scene_groups(
        [dict(name=g["name"], fraction=g["fraction"], terrain=["flat"]) for g in plan["train_groups"]], n))


@pytest.mark.parametrize("n", OLD["allowed_num_envs"])
def test_sampler_population_can_be_allocated_for_every_declared_size(n):
    counts = population(NEW, n)
    assert sum(counts.values()) == n and len(counts) == len(NEW["train_groups"])


def test_512_population_and_long_retention_are_preserved():
    old, new = population(OLD, 512), population(NEW, 512)
    for name, count in old.items():
        if name in ("forward_edge3", "backward_edge3"):
            assert new[name] + new[name+"_start"] == count
            before = count / OLD["episode_seconds"]
            after = new[name] / NEW["episode_seconds"] + new[name+"_start"] / NEW["group_episode_seconds"][name+"_start"]
            assert after / before >= 5.
        else:
            assert new[name] == count
    for key in ("acceptance", "physics_dt", "policy_dt", "stationary_tracking", "training_command_reference",
                "actor_adaptation", "feedback_training_scope", "command_branch_restore", "learning_rate",
                "prior_discontinued_optimizer_updates", "additional_critic_warmup_updates"):
        assert NEW[key] == OLD[key]
    for group in OLD["train_groups"]:
        if group["name"].startswith("stand_"):
            assert group == next(g for g in NEW["train_groups"] if g["name"] == group["name"])


@pytest.mark.parametrize("kind", ("forward_edge3", "backward_edge3_start", "turn_right", "start_stop_3_0", "curve_left"))
def test_actual_sampler_covers_nominal_and_packaged_height_endpoints_without_new_standing_commands(kind):
    spec = deepcopy(next(g["skill"] for g in NEW["train_groups"] if g["name"] == kind))
    count = 8192
    generator = torch.Generator().manual_seed(191019)
    env = SimpleNamespace(num_envs=count, device="cpu", global_actor_update=510, stage_actor_update=510,
        training_transitions=640*512*96, scene_groups=[kind]*count,
        cfg=dict(skill_specs={kind:spec}, new_asset_curriculum=dict(enabled=False)),
        random=lambda n:torch.rand(n, generator=generator), policy_dt=.02,
        commands=torch.zeros(count,3), command_target=torch.zeros(count,2),
        mode=torch.zeros(count,dtype=torch.long), push_enabled=torch.zeros(count,dtype=torch.bool))
    for name in ("command_clock", "height_clock", "push_clock", "episode_limits", "episode_length_buf"):
        setattr(env,name,torch.zeros(count))
    sampler = SkillCommands(env)
    sampler.sample(torch.arange(count))
    height = sampler.command_base[:,2]
    assert all(float(v) == pytest.approx(w, abs=1e-6) for v,w in zip(height.unique().sort().values,[.23,.305,.42],strict=True))
    for value, fraction in ((.23,.25),(.305,.5),(.42,.25)):
        assert torch.isclose(height, height.new_tensor(value)).float().mean().item() == pytest.approx(fraction,abs=.02)
    assert torch.equal(height, env.commands[:,2])
    assert bool((env.command_target.abs().amax(-1)>0).all())
    before = height.clone()
    sampler.sample(torch.arange(count), reset_height=False)
    assert torch.equal(before,sampler.command_base[:,2])
