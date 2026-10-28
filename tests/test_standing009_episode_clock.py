"""Execute the native timeout expression against declared episode boundaries."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / "contracts/wheel_leg_009_reproduction_v2.json").read_text())


def native_timeout(ticks, limits, global_limit):
    # Importing the full environment requires a running Isaac application.
    # Evaluate the actual step expression, rather than copying its rule here.
    tree = ast.parse((ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RobustGetupEnv")
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    assignment = next(n for n in ast.walk(step) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "timeout" for t in n.targets))
    expression = ast.Expression(assignment.value)
    env = SimpleNamespace(episode_length_buf=torch.tensor(ticks),
                          episode_limits=torch.tensor(limits), max_episode_length=global_limit)
    return eval(compile(expression, "native_timeout", "eval"), {"self": env})


def test_movement_ends_at_12_seconds_while_standing_continues_to_30():
    motion = round(PLAN["repair"]["motion_episode_seconds"] * PLAN["policy_hz"])
    standing = round(PLAN["repair"]["standing_episode_seconds"] * PLAN["policy_hz"])
    assert native_timeout([motion - 1, motion, motion, standing - 1, standing],
                          [motion, motion, standing, standing, standing], standing).tolist() == [
                              False, True, False, False, True]


def test_uniform_historical_foundation_and_full_30_second_evaluation_are_unchanged():
    for limit in (1600, 6000):
        ticks = [limit - 1, limit, limit + 1]
        assert native_timeout(ticks, [limit] * len(ticks), limit).tolist() == [False, True, True]


def test_motion_roles_in_cold_evaluation_still_get_the_full_30_seconds():
    assert native_timeout([2400, 5999, 6000], [6000] * 3, 6000).tolist() == [False, False, True]
