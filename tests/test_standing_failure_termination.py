import math
from types import SimpleNamespace
import torch

from wheeled_tasks.chassis.standing_failure_termination import install


def environment(*, fixed=False, native_done=False):
    state = dict(gravity=torch.tensor([[math.sin(math.radians(25)), 0., -math.cos(math.radians(25))]]),
        height=torch.tensor([.305]), commands=torch.tensor([[0., 0., .305]]),
        requested_command=torch.tensor([[0., 0., .305]]), terminated=torch.tensor([native_done]))
    extras = dict(diagnostics=state, policy_learning_mask=torch.tensor([True]),
        policy_bootstrap_mask=torch.tensor([not native_done]), time_outs=torch.tensor([False]), log={})
    calls = []
    env = SimpleNamespace(cfg={"record_diagnostics": True}, getup_cfg={"motion_fraction": 1., "domains_enabled": False},
        fixed_cases=[{}] if fixed else None, history=SimpleNamespace(last_tick=10),
        step=lambda action: ({"policy": "before"}, torch.tensor([.1]), torch.tensor([native_done]), extras),
        reset=lambda ids: calls.append(ids.tolist()), get_observations=lambda: {"policy": "reset"})
    install(env)
    return env, calls


def test_failed_standing_is_a_real_terminal_transition_not_a_timeout():
    env, resets = environment()
    observation, reward, done, extras = env.step(None)
    assert resets == [[0]] and observation["policy"] == "reset" and done.item()
    assert abs(reward.item() + 2.9) < 1e-6
    assert not extras["time_outs"].item() and not extras["policy_bootstrap_mask"].item()
    assert extras["policy_learning_mask"].item() and extras["diagnostics"]["terminated"].item()


def test_fixed_evaluation_is_never_truncated_or_reset():
    env, resets = environment(fixed=True)
    observation, reward, done, _ = env.step(None)
    assert not resets and observation["policy"] == "before" and not done.item()
    assert abs(reward.item() - .1) < 1e-6


def test_existing_mechanical_failure_is_not_charged_or_reset_twice():
    env, resets = environment(native_done=True)
    _, reward, done, _ = env.step(None)
    assert done.item() and not resets and abs(reward.item() - .1) < 1e-6
