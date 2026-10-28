"""Real native PPO across a domain boundary, without stale observations or a second learn call."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from wheeled_algo.continuous_rollout import RolloutDomainSwitch
from wheeled_algo.scratch_standing import agent_config


class Env:
    num_envs, num_actions, device, cfg = 8, 6, "cpu", {}

    def __init__(self, marker):
        self.marker, self.actions = marker, []

    def get_observations(self):
        return TensorDict(dict(policy=torch.full((8, 35), self.marker),
                               critic=torch.full((8, 83), self.marker)), batch_size=[8])

    def step(self, action):
        self.actions.append(action.clone())
        return self.get_observations(), -action.square().mean(-1), torch.zeros(8, dtype=torch.long), {}


def run_native(switch_at=None, binding=True):
    torch.manual_seed(381)
    cfg = agent_config(381, "cpu", 2)
    cfg["actor"]["hidden_dims"] = cfg["critic"]["hidden_dims"] = [8]
    runner = OnPolicyRunner(Env(.1), cfg, device="cpu")
    completed, current, observed, totals = [0], [("nominal", 0.)], [], []
    native_act, native_update = runner.alg.act, runner.alg.update

    def act(obs):
        observed.append(obs["policy"].clone())
        return native_act(obs)

    def update():
        result = native_update()
        completed[0] += 1
        return result

    def rebuild(profile, strength):
        assert not torch.is_inference_mode_enabled()
        runner.env = Env(.9)
        current[0] = (profile, strength)

    runner.alg.act, runner.alg.update = act, update
    guard = RolloutDomainSwitch(runner, completed=lambda: completed[0],
        desired=lambda n: ("requested", 1.) if switch_at is not None and n >= switch_at else ("nominal", 0.),
        current=lambda: current[0], rebuild=rebuild) if binding else None
    ids = tuple(id(x) for x in (runner.alg.actor, runner.alg.critic, runner.alg.optimizer, runner.alg.storage))
    native_log = runner.logger.log

    def log(**kw):
        totals.append((kw["it"], kw["start_it"], kw["total_it"]))
        return native_log(**kw)

    runner.logger.log = log
    # Exactly ONE native call. Both algorithm updates, GAE and Adam remain native.
    runner.learn(2, init_at_random_ep_len=False)
    assert completed[0] == 2
    assert totals == [(0, 0, 2), (1, 0, 2)]
    assert ids == tuple(id(x) for x in (runner.alg.actor, runner.alg.critic, runner.alg.optimizer, runner.alg.storage))
    if guard:
        assert guard.switches == int(switch_at is not None)
        guard.close()
        assert runner.alg.act is act
    return deepcopy(runner.alg.save()), observed, torch.get_rng_state().clone()


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_equal(left, right)
    else:
        assert a == b


def test_no_domain_change_preserves_native_models_adam_rng_and_observations():
    assert_equal(run_native(binding=False), run_native(binding=True))


def test_real_native_ppo_first_action_after_switch_uses_new_domain_observation():
    state, observations, _ = run_native(switch_at=1)
    assert len(observations) == 4
    assert all(torch.equal(v, torch.full_like(v, .1)) for v in observations[:2])
    assert all(torch.equal(v, torch.full_like(v, .9)) for v in observations[2:])
    assert all(torch.isfinite(t).all() for t in state["actor_state_dict"].values())
    assert {int(row["step"]) for row in state["optimizer_state_dict"]["state"].values()} == {24}


def test_switch_refuses_a_partial_rollout_before_rebuilding():
    alg = SimpleNamespace(act=lambda obs: obs, storage=SimpleNamespace(step=1))
    runner = SimpleNamespace(alg=alg)
    rebuilt = []
    guard = RolloutDomainSwitch(runner, completed=lambda: 6000,
        desired=lambda _: ("requested", 1.), current=lambda: ("nominal", 0.),
        rebuild=lambda *args: rebuilt.append(args))
    with pytest.raises(RuntimeError, match="empty rollout"):
        alg.act(None)
    assert not rebuilt
    guard.close()


def test_switch_preserves_global_eta_but_clears_partial_episode_totals():
    logger = SimpleNamespace(writer=object(), tot_time=19., tot_timesteps=128,
        cur_reward_sum=torch.ones(2), cur_episode_length=torch.ones(2))
    alg = SimpleNamespace(act=lambda obs: obs, actor=object(), critic=object(), optimizer=object(),
        storage=SimpleNamespace(step=0))
    runner = SimpleNamespace(alg=alg, logger=logger, device="cpu")
    current = [("nominal", 0.)]
    def rebuild(*profile):
        current[0] = profile
        runner.env = Env(.9)
    guard = RolloutDomainSwitch(runner, completed=lambda: 6000,
        desired=lambda _: ("sensors_only", .25), current=lambda: current[0], rebuild=rebuild)
    obs = alg.act(Env(.1).get_observations())
    assert obs["policy"][0, 0] == .9
    assert logger.tot_time == 19. and logger.tot_timesteps == 128
    assert not logger.cur_reward_sum.any() and not logger.cur_episode_length.any()
    assert guard.switches == 1
    guard.close()


@pytest.mark.parametrize("replace", ["optimizer", "writer"])
def test_switch_rejects_replacing_the_learner_or_event_writer(replace):
    alg = SimpleNamespace(act=lambda obs: obs, actor=object(), critic=object(), optimizer=object(),
        storage=SimpleNamespace(step=0))
    runner = SimpleNamespace(alg=alg, logger=SimpleNamespace(writer=object()))
    def rebuild(*_):
        if replace == "optimizer":
            alg.optimizer = object()
        else:
            runner.logger.writer = object()
    guard = RolloutDomainSwitch(runner, completed=lambda: 6000,
        desired=lambda _: ("requested", 1.), current=lambda: ("nominal", 0.), rebuild=rebuild)
    with pytest.raises(RuntimeError, match="live learner or logger"):
        alg.act(None)
    guard.close()
