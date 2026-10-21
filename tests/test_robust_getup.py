"""Likelihood, reset isolation and quiet-control regressions, before native PPO."""
import torch
from torch import nn

from wheeled_algo.robust_getup import RobustGetupActor, SensorConditionedMean, quiet_gate, tilt_gate
from wheeled_tasks.chassis.robust_getup import RobustDomain, settings


def observation(count=1):
    obs = torch.zeros(count, 35)
    obs[:, 9] = -1.
    return obs


def test_quiet_control_reduces_target_jumps_and_releases_for_motion_and_disturbance():
    parent = nn.Linear(35, 6)
    with torch.no_grad():
        parent.weight.zero_()
        parent.bias.fill_(1.)
    actor = RobustGetupActor(parent)
    obs = observation(4)
    obs[1, 0], obs[2, 2], obs[3, 4] = 1., 1., 1.
    result = actor(obs)
    torch.testing.assert_close(result[0], actor.quiet_alpha)
    torch.testing.assert_close(result[1:], torch.ones(3, 6))


def test_large_leg_angles_do_not_trap_recovered_body_in_recovery_head():
    obs = observation(2)
    obs[:, 10:14] = 2.
    obs[1, 9] = 1.
    assert tilt_gate(obs).flatten().tolist() == [0., 1.]
    assert quiet_gate(obs)[1] == 0.


def test_stationary_wheels_and_legs_both_receive_learning_gradients():
    actor = RobustGetupActor(nn.Linear(35, 6))
    actor(observation()).sum().backward()
    assert actor.quiet[-1].bias.grad.abs().min() > 0
    assert all(p.grad is None for p in actor.parent.parameters())


def test_native_likelihood_uses_actual_state_dependent_exploration():
    from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
    actor = SensorConditionedMean(RobustGetupActor(nn.Linear(35, 6)))
    obs = observation(3)
    obs[1, 0], obs[2, 9] = 1., 1.
    raw = actor(obs)
    distribution = HeteroscedasticGaussianDistribution(6, std_type="log", std_range=(.003, .8))
    distribution.update(raw)
    assert distribution.std[0, 0] < distribution.std[1, 0] < distribution.std[2, 0]
    action = distribution.sample()
    expected = torch.distributions.Normal(raw[:, 0], raw[:, 1].exp()).log_prob(action).sum(-1)
    torch.testing.assert_close(distribution.log_prob(action), expected)
    torch.testing.assert_close(distribution.kl_divergence(distribution.params, distribution.params), torch.zeros(3))


def test_nominal_domain_is_identity_and_passive_is_zero_even_with_history():
    domain = RobustDomain(4, "cpu", settings(domains_enabled=False), 53, .001)
    sensors = torch.randn(4, 18)
    sensors[:, 3:6] /= sensors[:, 3:6].norm(dim=-1, keepdim=True)
    domain.reset(torch.arange(4), sensors)
    torque = torch.randn(4, 6)
    torch.testing.assert_close(domain.apply_torque(torque, torch.zeros(4, dtype=torch.bool)), torque)
    assert domain.apply_torque(torque, torch.ones(4, dtype=torch.bool)).count_nonzero() == 0
    torch.testing.assert_close(domain.observe(sensors, 1), sensors)


def test_noise_is_held_within_one_policy_tick_and_reset_clears_only_selected_history():
    domain = RobustDomain(4, "cpu", settings(nominal_fraction=0.), 53, .001)
    sensors = torch.zeros(4, 18)
    sensors[:, 5] = -1.
    domain.reset(torch.arange(4), sensors)
    first = domain.observe(sensors, 1).clone()
    torch.testing.assert_close(domain.observe(sensors, 1), first, rtol=0, atol=0)
    domain.torque_state.fill_(3.)
    domain.reset(torch.tensor([1]), sensors[1:2])
    assert domain.torque_state[1].count_nonzero() == 0
    torch.testing.assert_close(domain.torque_state[0], torch.full((6,), 3.))
    assert domain.strength.min() >= .85 and domain.strength.max() <= 1.1


def test_exact_side_and_inverted_cases_remain_held_out_when_bank_expands():
    import importlib.util
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("robust_train", root / "scripts/train_robust_getup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bank = json.loads((root / "reports/unified_robust_20261021/pose_bank002.json").read_text())
    cases = module.fixed_cases(bank, 64, 52)
    for case in cases:
        if case["name"].startswith(("left_", "right_", "inverted_")):
            pose = bank["poses"][case["pose_index"]]
            assert max(abs(pose["roll_deg"]), abs(pose["pitch_deg"])) == (180 if case["name"].startswith("inverted") else 90)


def test_getup_square_supports_all_headings_without_long_thin_triangles():
    import numpy as np
    from wheeled_tasks.chassis.getup_ground import flat_patch
    points, indices = flat_patch()
    vertices = np.asarray(points)
    faces = vertices[np.asarray(indices).reshape(-1, 3)]
    assert vertices[:, 0].min() == -48 and vertices[:, 0].max() == 48
    assert vertices[:, 1].min() == -48 and vertices[:, 1].max() == 48
    assert np.ptp(faces, axis=1)[:, :2].max() <= 8.
    assert (np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])[:, 2] > 0).all()
    # 12 s at the fastest requested 3 m/s plus a 4 m runout fits every heading.
    assert 3 * 12 + 4 < 48


def test_exploration_wrapper_moves_all_parameters_with_the_actor():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wrapper = SensorConditionedMean(RobustGetupActor(nn.Linear(35, 6))).to(device)
    assert all(p.device == device for p in wrapper.parameters())
    assert wrapper(observation().to(device)).device == device
