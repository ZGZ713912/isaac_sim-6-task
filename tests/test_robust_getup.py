"""Likelihood, reset isolation and quiet-control regressions, before native PPO."""
import torch
from torch import nn

from wheeled_algo.robust_getup import RobustGetupActor, SensorConditionedMean, quiet_gate, tilt_gate
from wheeled_tasks.chassis.robust_getup import RobustDomain, settings


def observation(count=1):
    obs = torch.zeros(count, 35)
    obs[:, 9] = -1.
    return obs


def test_unified_initialization_allows_only_the_declared_new_quiet_heads():
    import pytest
    for experts in (False, True):
        actor = RobustGetupActor(nn.Linear(35, 6), quiet_height_experts=experts)
        state = {key: value.clone() for key, value in actor.state_dict().items()
                 if not key.startswith(("quiet.", "quiet_low.", "quiet_high."))
                 and key not in ("quiet_alpha", "quiet_bound")}
        actor.initialize_from_unified(state)
        del state["recovery.0.bias"]
        with pytest.raises(ValueError, match="Unexpected unified-to-robust"):
            actor.initialize_from_unified(state)


def test_cloned_height_experts_preserve_original_function_exactly():
    from copy import deepcopy
    torch.manual_seed(12)
    old = RobustGetupActor(nn.Linear(35, 6))
    torch.nn.init.normal_(old.quiet[-1].weight, std=.02)
    new = deepcopy(old)
    new.quiet_height_experts = True
    new.quiet_low, new.quiet_high = deepcopy(old.quiet), deepcopy(old.quiet)
    obs = observation(256)
    obs[:, 3] = torch.linspace(.9, 2.25, len(obs))
    obs[:, 10:28] = torch.randn(256, 18) * .1
    assert torch.equal(old(obs), new(obs))


def test_anchor_height_gradients_reach_only_its_own_quiet_expert():
    actor = RobustGetupActor(nn.Linear(35, 6), quiet_height_experts=True)
    for height, name in ((.23, "quiet_low"), (.305, "quiet"), (.42, "quiet_high")):
        actor.zero_grad(set_to_none=True)
        obs = observation()
        obs[:, 3] = height * 5.
        result = actor.quiet_correction(torch.ones_like(obs), obs)
        result.sum().backward()
        for head in ("quiet_low", "quiet", "quiet_high"):
            grad = getattr(actor, head)[-1].bias.grad
            assert grad is not None
            assert bool(grad.abs().sum() > 0) == (head == name)


def test_native_float32_height_observations_use_exact_endpoints_and_frozen_gradients():
    actor = RobustGetupActor(nn.Linear(35, 6), quiet_height_experts=True, quiet_native_height_anchors=True)
    with torch.no_grad():
        actor.quiet_low[-1].bias.fill_(.2)
        actor.quiet_high[-1].bias.fill_(-.3)
    heights = torch.tensor([.23, .305, .42], dtype=torch.float32) * 5.
    assert heights[1] != torch.tensor(1.525, dtype=torch.float32)
    for height, name in zip(heights, ("quiet_low", "quiet", "quiet_high")):
        obs = observation()
        obs[:, 3] = height
        actor.zero_grad(set_to_none=True)
        features = torch.ones_like(obs)
        value = actor.quiet_correction(features, obs)
        assert torch.equal(value, torch.tanh(getattr(actor, name)(features)))
        value.sum().backward()
        for head in ("quiet_low", "quiet", "quiet_high"):
            assert bool(getattr(actor, head)[-1].bias.grad.abs().sum() > 0) == (head == name)


def test_intermediate_height_is_continuous_and_nonquiet_output_is_unchanged():
    from copy import deepcopy
    old = RobustGetupActor(nn.Linear(35, 6))
    new = deepcopy(old)
    new.quiet_height_experts = True
    new.quiet_low, new.quiet_high = deepcopy(old.quiet), deepcopy(old.quiet)
    with torch.no_grad():
        new.quiet_low[-1].bias.fill_(.3)
        new.quiet_high[-1].bias.fill_(-.2)
    obs = observation(6)
    obs[:, 3] = torch.tensor([1.15, 1.525, 2.1, 1.3375, 1.8125, 1.525])
    value = new.quiet_correction(torch.ones_like(obs), obs)
    torch.testing.assert_close(value[3], (value[0] + value[1]) * .5)
    torch.testing.assert_close(value[4], (value[1] + value[2]) * .5)
    obs[:3, 0] = 1.
    obs[3:, 9] = 1.
    assert torch.equal(old(obs), new(obs))


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


def test_normal_leg_lean_keeps_balance_authority_and_fallen_states_keep_getup():
    import math
    obs = observation(5)
    obs[:, 9] = torch.tensor([-math.cos(math.radians(40)), -math.cos(math.radians(40)),
                              0., 1., -1.])
    obs[1, 10:14] = 2.
    obs[4, 10:14] = 2.
    gate = tilt_gate(obs, True).flatten()
    assert gate[0] == 0.
    assert 0. < gate[1] < 1.
    assert gate[2:].tolist() == [1., 1., 0.]


def test_stance_gate_retains_parent_during_requested_motion_lean():
    parent = nn.Linear(35, 6)
    with torch.no_grad():
        parent.weight.zero_()
        parent.bias.fill_(.2)
    actor = RobustGetupActor(parent, stance_aware_recovery=True)
    with torch.no_grad():
        actor.recovery[-1].bias.fill_(1.)
    obs = observation()
    obs[:, 0], obs[:, 9] = .7, -.8
    torch.testing.assert_close(actor(obs), torch.full((1, 6), .2), atol=0, rtol=0)


def test_collapsed_knee_does_not_release_recovery_when_common_phase_looks_upright():
    obs = observation(3)
    obs[:, 9] = -.8
    obs[1, 10:14] = torch.tensor([.25, .82, -.21, .20])
    obs[2, 10:14] = torch.tensor([.1, .12, -.1, -.12])
    assert tilt_gate(obs, True)[1] == 0.
    gate = tilt_gate(obs, True, True)
    assert gate[1] > .3
    assert gate[0] == gate[2] == 0.


def test_shape_aware_recovery_still_releases_at_upright_with_any_knee_shape():
    obs = observation(4)
    obs[:, 10:14] = torch.tensor([.25, .82, -.21, .20])
    assert torch.equal(tilt_gate(obs, True, True), torch.zeros(4, 1))


def test_shape_aware_capture_preserves_exact_upright_mean_and_distribution():
    from copy import deepcopy
    actor = RobustGetupActor(nn.Linear(35, 6), stance_aware_recovery=True)
    candidate = deepcopy(actor)
    candidate.recovery_shape_aware = True
    obs = observation(8)
    obs[:, 10:28] = torch.randn(8, 18)
    torch.testing.assert_close(actor(obs), candidate(obs), rtol=0, atol=0)
    a, b = SensorConditionedMean(actor), SensorConditionedMean(candidate)
    torch.testing.assert_close(a(obs), b(obs), rtol=0, atol=0)


def test_stationary_wheels_and_legs_both_receive_learning_gradients():
    actor = RobustGetupActor(nn.Linear(35, 6))
    actor(observation()).sum().backward()
    assert actor.quiet[-1].bias.grad.abs().min() > 0
    assert all(p.grad is None for p in actor.parent.parameters())


def test_ordinary_straight_motion_can_learn_domain_compensation_on_every_axis():
    actor = RobustGetupActor(nn.Linear(35, 6), canonical_recovery=True, motion_residual_fraction=.25)
    obs = observation()
    obs[:, 0] = 1.
    actor(obs).sum().backward()
    assert actor.arc[-1].bias.grad.abs().min() > 0
    assert all(p.grad is None for p in actor.parent.parameters())


def test_new_motion_compensation_cannot_inject_an_unrequested_standing_action():
    from copy import deepcopy
    base = RobustGetupActor(nn.Linear(35, 6))
    with torch.no_grad():
        base.arc[-1].bias.fill_(.5)
    adaptive = deepcopy(base)
    adaptive.motion_residual_fraction = .25
    obs = observation(8)
    obs[:, 22:28] = torch.randn(8, 6) * .1
    torch.testing.assert_close(adaptive(obs), base(obs), rtol=0, atol=0)


def test_straight_compensation_ablation_preserves_full_arcs_standing_and_fallen_actions():
    from copy import deepcopy
    base = RobustGetupActor(nn.Linear(35, 6), motion_residual_fraction=.25)
    with torch.no_grad():
        base.arc[-1].bias.fill_(.7)
    ablated = deepcopy(base)
    ablated.motion_residual_fraction = 0.
    obs = observation(6)
    obs[0, 0] = 3.  # The ablation must actually affect straight motion.
    obs[1, 2] = 4.
    obs[2, 0], obs[2, 2] = 1., -1.
    obs[3, 0], obs[3, 2] = -.75, 1.
    obs[5, 9], obs[5, 0] = 1., .5
    current, revised = base(obs), ablated(obs)
    assert not torch.equal(current[:2], revised[:2])
    torch.testing.assert_close(current[2:], revised[2:], rtol=0, atol=0)


class ReferenceParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.command = nn.Module()
        self.command.base = nn.Linear(35, 6)
        with torch.no_grad():
            self.command.base.weight.zero_()
            self.command.base.bias.fill_(.2)

    def forward(self, obs):
        return self.command.base(obs) + .4


def test_reference_forward_preserves_low_reverse_turns_and_quiet_actions_exactly():
    from copy import deepcopy
    base = RobustGetupActor(ReferenceParent(), motion_residual_fraction=.25)
    candidate = deepcopy(base)
    candidate.fast_forward_reference = True
    obs = observation(6)
    obs[1, 0], obs[2, 0] = 1., -2.5
    obs[3, 0], obs[3, 2] = 3., 1.
    obs[4, 2], obs[5, 0] = -4., 1.5
    torch.testing.assert_close(base(obs), candidate(obs), rtol=0, atol=0)


def test_reference_forward_uses_original_feedback_at_high_speed_and_blends_continuously():
    actor = RobustGetupActor(ReferenceParent(), fast_forward_reference=True)
    obs = observation(4)
    obs[:, 0] = torch.tensor([1.5, 1.75, 2., 3.])
    actual = actor(obs)
    torch.testing.assert_close(actual[:, 0], torch.tensor([.6, .4, .2, .2]))
    torch.testing.assert_close(actual[2:], actor.parent.command.base(obs[2:]), rtol=0, atol=0)
    assert all(not p.requires_grad for p in actor.parent.parameters())


def test_reference_forward_rejects_parent_without_authenticated_original_feedback():
    import pytest
    with pytest.raises(ValueError, match="authenticated original feedback"):
        RobustGetupActor(nn.Linear(35, 6), fast_forward_reference=True)


def test_reference_speed_interval_preserves_old_contract_and_allows_explicit_lower_range():
    from wheeled_algo.robust_getup import fast_forward_reference_gate
    obs = observation(5)
    obs[:, 0] = torch.tensor([1., 1.25, 1.5, 1.75, 2.])
    torch.testing.assert_close(fast_forward_reference_gate(obs)[:, 0], torch.tensor([0., 0., 0., .5, 1.]))
    actor = RobustGetupActor(ReferenceParent(), fast_forward_reference=True, fast_forward_reference_range=[1., 1.5])
    torch.testing.assert_close(actor(obs)[:, 0], torch.tensor([.6, .4, .2, .2, .2]))
    assert actor.fast_forward_reference_range == (1., 1.5)


def test_reference_speed_interval_rejects_zero_width_reversed_or_nonfinite_ranges():
    import pytest
    for bounds in ([1., 1.], [2., 1.], [-1., 2.], [1., 3.1], [float('nan'), 2.], [1.]):
        with pytest.raises(ValueError, match="transition"):
            RobustGetupActor(ReferenceParent(), fast_forward_reference_range=bounds)


def test_all_yaw_forward_reference_retains_both_requests_and_only_changes_fast_forward():
    from copy import deepcopy
    original = RobustGetupActor(ReferenceParent(), fast_forward_reference=True, fast_forward_reference_range=[1., 1.5])
    revised = deepcopy(original)
    revised.fast_forward_reference_yaw_fade = None
    obs = observation(6)
    obs[0, 0], obs[0, 2] = 3., .5
    obs[1, 0], obs[1, 2] = 1.5, -1.
    obs[2, 0], obs[2, 2] = 1.25, .25
    obs[3, 0], obs[3, 2] = -2.5, 1.
    obs[4, 2] = 4.
    before = obs.clone()
    actual = revised(obs)
    torch.testing.assert_close(actual[:3, 0], torch.tensor([.2, .2, .4]))
    torch.testing.assert_close(actual[3:], original(obs)[3:], rtol=0, atol=0)
    torch.testing.assert_close(obs, before, rtol=0, atol=0)


def test_yaw_reference_interval_requires_a_positive_finite_fade_or_explicit_none():
    import pytest
    for value in (0., -1., 4.1, float('nan')):
        with pytest.raises(ValueError, match="yaw fade"):
            RobustGetupActor(ReferenceParent(), fast_forward_reference_yaw_fade=value)


def test_cyclic_recovery_does_not_change_upright_quiet_actions():
    from copy import deepcopy
    original = RobustGetupActor(nn.Linear(35, 6))
    cyclic = deepcopy(original)
    cyclic.cyclic_leg_targets = True
    cyclic.smooth_knee_targets = True
    cyclic.recovery_leg_step = .3
    cyclic.action_bound[:4].fill_(16.)
    obs = observation(16)
    obs[:, 10:14] = torch.randn(16, 4) * .3
    obs[:, 22:28] = torch.randn(16, 6) * .1
    torch.testing.assert_close(cyclic(obs), original(obs), rtol=0, atol=0)


def test_quiet_feature_changes_leave_requested_motion_bitwise_unchanged():
    from copy import deepcopy
    original = RobustGetupActor(nn.Linear(35, 6))
    improved = deepcopy(original)
    improved.quiet_velocity_feature_scale = 10.
    improved.quiet_bound[4:].fill_(2.)
    obs = observation(8)
    obs[:, 0] = 1.
    obs[:, 16:22] = torch.randn(8, 6) * .1
    torch.testing.assert_close(improved(obs), original(obs), rtol=0, atol=0)


def test_mild_domains_include_nominal_and_scale_physical_units_consistently():
    from wheeled_tasks.chassis.robust_getup import rigid_body_domain
    cfg = settings(domain_scale=.35)
    assert cfg["motor_strength_range"][0] >= .94
    assert cfg["pd_feedback_period_ms"] == [1, 2]
    assert cfg["pd_feedback_delay_ms"] == [0, 1]
    assert cfg["sensor_delay_policy_ticks"] == [0, 1]
    mechanics = rigid_body_domain(True, .35)
    torch.testing.assert_close(torch.tensor(mechanics["base_mass_scale"]), torch.tensor([.965, 1.035]))
    assert abs(mechanics["base_com_offset_m"]["x"][1] - .0028) < 1e-8


def test_high_common_filter_preserves_knee_difference_low_mid_stance_and_motion():
    from copy import deepcopy
    parent = nn.Linear(35, 6)
    with torch.no_grad():
        parent.weight.zero_()
        parent.bias.copy_(torch.tensor([1., 2., -1., -2., .1, -.1]))
    base = RobustGetupActor(parent)
    faster = deepcopy(base)
    faster.quiet_high_common_alpha = .65
    obs = observation(4)
    obs[:, 3] = torch.tensor([.23, .305, .42, .42]) * 5.
    obs[3, 0] = 1.
    a, b = base(obs), faster(obs)
    torch.testing.assert_close(a[[0, 1, 3]], b[[0, 1, 3]], rtol=0, atol=0)
    torch.testing.assert_close(a[:, 4:], b[:, 4:], rtol=0, atol=0)
    torch.testing.assert_close(b[2, [1, 3]] - b[2, [0, 2]], a[2, [1, 3]] - a[2, [0, 2]])
    torch.testing.assert_close(b[2, :4] - a[2, :4], torch.tensor([.6, .6, -.6, -.6]))


def test_diagnostic_sensor_group_keeps_other_delay_and_same_noise_draws():
    from copy import deepcopy
    domain = RobustDomain(2, "cpu", settings(domain_scale=.35), 15, .001)
    initial = torch.zeros(2, 18)
    initial[:, 5] = -1.
    domain.reset(torch.arange(2), initial)
    domain.sensor_delay.fill_(1)
    control = deepcopy(domain)
    domain.evaluation_immediate_features = list(range(6))
    sensors = initial.clone()
    sensors[:, :3] = .2
    sensors[:, 6:] = .4
    observed = domain.observe(sensors, 0)
    reference = control.observe(sensors, 0)
    torch.testing.assert_close(observed[:, 6:], reference[:, 6:], rtol=0, atol=0)
    assert observed[0, 0] > .19 and reference[0, 0] == 0.
    assert torch.equal(domain.rng.get_state(), control.rng.get_state())


def test_leg_prediction_uses_frozen_units_and_preserves_other_requests():
    from copy import deepcopy
    parent = nn.Linear(35, 6)
    with torch.no_grad():
        parent.weight.zero_()
        parent.bias.zero_()
        parent.weight[:4, 10:14] = torch.eye(4)
    base = RobustGetupActor(parent, quiet_alpha=(1.,) * 6)
    predicted = deepcopy(base)
    predicted.quiet_high_leg_prediction_s = .005
    obs = observation(4)
    obs[:, 3] = torch.tensor([.23, .305, .42, .42]) * 5.
    obs[:, 10:14] = .1
    obs[:, 16:20] = .1  # 1 rad/s in the frozen 35D interface.
    obs[3, 0] = 1.
    a, b = base(obs), predicted(obs)
    torch.testing.assert_close(a[[0, 1, 3]], b[[0, 1, 3]], rtol=0, atol=0)
    torch.testing.assert_close(b[2, :4] - a[2, :4], torch.full((4,), .005))
    torch.testing.assert_close(a[:, 4:], b[:, 4:], rtol=0, atol=0)


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


def test_moderate_rehearsal_exposes_both_requests_and_all_four_directions():
    from wheeled_tasks.chassis.arc_rehearsal import moderate_arc_commands
    draws = torch.tensor([[0., .25, 0., .25], [1., .75, 1., .25],
                          [0., .25, 1., .75], [1., .75, 0., .75]])
    requests = moderate_arc_commands(draws)
    assert set(map(tuple, requests.sign().tolist())) == {(-1.,-1.), (1.,-1.), (-1.,1.), (1.,1.)}
    torch.testing.assert_close(requests[:, 0].abs(), torch.tensor([.4,1.25,.4,1.25]))
    torch.testing.assert_close(requests[:, 1].abs(), torch.tensor([.4,1.4,1.4,.4]))
    assert (requests.prod(-1).abs() <= 2.06).all()


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
