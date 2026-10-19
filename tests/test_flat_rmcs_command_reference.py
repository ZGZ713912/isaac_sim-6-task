"""Check discontinuous commands, heading capture and per-lane reference clocks."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from flat_rmcs_command_reference import RmcsFlatCommandReference
from wheeled_tasks.chassis.flat_command_reference import install_training_reference


def reference(count=1):
    cfg = dict(nominal_height_m=.305, policy_hz=50, height_reference_rate_max_m_s=2/3,
        wheel_track_m=.4373, wheel_radius_m=.06, wheel_speed_limit_rad_s=90.,
        heading_hold_enabled=True, heading_hold_gain=5., heading_hold_yaw_rate_max_rad_s=4.,
        vx_max_m_s=3., yaw_rate_max_rad_s=4., lateral_acceleration_m_s2=2.06)
    return RmcsFlatCommandReference(count, "cpu", cfg)


def test_first_tick_direct_planar_step_and_six_tick_height_reference():
    ref = reference()
    request = torch.tensor([[3., 0., .23]])
    heights = []
    for tick in range(6):
        result = ref.update(request, torch.zeros(1), torch.tensor([tick]))
        assert result[0, 0] == 3.  # no acceleration ramp
        heights.append(float(result[0, 2]))
    torch.testing.assert_close(torch.tensor(heights),
        torch.tensor([.2916666667, .2783333333, .265, .2516666667, .2383333333, .23]), atol=1e-7, rtol=0)


def test_repeated_observation_does_not_advance_height_or_capture_heading_again():
    ref = reference()
    request = torch.tensor([[1., 0., .43]])
    first = ref.update(request, torch.tensor([.3]), torch.tensor([0])).clone()
    again = ref.update(request, torch.tensor([.8]), torch.tensor([0])).clone()
    assert torch.equal(first, again)
    next_tick = ref.update(request, torch.tensor([.32]), torch.tensor([1]))
    assert float(next_tick[0, 1]) == pytest.approx(-.1, abs=1e-6)


def test_heading_is_kept_across_reverse_and_recaptured_after_manual_turn():
    ref = reference()
    ref.update(torch.tensor([[1., 0., .305]]), torch.tensor([3.13]), torch.tensor([0]))
    reverse = ref.update(torch.tensor([[-1., 0., .305]]), torch.tensor([-3.13]), torch.tensor([1]))
    assert -.12 < float(reverse[0, 1]) < -.11  # wrapped shortest heading error
    manual = ref.update(torch.tensor([[0., -4., .305]]), torch.tensor([-3.13]), torch.tensor([2]))
    assert float(manual[0, 1]) == -4.
    straight = ref.update(torch.tensor([[1., 0., .305]]), torch.tensor([1.]), torch.tensor([3]))
    assert float(straight[0, 1]) == 0.


def test_combined_command_lateral_bound_and_independent_lane_reset():
    ref = reference(2)
    request = torch.tensor([[3., 4., .43], [-3., -4., .23]])
    result = ref.update(request, torch.zeros(2), torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(result[:, 0], torch.tensor([.515, -.515]))
    assert torch.equal(result[:, 1], torch.tensor([4., -4.]))
    retained = result[1].clone()
    ref.reset(torch.tensor([0]))
    assert torch.equal(ref.reference[1], retained)
    assert torch.equal(ref.reference[0], torch.tensor([0., 0., .305]))
    assert ref.last_tick.tolist() == [-1, 0]


@pytest.mark.parametrize("command_values", [[[3.1, 0., .305]], [[0., 4.1, .305]], [[0., 0., float("nan")]]])
def test_invalid_or_outside_evaluated_envelope_requests_are_rejected(command_values):
    with pytest.raises(ValueError):
        reference().update(torch.tensor(command_values), torch.zeros(1), torch.zeros(1, dtype=torch.long))


def test_mixed_training_uses_raw_targets_and_independent_resets_without_advancing_twice():
    from types import SimpleNamespace
    # Lane 0 is standing; lanes 1 and 2 share the reverse skill. Only lane 1
    # takes direct commands; lane 2 keeps the exact preceding legacy ramp.
    env = SimpleNamespace(num_envs=3, device="cpu", policy_dt=.02, references=None,
        cfg=dict(flat_base_continuation=True, evaluation_exact_cases=False, physics_dt=.005, command_slew=True),
        commands=torch.tensor([[0., 0., .23], [-.03, 0., .305], [-.03, 0., .305]]),
        command_target=torch.tensor([[0., 0.], [-3., 0.], [-3., 0.]]),
        episode_length_buf=torch.zeros(3, dtype=torch.long), command_acceleration=torch.full((3,), 1.5))
    env.skills = SimpleNamespace(command_base=torch.tensor([[0., 0., .23], [-3., 0., .305], [-3., 0., .305]]),
        batches=[(torch.tensor([0]), dict(kind="stand", group_name="stand")),
                 (torch.tensor([1, 2]), dict(kind="backward", group_name="backward"))], yaw=lambda: torch.zeros(3))
    env.reset = lambda ids: env.episode_length_buf.__setitem__(ids, 0)
    env.get_observations = lambda: env.commands.clone()
    rng = torch.get_rng_state().clone()
    result = install_training_reference(env, dict(contract=reference().cfg, moving_direct_fraction=.5,
                                                  standing_direct_fraction=1.))
    assert torch.equal(rng, torch.get_rng_state())
    assert result["population"]["backward"] == dict(total=2, direct=1, ramped=1)
    first = env.get_observations()
    torch.testing.assert_close(first, torch.tensor([[0., 0., .2916666667], [-3., 0., .305], [-.03, 0., .305]]))
    assert torch.equal(first, env.get_observations())
    for _ in range(5):
        env.episode_length_buf += 1
        env.get_observations()
    assert float(env.commands[0, 2]) == pytest.approx(.23)
    assert torch.equal(env.skills.command_base[:, 2], torch.tensor([.23, .305, .305]))
    # A stop is an immediate request, and resetting one lane leaves the other
    # reference clock and height untouched.
    env.command_target[1] = 0.
    env.episode_length_buf += 1
    assert env.get_observations()[1, 0] == 0.
    retained = env.rmcs_training_reference.reference[1:].clone()
    env.reset(torch.tensor([0]))
    assert torch.equal(retained, env.rmcs_training_reference.reference[1:])
    assert float(env.get_observations()[0, 2]) == pytest.approx(.2916666667)


def evaluation_env(physics_dt, diagnostic):
    from types import SimpleNamespace
    return SimpleNamespace(num_envs=1, device="cpu", references=None, policy_dt=.02,
        cfg=dict(flat_base_continuation=True, evaluation_exact_cases=True, policy_dt=.02,
                 physics_dt=physics_dt, evaluation=dict(diagnostic_only=diagnostic)),
        cases=dict(forward=dict(command=[3., 0., .305], terrain="flat", task="survive")),
        scene_groups=["forward"], commands=torch.zeros(1, 3), command_acceleration=torch.ones(1),
        episode_length_buf=torch.zeros(1, dtype=torch.long), skills=SimpleNamespace(yaw=lambda: torch.zeros(1)),
        reset=lambda ids: None, get_observations=lambda: None)


def test_clock_isolation_preserves_commands_rng_and_does_not_change_physics():
    from flat_rmcs_command_reference import install_reference
    old, new = evaluation_env(.001, True), evaluation_env(.005, True)
    rng = torch.get_rng_state().clone()
    install_reference(old, reference().cfg)
    install_reference(new, reference().cfg, diagnostic_training_clock=True)
    for tick in range(8):
        old.episode_length_buf.fill_(tick)
        new.episode_length_buf.fill_(tick)
        old.get_observations(); new.get_observations()
        assert torch.equal(old.commands, new.commands)
    assert torch.equal(rng, torch.get_rng_state())
    assert old.cfg["physics_dt"] == .001 and new.cfg["physics_dt"] == .005


@pytest.mark.parametrize("physics_dt,diagnostic,flag", [(.005, False, True), (.005, True, False), (.001, True, True)])
def test_clock_isolation_cannot_be_used_as_an_ordinary_deployment_replay(physics_dt, diagnostic, flag):
    from flat_rmcs_command_reference import install_reference
    with pytest.raises(ValueError):
        install_reference(evaluation_env(physics_dt, diagnostic), reference().cfg, diagnostic_training_clock=flag)
