"""Explicit tuning exposure survives the gap randomization overlay."""
from types import SimpleNamespace

import pytest

from train_robust_getup import prepare


@pytest.mark.parametrize("mode,fraction", [("quiet_only", .8), ("side_recovery_focus", .2), ("motion_recovery_focus", .8)])
def test_gap_domains_do_not_replace_explicit_tuning_population(tmp_path, mode, fraction):
    args = SimpleNamespace(output=tmp_path, bank=tmp_path / "unused.json", resume=None,
        no_domains=False, curriculum_updates=128, quiet_leg_alpha=.25,
        quiet_wheel_alpha=.65, domain_scale=.35, num_envs=32, rollout_steps=128,
        mode="train", friction="high", seed=241501, isolated_origins=True,
        gap_robustness=True, gap_severity=.6, gap_actuator_timing="native",
        quiet_only=mode == "quiet_only", recovery_only=mode in ("side_recovery_focus", "motion_recovery_focus"),
        side_recovery_focus=mode == "side_recovery_focus", motion_recovery_focus=mode == "motion_recovery_focus")
    args.bank.write_text("{}")
    _, task = prepare(args)
    assert task["gap_robustness_version"] == "gap-v1"
    assert task["motion_fraction"] == fraction
    assert task["leg_kp_scale_range"] == pytest.approx([.91, 1.09])
    if mode == "quiet_only":
        assert task["quiet_training"]
    elif mode == "side_recovery_focus":
        assert task["recovery_family_weights"] == [0., .05, .05, .35, .35, .15, .05]
        assert task["side_recovery_birth_focus"]
        assert task["motion_push_fraction"] == 1.
    else:
        assert task["recovery_family_weights"] == [.4, .05, .05, .125, .125, .2, .05]
        assert not task["side_recovery_birth_focus"]
        assert task["motion_push_fraction"] == 1.
        assert task["push_force_n"] == 80.
        assert task["push_torque_nm"] == 32.
        assert task["push_start_seconds"] == 5.
        assert task["push_duration_seconds"] == .3
