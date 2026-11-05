"""A longer disabled placement window, without privileged enable conditions."""
from copy import deepcopy
import math


def contract():
    return dict(schema="wheel-leg-recovery-passive-window-v1",
        minimum_seconds=1., maximum_seconds=1.2, settled_hold_seconds=.05,
        enable_condition="existing_angular_and_motor_speed_or_maximum_time",
        disabled_effort="existing_zero_motor_effort",
        history="continuous_without_extra_reset",
        linear_velocity_and_contact="diagnostics_only",
        active_budget_and_acceptance="unchanged",
        placement_load_reporting="separate_disabled_active_and_whole_trajectory")


def validate(config):
    if (type(config) is not dict or config != contract()
            or any(type(config.get(k)) not in (int, float) for k in (
                "minimum_seconds", "maximum_seconds", "settled_hold_seconds"))):
        raise ValueError("Use the complete disabled recovery placement window contract")
    return deepcopy(config)


def validate_plan(plan):
    from .control_contract import FINE_PD200_TIMING
    validate(plan["recovery_passive_window"])
    if (plan.get("schema") != "wheel-leg-prone-motion-v4"
            or not plan.get("multi_pose_recovery")
            or plan.get("global_curriculum", {}).get("recovery_focus") is None
            or "recovery_return_horizon" not in plan or "jump_training" in plan
            or plan.get("timing") != FINE_PD200_TIMING):
        raise ValueError("Disabled placement comparison requires the fine recovery-focus return-horizon course")


def configure(plan, settings):
    """Only the existing recovery lane passive timers change."""
    validate_plan(plan)
    config = plan["recovery_passive_window"]
    settings.update(passive_min_seconds=config["minimum_seconds"],
        passive_max_seconds=config["maximum_seconds"],
        passive_hold_seconds=config["settled_hold_seconds"])


class EnableObservations:
    """Read-only probe records at the original policy-boundary enable event.

    Velocity is the state after the completed disabled interval. It is never an
    actor input or a new enable condition. These records are not raw-force logs.
    """
    def __init__(self):
        self.events = []

    def record(self, state):
        for lane in state["enable_event"].nonzero().flatten().tolist():
            self.events.append(dict(lane=lane, elapsed_seconds=float(state["time"][lane]),
                com_world_z_velocity_m_s=float(state["velocity_world"][lane, 2]),
                angular_speed_rad_s=float(state["omega"][lane].norm()),
                motor_speed_peak_rad_s=float(state["motor_velocity"][lane].abs().max()),
                preceding_interval_disabled=bool(state["passive"][lane]),
                disabled_motor_peak_nm=float(state["passive_motor_peak"][lane])))

    def report(self, config, policy_dt):
        config = validate(config)
        checks = dict(enable_event_observed=bool(self.events),
            minimum_disabled_window_respected=bool(self.events) and all(
                e["elapsed_seconds"] >= config["minimum_seconds"]-1e-6 for e in self.events),
            original_maximum_time_bounds_enable=bool(self.events) and all(
                e["elapsed_seconds"] <= config["maximum_seconds"]+policy_dt+1e-6 for e in self.events),
            enabling_interval_zero_effort=bool(self.events) and all(
                e["preceding_interval_disabled"] and e["disabled_motor_peak_nm"] == 0. for e in self.events),
            diagnostic_values_finite=all(all(math.isfinite(e[k]) for k in (
                "elapsed_seconds", "com_world_z_velocity_m_s", "angular_speed_rad_s",
                "motor_speed_peak_rad_s", "disabled_motor_peak_nm")) for e in self.events))
        return dict(checks=checks, events=deepcopy(self.events), new_PPO=0,
            learned_behavior=False, static_equilibrium_certified=False,
            scope="original enable boundary after completed disabled interval; velocity diagnostic only; no raw force audit")
