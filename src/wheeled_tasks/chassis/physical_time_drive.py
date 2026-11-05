"""Physical-clock wheel delay and motor lag for evaluation controller changes.

This opt-in plant preserves its history when a simulator changes its physical
step. It is independent of controller ABI, training, and any simulator API.
The fixed-grid plant remains the default for all existing training runs.
"""
from collections import deque
import math

from .identified_wheel_response import IdentifiedWheelResponse, WHEEL_IDS


def positive_dt(dt):
    if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Physical dt must be finite and positive")
    return float(dt)


class PhysicalTimeWheelResponse(IdentifiedWheelResponse):
    """The identified wheel plant with causal, timestamped fractional delay."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        positive_dt(self.dt)
        if not self.enabled:
            raise ValueError("Physical-clock response requires the explicit identified wheel model")
        self.clock = 0.
        self.zero_prime_time = 0.
        self.samples = deque()
        self.delays = [row["aggregate_delay_s"] if self.cfg["apply_aggregate_delay"] else 0.
                       for row in self.rows]
        # Until the first interval change, keep the parent's exact arithmetic
        # (including its zero-command motoring/braking branch). Equivalent
        # timestamp interpolation can leave a tiny residual at exact zero.
        self.fixed_grid = True
        self.reset_calls = 0

    def set_dt(self, dt):
        """Change only the next physical interval; do not reconstruct the plant."""
        dt = positive_dt(dt)
        if dt != self.dt:
            self.fixed_grid = False
            # Physical timestamps now replace the fixed-grid delay. The
            # inherited ring remains untouched; no second delay is applied.
            self.whole_delay = [0, 0]
            self.fractional_delay = [0., 0.]
        self.dt = dt

    def apply_torque(self, effort, actual_velocity):
        self._check(effort, actual_velocity)
        torch = self.torch
        self.clock += self.dt
        cap = self.model["software_torque_cap_nm"]
        request = effort[:, WHEEL_IDS].clamp(-cap, cap)
        self.samples.append((self.clock, request.clone()))
        oldest_needed = self.clock-max(self.delays)-.02
        while len(self.samples) > 2 and self.samples[1][0] < oldest_needed:
            self.samples.popleft()
        if self.fixed_grid:
            return super().apply_torque(effort, actual_velocity)
        delayed = torch.zeros_like(request)
        for side, delay in enumerate(self.delays):
            target = self.clock-delay
            before, after = None, None
            for timestamp, sample in self.samples:
                if timestamp <= target+1e-12:
                    before = (timestamp, sample)
                if timestamp >= target-1e-12:
                    after = (timestamp, sample)
                    break
            # The pre-birth command is zero, matching the fixed-grid buffer.
            t0, a = before if before is not None else (self.zero_prime_time, torch.zeros_like(request))
            t1, b = after if after is not None else self.samples[-1]
            if target <= self.zero_prime_time:
                continue
            if t1 <= t0+1e-12:
                delayed[:, side] = a[:, side]
            else:
                alpha = min(1., max(0., (target-t0)/(t1-t0)))
                delayed[:, side] = torch.lerp(a[:, side], b[:, side], alpha)
        command = effort.clone()
        command[:, WHEEL_IDS] = delayed
        return super().apply_torque(command, actual_velocity)

    def reset(self, ids=None):
        """Only an explicit episode/disable reset clears the selected lanes."""
        super().reset(ids)
        self.reset_calls += 1
        if ids is None:
            self.samples.clear()
            self.zero_prime_time = self.clock
        else:
            for _, sample in self.samples:
                sample[ids] = 0.

    def history_snapshot(self):
        return dict(clock_s=self.clock, zero_prime_time_s=self.zero_prime_time, reset_calls=self.reset_calls,
            samples=[dict(time_s=t, torque_nm=v.detach().cpu().tolist()) for t, v in self.samples],
            inherited_head=self.head, inherited_history=self.history.detach().cpu().tolist(),
            passive_effort=self.passive_effort.detach().cpu().tolist(),
            drive_bound=self.drive_bound.detach().cpu().tolist(),
            inertia=self.inertia.detach().cpu().tolist())


def set_motor_response_dt(response, dt):
    """Recompute the first-order coefficient without clearing motor state.

    The response may carry an outer domain/transport wrapper; this helper does
    not replace that callable or alter any communication or filter history.
    The caller must separately qualify all those components for the new clock.
    """
    dt = positive_dt(dt)
    response.dt = dt
    if response.enabled:
        values = [1. if tau == 0 else -math.expm1(-dt/tau)
                  for tau in response.cfg["time_constant_s"]]
        response.alpha.copy_(response.alpha.new_tensor(values)[None, :])
