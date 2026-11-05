"""Explicit six-axis ABI mapping and a measured recovery handoff gate.

The two policies both have 35 observations but have different position,
previous-action and context semantics. This module deliberately maps only
six-axis vectors. It cannot convert one policy's observation frame into the
other's frame, and never changes a robot's physical state.
"""
import math

import torch

from .actuator_response import JOINT_NAMES as CONTROL_JOINT_NAMES

RECOVERY_POLICY_JOINT_NAMES = ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3")
STANDING_POLICY_JOINT_NAMES = ("L_joint1", "R_joint1", "LL_joint1", "RR_joint1", "L_joint3", "R_joint3")


def map_motor_vector(values, source_names, target_names):
    """Reorder six physical-axis values; this does not change units or scale."""
    source, target = tuple(source_names), tuple(target_names)
    expected = set(CONTROL_JOINT_NAMES)
    if (len(source) != 6 or len(target) != 6 or set(source) != expected or set(target) != expected
            or values.ndim != 2 or values.shape[1] != 6):
        raise ValueError("Motor mapping needs exact six-axis joint identities and N x 6 values")
    return values[:, [source.index(name) for name in target]]


class RecoveryHandoffGate:
    """Require original measured recovery plus contiguous strict quiet capture.

    The caller supplies the original task's completed event and current
    recovered/stable predicates. No reward, update counter or nominal pose can
    unlock this gate. The readiness result is not a post-handoff qualification.
    """

    def __init__(self, count, device):
        if type(count) is not int or count < 1:
            raise ValueError("Gate count must be positive")
        self.completed = torch.zeros(count, dtype=torch.bool, device=device)
        self.quiet_seconds = torch.zeros(count, device=device, dtype=torch.float64)

    def update(self, completed_event, recovered, stable, terminated, gravity, gyro,
               control_velocity, dt):
        if type(dt) not in (float, int) or not math.isfinite(dt) or not 0 < dt <= .02:
            raise ValueError("Gate interval must be a completed physical/policy interval <=20ms")
        count = len(self.completed)
        flags = (completed_event, recovered, stable, terminated)
        if any(value.dtype != torch.bool or value.shape != (count,)
               or value.device != self.completed.device for value in flags):
            raise ValueError("Gate needs explicit boolean events and predicates for every lane")
        if gravity.shape != (count, 3) or gyro.shape != (count, 3) or control_velocity.shape != (count, 6):
            raise ValueError("Gate sensor shape differs from lane count")
        self.completed |= completed_event & ~terminated
        self.completed &= ~terminated
        norm = gravity.norm(dim=-1)
        finite = (torch.isfinite(gravity).all(-1) & torch.isfinite(gyro).all(-1)
                  & torch.isfinite(control_velocity).all(-1))
        quiet = (finite & (norm > .5) & (norm < 1.5)
                 & (gravity[:, 2]/norm.clamp_min(1e-12) < -math.cos(math.radians(3.)))
                 & (gyro[:, :2].norm(dim=-1) < .2)
                 & (control_velocity[:, [0, 1, 3, 4]].abs().amax(-1) < .3)
                 & (control_velocity[:, [2, 5]].abs().amax(-1) < .5))
        eligible = self.completed & recovered & stable & ~terminated & quiet
        self.quiet_seconds = torch.where(eligible, self.quiet_seconds+dt, 0.)
        return eligible & (self.quiet_seconds >= .3-1e-12)

    def reset(self, ids=None):
        if ids is None:
            self.completed.zero_()
            self.quiet_seconds.zero_()
        else:
            self.completed[ids] = False
            self.quiet_seconds[ids] = 0.
