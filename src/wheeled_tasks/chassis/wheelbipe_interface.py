"""Fixed WheelBipe 16001 standing ABI, using encoder/IMU inputs only.

Reference: the authenticated wheelbipe_V14 / wheelbipe25_v3 policy frame
and wheelbipe_port PairedTargets. This is a controller, not a recovery gate.
The caller owns physical dynamics, sampled feedback and actuator histories.
"""
import math

import torch

from .controller_handoff import CONTROL_JOINT_NAMES, STANDING_POLICY_JOINT_NAMES, map_motor_vector


class WheelbipeStandingInterface:
    def __init__(self, nominal_control_q, gas_model, count):
        if nominal_control_q.shape != (6,) or count < 1:
            raise ValueError("Standing ABI requires six nominal motor angles and positive lane count")
        self.nominal = nominal_control_q.clone()
        self.bounds = nominal_control_q.new_tensor(
            [[gas_model["sides"][s]["delta_rad"][i] for i in (0, -1)]
             for s in ("left", "right")], dtype=torch.float64)
        self.previous_action = nominal_control_q.new_zeros(count, 6)

    def canonical_legs(self, control_q):
        pair = control_q[:, [0, 1, 3, 4]].double().unflatten(-1, (2, 2))
        sign = pair.new_tensor([1., -1.])
        delta = sign*(pair[..., 1]-pair[..., 0])
        relative = delta+math.tau*torch.round((self.bounds.mean(-1)-delta)/math.tau)
        first = pair[..., 0]
        second = first+sign*relative
        mean = (first+second)*.5
        center = self.nominal[[0, 1, 3, 4]].double().unflatten(-1, (2, 2)).mean(-1)
        turns = torch.floor((mean-center)/math.tau+.5)
        result = control_q.clone()
        result[:, [0, 1, 3, 4]] = torch.stack(
            (first-turns*math.tau, second-turns*math.tau), -1).flatten(-2).to(control_q)
        return result

    def observations(self, control_q, control_dq, gyro, gravity, command):
        """command is vx, vy, yaw, height; no simulator truth is accepted."""
        count = len(self.previous_action)
        if (control_q.shape != (count, 6) or control_dq.shape != (count, 6)
                or gyro.shape != (count, 3) or gravity.shape != (count, 3)
                or command.shape != (count, 4)):
            raise ValueError("Standing sensor/command ABI shape differs")
        if not all(bool(torch.isfinite(v).all()) for v in (control_q, control_dq, gyro, gravity, command)):
            raise ValueError("Nonfinite standing input")
        q = self.canonical_legs(control_q)-self.nominal
        q = map_motor_vector(q, CONTROL_JOINT_NAMES, STANDING_POLICY_JOINT_NAMES)
        q[:, 4:] = 0.
        dq = map_motor_vector(control_dq, CONTROL_JOINT_NAMES, STANDING_POLICY_JOINT_NAMES)
        context = q.new_zeros(count, 7)
        context[:, 0] = 1.  # Explicit ordinary standing/motion command, no jump.
        return torch.cat((command[:, :3].clamp(-100., 100.), command[:, 3:].clamp(0., 1.)*5.,
            gyro.clamp(-100., 100.)*.5, gravity.clamp(-100., 100.), q.clamp(-100., 100.),
            dq.clamp(-200., 200.)*.1, self.previous_action.clamp(-100., 100.), context), -1)

    def targets(self, raw_action, control_q):
        if raw_action.shape != self.previous_action.shape or not bool(torch.isfinite(raw_action).all()):
            raise ValueError("Standing policy must return finite N x 6 raw actions")
        action = map_motor_vector(raw_action, STANDING_POLICY_JOINT_NAMES, CONTROL_JOINT_NAMES)
        wanted = (self.nominal[[0, 1, 3, 4]]+.5*action[:, [0, 1, 3, 4]]).double().unflatten(-1, (2, 2))
        measured = self.canonical_legs(control_q)[:, [0, 1, 3, 4]].double().unflatten(-1, (2, 2))
        sign = wanted.new_tensor([1., -1.])
        mean = wanted.mean(-1)
        delta = (sign*(wanted[..., 1]-wanted[..., 0])).clamp(self.bounds[:, 0], self.bounds[:, 1])
        turns = torch.floor((measured.mean(-1)-mean)/math.tau+.5)
        legs = torch.stack((mean-sign*delta*.5+turns*math.tau,
                            mean+sign*delta*.5+turns*math.tau), -1).flatten(-2).to(control_q)
        self.previous_action.copy_(raw_action)
        return legs, (10.*action[:, [2, 5]]).clamp(-90., 90.)
