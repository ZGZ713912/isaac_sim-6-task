"""Read-only wheel support sampling at every native physical contact update."""
from __future__ import annotations

import torch

from .standing_support import wheel_bottom_clearance


class StandingContactMonitor:
    """Observe the existing closure-check boundary without changing transitions.

    RobustGetupEnv calls closure_gap immediately after its 1ms force/clearance
    update. No physics state, action, success predicate or reward is modified.
    The missing-load counter uses the existing2N contact threshold. Geometric
    gaps are reported independently, including sub-millimetre separation.
    """

    def __init__(self, env, *, record=False):
        if hasattr(env, 'standing_contact_monitor'):
            raise ValueError('A native contact monitor may be installed only once')
        self.env = env
        self.record = record
        self.inside = False
        self.samples = []
        self.trace = []
        self.last = None
        native_gap, native_step = env.closure_gap, env.step

        def gap():
            result = native_gap()
            if self.inside:
                force = env.contact_force[:, env.wheel_ids].clone()
                pose = env.robot.data.body_link_pose_w.torch[:, env.wheel_ids]
                clearance = wheel_bottom_clearance(env.wheel_centers(), pose[:, :, 3:], env.origins[:, 2:3])
                self.samples.append((force, clearance))
            return result

        def step(actions):
            if self.inside:
                raise RuntimeError('Recursive native step')
            self.samples.clear()
            self.inside = True
            try:
                result = native_step(actions)
            finally:
                self.inside = False
            if len(self.samples) != env.decimation or env.dt != .001:
                raise ValueError('Expected exactly five native1ms wheel contact samples')
            force = torch.stack([x[0] for x in self.samples])
            clearance = torch.stack([x[1] for x in self.samples])
            magnitude = force.norm(dim=-1)
            present = magnitude > 2.
            self.last = dict(force=force, clearance=clearance,
                contact_every_ms=present.all(dim=0).all(dim=-1),
                missing_ms=(~present.all(dim=-1)).sum(dim=0),
                normal_load_min_n=force[..., 2].amin(dim=0),
                clearance_peak_m=clearance.amax(dim=0))
            if self.record:
                self.trace.append(dict(force=force.cpu(), clearance=clearance.cpu()))
            return result

        env.closure_gap, env.step = gap, step
        env.standing_contact_monitor = self

    def arrays(self):
        if not self.record or not self.trace:
            raise ValueError('No native contact trace recorded')
        # [policy_frame, physical_substep, environment, wheel, xyz]
        return {key: torch.stack([x[key] for x in self.trace]).numpy() for key in ('force', 'clearance')}
