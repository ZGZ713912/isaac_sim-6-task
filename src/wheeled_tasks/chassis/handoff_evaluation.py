"""One-world, nominal-domain continuous evaluation on the original recovery plant.

Recovery stepping and completion/terminal predicates remain with ProneMotionEnv.
Only an actual completion plus quiet capture may transfer motor ownership.
The independent standing reference explicitly starts in a different mode.
"""
from collections import deque
import torch

from .controller_handoff import RecoveryHandoffGate
from .physical_time_drive import PhysicalTimeWheelResponse, set_motor_response_dt
from .wheelbipe_interface import WheelbipeStandingInterface


class CompletedPhysicsClock:
    """Observe dt submitted to the native solver and committed by fetch_results.

    Some installed PhysX bindings omit the simulation-time getter used by
    Isaac Lab's scene adapter. This forwards the same native object and records
    completed calls, without inventing a second physics engine clock.
    """
    def __init__(self, native):
        self.native = native
        self.time_s = 0.
        self.steps = 0
        self.pending_dt = None
        self.last_dt = None

    def simulate(self, dt, *args, **kwargs):
        if self.pending_dt is not None or dt <= 0:
            raise RuntimeError("Uncommitted or invalid native physics interval")
        result = self.native.simulate(dt, *args, **kwargs)
        self.pending_dt = dt
        return result

    def fetch_results(self, *args, **kwargs):
        if self.pending_dt is None:
            raise RuntimeError("Native fetch has no submitted interval")
        result = self.native.fetch_results(*args, **kwargs)
        self.last_dt = self.pending_dt
        self.time_s += self.last_dt
        self.steps += 1
        self.pending_dt = None
        return result

    def __getattr__(self, name):
        return getattr(self.native, name)


def contact_summary(records, body_names, enabled_at, handoff_at):
    """Raw per-body 20ms integral, including a fractional oldest interval."""
    import numpy as np
    groups = {"chassis": [i for i, n in enumerate(body_names) if n == "base_link"],
              "gimbal": [i for i, n in enumerate(body_names) if n.startswith("M_")],
              "wheels": [i for i, n in enumerate(body_names) if n in ("L_link3", "R_link3")]}
    groups["legs"] = [i for i in range(len(body_names)) if not any(i in ids for ids in groups.values())]
    total = np.zeros(len(body_names))
    peak = total.copy()
    history = deque()
    if enabled_at is not None:
        for r in records:
            if r["time"] <= enabled_at+1e-10 or (handoff_at is not None and r["time"] > handoff_at+1e-10):
                continue
            force = np.maximum(np.asarray(r["contact_force"])[:, 2], 0.)
            start, end = r["time"]-r["dt"], r["time"]
            history.append([start, end, force])
            total += force*(end-start)
            cutoff = end-.02
            while history and history[0][1] <= cutoff+1e-12:
                left, right, old = history.popleft()
                total -= old*(right-left)
            if history and history[0][0] < cutoff:
                left, right, old = history[0]
                total -= old*(cutoff-left)
                history[0][0] = cutoff
            peak = np.maximum(peak, total)
    return dict(measurement_available=enabled_at is not None, group_order=list(groups),
        group_body_names={g: [body_names[i] for i in ids] for g, ids in groups.items()},
        per_body_positive_world_z_20ms_peak_ns=peak.tolist(),
        group_positive_world_z_20ms_peak_ns={g: float(peak[ids].max()) if ids else 0. for g, ids in groups.items()},
        force_scope="native ground contact resultant per rigid body; static load included; no structural stress qualification")


def construct_evaluation_env(env_class, *args, **kwargs):
    """Scope the opt-in plant factory to one independent environment construction.

    The original hash-bound training environment is unchanged. No dynamics
    step occurs while another caller can select this factory; this entry point
    is restricted to the standalone one-world playback process.
    """
    if kwargs.get("run_role") != "playback" or kwargs.get("num_envs") != 1:
        raise ValueError("Physical-clock plant selection requires one-world playback")
    from . import env as base
    original = base.IdentifiedWheelResponse
    if original is not PhysicalTimeWheelResponse.__bases__[0]:
        raise RuntimeError("Another evaluation already owns the wheel constructor")
    try:
        base.IdentifiedWheelResponse = PhysicalTimeWheelResponse
        return env_class(*args, **kwargs)
    finally:
        base.IdentifiedWheelResponse = original


def _serial(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, torch.Generator):
        return value.get_state().cpu().tolist()
    if isinstance(value, (list, tuple, deque)):
        return [_serial(v) for v in value]
    if isinstance(value, dict):
        return {k: _serial(v) for k, v in value.items()}
    return value


def history_snapshot(env):
    result = {"wheel": env.identified_wheels.history_snapshot(),
              "recovery_actions": _serial(env.actions), "recovery_previous_actions": _serial(env.previous_actions)}
    for name in ("actuator_response", "domain", "getup_feedback", "gimbal", "native_wheel_encoder"):
        obj = getattr(env, name, None)
        if obj is None:
            result[name] = None
            continue
        result[name] = {k: _serial(v) for k, v in vars(obj).items()
            if isinstance(v, (torch.Tensor, torch.Generator))
            or k in ("torque_cursor", "sensor_cursor", "last_sensor_tick", "cursor", "tick")}
        # Recomputed integration coefficients are configuration, not history.
        result[name].pop("alpha", None)
    return result


def admit_nominal(env):
    """Reject unqualified clocks before a controller change can occur."""
    if env.num_envs != 1 or not isinstance(env.identified_wheels, PhysicalTimeWheelResponse):
        raise ValueError("Continuous evaluation needs one world and a physical wheel clock selected at construction")
    if env.dt != .001 or env.policy_dt != .01 or env.control_audit.control_stride != 5:
        raise ValueError("Recovery rates must be original 1000/100/200Hz")
    for name in ("signal_delay", "usb_transport", "feedback_transport", "command_transport", "perturbations"):
        if getattr(env, name, None) is not None:
            raise ValueError("Unqualified active handoff clock: "+name)
    domain = env.domain
    checks = [not bool(domain.enabled.any()), bool((domain.strength == 1.).all()),
        bool((domain.kp == 1.).all()), bool((domain.kd == 1.).all()), bool((domain.spring == 1.).all()),
        bool((domain.motor_alpha == 1.).all())]
    for name in ("torque_delay", "torque_delay_fraction", "sensor_delay", "bias", "sigma"):
        checks.append(not bool(getattr(domain, name).any()))
    feedback = env.getup_feedback
    checks += [not bool(feedback.enabled.any()), bool((feedback.period == 1).all()),
        not bool(feedback.delay.any()), not bool(feedback.delay_fraction.any()), not bool(feedback.bias.any())]
    mount = domain.imu_mount
    if mount is not None:
        checks += [not bool(mount.fixed.any()), not bool(mount.amplitude.any())]
    if not all(checks):
        raise ValueError("Randomized motor/sensor histories require a separate physical-clock qualification")
    return {"nominal_only": True, "active_communication_modules": [],
            "recovery_rates_hz": [1000, 100, 200], "checks": checks}


class HandoffEvaluation:
    def __init__(self, env, *, standing_reference=False):
        self.env = env
        self.admission = admit_nominal(env)
        self.gate = RecoveryHandoffGate(1, env.device)
        self.standing = WheelbipeStandingInterface(env.v5.nominal, env.v5.gas_compensation.model, 1)
        self.mode = "standing_reference" if standing_reference else "recovery"
        self.time = 0.
        self.enabled_at = self.handoff_at = None
        self.transition = None
        self.original_termination = False
        self.records = []
        self.policy_records = []
        self.controller_frames = []
        self.last_native_step = env.sim.get_physics_step_count()
        # class_type may be an immutable lazy ResolvableString. Its bound
        # classmethod identifies the actual backend class that performs step.
        manager = env.sim.physics_manager.step.__self__
        self.native_clock = CompletedPhysicsClock(manager._physx_sim)
        manager._physx_sim = self.native_clock
        self.command = env.v5.nominal.new_tensor([[0., 0., 0., .305]])
        self.last_motor_effort = torch.zeros(1, 6, device=env.device)
        native_wheel = env.identified_wheels.apply_torque
        def wheel(effort, velocity):
            result = native_wheel(effort, velocity)
            self.last_motor_effort.copy_(result)
            return result
        env.identified_wheels.apply_torque = wheel
        native_updated = env.control_audit.physics_updated
        def physics_updated(dt):
            native_updated(dt)
            # This hook follows the original robot.update, before the task's
            # reductions. Query raw ground contact resultants independently.
            self.read_contacts(dt)
            self.observe_physics(dt, self.last_motor_effort, env.robot.data.applied_torque.torch,
                ~(env.passive | env.finished))
        env.control_audit.physics_updated = physics_updated
        if standing_reference:
            self._switch(reference=True)

    def read_contacts(self, dt):
        import warp as wp
        env = self.env
        for ids, view in env.contact_views:
            matrix = wp.to_torch(view.get_contact_force_matrix(dt=dt))
            env.contact_force[ids] = matrix.reshape(len(ids), env.body_count, -1, 3).sum(2)

    def observe_physics(self, dt, motor_effort, physical_effort, active):
        env = self.env
        native_step = env.sim.get_physics_step_count()
        native_time = self.native_clock.time_s
        native_dt = self.native_clock.last_dt
        if (native_step != self.last_native_step+1 or self.native_clock.steps != len(self.records)+1
                or native_dt != dt or env.sim.get_physics_dt() != dt):
            raise RuntimeError(f"Native physics clock differs: step={native_step-self.last_native_step}, dt={native_dt}, expected={dt}")
        self.last_native_step = native_step
        self.time += dt
        data = env.robot.data
        self.records.append(dict(time=self.time, dt=dt, standing=self.mode != "recovery", active=_serial(active),
            native_physics_step=native_step, native_time_s=native_time, native_dt_s=native_dt,
            contact_force=_serial(env.contact_force[0]), motor_effort=_serial(motor_effort[0]),
            physical_effort=_serial(physical_effort[0]),
            joint_position=_serial(data.joint_pos.torch[0]), joint_velocity=_serial(data.joint_vel.torch[0]),
            root_pose=_serial(data.root_link_pose_w.torch[0]),
            root_com_velocity=_serial(data.root_com_vel_w.torch[0]),
            gravity=_serial(data.projected_gravity_b.torch[0]),
            gyro=_serial(data.root_com_ang_vel_b.torch[0]), closure_gap=float(env.closure_gap()[0]),
            gimbal_target=_serial(env.gimbal.target[0]) if env.gimbal is not None else [],
            wheel_clock=env.identified_wheels.clock,
            wheel_drive_bound=_serial(env.identified_wheels.drive_bound[0]),
            requested_motor_effort=_serial(env.v5.requested_motor_effort[0]),
            wheel_input_effort=_serial(env.identified_wheels.samples[-1][1][0]),
            domain_torque_state=_serial(env.domain.torque_state[0]),
            feedback_position=_serial(env.getup_feedback.q[0]),
            feedback_velocity=_serial(env.getup_feedback.dq[0])))

    def observe_recovery(self, diagnostics):
        if bool(diagnostics["enable_event"][0]) and self.enabled_at is None:
            self.enabled_at = self.time
        self.original_termination |= bool(diagnostics["terminated"][0] or diagnostics["done"][0])
        ready = self.gate.update(diagnostics["recovery_completed_event"], diagnostics["recovered"],
            diagnostics["stable"], diagnostics["terminated"] | diagnostics["done"],
            diagnostics["gravity"], diagnostics["omega"], diagnostics["motor_velocity"], self.env.policy_dt)
        self.policy_records.append({"time": self.time, "completed": bool(diagnostics["recovery_completed_event"][0]),
            "recovered": bool(diagnostics["recovered"][0]), "stable": bool(diagnostics["stable"][0]),
            "terminated": bool(diagnostics["terminated"][0]), "done": bool(diagnostics["done"][0]),
            "passive": bool(diagnostics["passive"][0]), "ready": bool(ready[0])})
        if bool(ready[0]):
            self._switch(reference=False)
        return self.mode != "recovery"

    def recovery_budget_exhausted(self, active_seconds):
        if self.enabled_at is None:
            return self.time >= self.env.getup_cfg["passive_max_seconds"]+self.env.policy_dt-1e-9
        return self.time-self.enabled_at >= active_seconds-1e-9

    def _switch(self, *, reference):
        env = self.env
        if self.transition is not None or (not reference and not bool(self.gate.quiet_seconds[0] >= .3-1e-12)):
            raise RuntimeError("Handoff requires original completion and uninterrupted capture")
        # Recheck before the first mutation; no nominal-only promise is inferred.
        admit_nominal(env)
        before = history_snapshot(env)
        q, dq = env.robot.data.joint_pos.torch.clone(), env.robot.data.joint_vel.torch.clone()
        root = env.robot.data.root_link_pose_w.torch.clone()
        root_velocity = env.robot.data.root_com_vel_w.torch.clone()
        wheel, motor = env.identified_wheels, env.actuator_response
        wrapped_motor = motor.apply_torque
        wrapped_wheel = wheel.apply_torque
        native_delegate = self.native_clock.native
        from .preflight import SubstepControlAudit
        # Isaac Lab 3 PhysxManager.step reads sim.cfg.dt for every simulate call.
        # Native getter readback is required; no physics reset/recreation occurs.
        env.sim.cfg.dt = .005
        if env.sim.get_physics_dt() != .005:
            raise RuntimeError("Native physics manager did not adopt 200Hz")
        env.dt, env.policy_dt, env.decimation = .005, .02, 4
        env.control_audit = SubstepControlAudit(.005, .02, .005)
        wheel.set_dt(.005)
        set_motor_response_dt(motor, .005)
        env.getup_feedback.dt = env.domain.dt = .005
        env.domain.sensor_policy_dt = .02
        if env.gimbal is not None:
            env.gimbal.dt = .005
        env.v5.settings = dict(env.v5.settings, leg_kp=60., leg_kd=2.)
        self.mode = "standing_reference" if reference else "standing"
        self.handoff_at = None if reference else self.time
        after = history_snapshot(env)
        unchanged = (torch.equal(q, env.robot.data.joint_pos.torch) and torch.equal(dq, env.robot.data.joint_vel.torch)
            and torch.equal(root, env.robot.data.root_link_pose_w.torch)
            and torch.equal(root_velocity, env.robot.data.root_com_vel_w.torch))
        self.transition = dict(time=self.time, reference=reference, before=before, after=after,
            history_identical=before == after, physical_state_unchanged=unchanged,
            same_wheel_object=wheel is env.identified_wheels, same_motor_object=motor is env.actuator_response,
            same_motor_callable=wrapped_motor is motor.apply_torque,
            same_wheel_callable=wrapped_wheel is wheel.apply_torque,
            same_native_physics_object=self.native_clock.native is native_delegate,
            new_actor_previous_action=_serial(self.standing.previous_action),
            actual_handoff=not reference, native_physics_dt=env.sim.get_physics_dt())
        if not all(self.transition[k] for k in ("history_identical", "physical_state_unchanged", "same_wheel_object",
                                               "same_motor_object", "same_motor_callable", "same_wheel_callable",
                                               "same_native_physics_object")):
            raise RuntimeError("Handoff mutated physical/actuator history")

    def standing_observations(self):
        env = self.env
        data = env.robot.data
        return self.standing.observations(data.joint_pos.torch[:, env.ids], data.joint_vel.torch[:, env.ids],
            data.root_com_ang_vel_b.torch, data.projected_gravity_b.torch, self.command)

    def step_standing(self, action):
        """Four actual 5ms PhysX steps per 20ms fixed-policy action."""
        import warp as wp
        from .root_kinematics import update_robot_data
        from .v5_control import V5Control
        env = self.env
        if self.mode == "recovery":
            raise RuntimeError("Standing controller has no motor ownership")
        q = env.robot.data.joint_pos.torch[:, env.ids]
        legs, wheels = self.standing.targets(action, q)
        for _ in range(4):
            q = env.robot.data.joint_pos.torch[:, env.ids]
            dq = env.robot.data.joint_vel.torch[:, env.ids]
            feedback_q, feedback_dq = env.getup_feedback.observe(q, dq)
            canonical = self.standing.canonical_legs(feedback_q)
            # Call original six-axis PD math explicitly, bypassing the recovery
            # gas-feedforward wrapper. The physical spring is still evaluated below.
            tau = V5Control.motor_efforts(env.v5, canonical, feedback_dq, legs, wheels, apply_gas_compensation=False)
            limits = env.v5.gas_compensation.cfg["relative_soft_limits"]
            sign = q.new_tensor([[-1., 1.], [1., -1.]])
            pair = tau[:, [0, 1, 3, 4]].reshape(-1, 2, 2)
            relative = (canonical[:, [0, 1, 3, 4]].reshape(-1, 2, 2)*sign).sum(-1)
            outward = (pair*sign).sum(-1)
            blocked = ((relative >= limits["maximum"]-limits["margin"]) & (outward > 0.)) | (
                (relative <= limits["minimum"]+limits["margin"]) & (outward < 0.))
            tau[:, [0, 1, 3, 4]] = (pair-sign*torch.where(blocked, outward/2., 0.).unsqueeze(-1)).flatten(-2)
            tau = env.actuator_response.apply_torque(tau)
            tau = env.identified_wheels.apply_torque(tau, dq)
            physical = torch.zeros_like(env.nominal)
            physical[:, env.ids] = tau
            if env.leg_friction.enabled:
                physical[:, [env.ids[i] for i in (0, 1, 3, 4)]] += env.leg_friction.effort(dq[:, [0, 1, 3, 4]])
            physical[:, [env.ids[i] for i in (2, 5)]] += env.identified_wheels.passive_effort
            physical[:, env.spring_ids] = env.v5.spring_efforts(env.robot.data.joint_pos.torch[:, env.spring_ids])
            if env.gimbal is not None:
                physical[:, env.gimbal_ids] = env.gimbal.step(env.robot.data.joint_pos.torch[:, env.gimbal_ids],
                    env.robot.data.joint_vel.torch[:, env.gimbal_ids], torch.ones(1, device=env.device, dtype=torch.bool))
            env.robot.set_joint_effort_target_index(target=physical)
            env.robot.write_data_to_sim()
            env.sim.step(render=False)
            update_robot_data(env, .005)
            if env.native_wheel_encoder is not None:
                env.native_wheel_encoder.update(env.robot.data.joint_pos.torch[:, env.ids][:, [2, 5]])
            for ids, view in env.contact_views:
                matrix = wp.to_torch(view.get_contact_force_matrix(dt=.005))
                env.contact_force[ids] = matrix.reshape(len(ids), env.body_count, -1, 3).sum(2)
            self.observe_physics(.005, tau, physical, torch.ones(1, device=env.device, dtype=torch.bool))
