"""Optional cold-evaluation records, without advancing a controller or sampler.

Physical frames follow a completed native step and the original robot update.
Files distinguish motor effort from total physical effort, the scorer's window
from the environment's enable mask, and initial state from measured contacts.
This is an observation archive, not a resumable simulator checkpoint.
"""
from collections import deque
import json
import math
from pathlib import Path
import shutil

import numpy as np


SCHEMA = "wheel-leg-recovery-physical-trace-v1"
CLOCK_COLUMNS = ("time_s", "dt_s", "native_step", "native_fetch_time_s",
                 "native_completed_dt_s", "policy_step")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class ArrayTrace:
    """Fixed layout with a committed prefix; unused capacity is never data."""
    def __init__(self, path, capacity, lanes, fields, *, reserve_bytes=5_000_000_000):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        if type(capacity) is not int or capacity <= 0 or type(lanes) is not int or lanes <= 0:
            raise ValueError("Trace capacity and lane count must be positive integers")
        self.capacity, self.lanes, self.count = capacity, lanes, 0
        self.layout, width = {}, 0
        for name, value in fields.items():
            value = np.asarray(value)
            if value.ndim < 1 or value.shape[0] != lanes or value.dtype.kind not in "bifu":
                raise ValueError("Trace fields need one numeric row per lane: " + name)
            shape = list(value.shape[1:])
            size = math.prod(shape)
            self.layout[name] = dict(offset=width, width=size, shape=shape)
            width += size
        if not width:
            raise ValueError("A trace needs at least one field")
        expected_bytes = capacity * (lanes * width * 4 + len(CLOCK_COLUMNS) * 8) + 65536
        if shutil.disk_usage(self.path).free < expected_bytes + reserve_bytes:
            raise OSError("Insufficient disk space for the bounded raw trace and reserve")
        self.values = np.lib.format.open_memmap(self.path / "values.npy", mode="w+", dtype="float32",
                                               shape=(capacity, lanes, width))
        self.clock = np.lib.format.open_memmap(self.path / "clock.npy", mode="w+", dtype="float64",
                                              shape=(capacity, len(CLOCK_COLUMNS)))
        self.manifest = dict(schema=SCHEMA, status="recording", valid_frames=0, capacity=capacity,
            lanes=lanes, value_dtype="float32", clock_dtype="float64", fields=self.layout,
            clock_columns=list(CLOCK_COLUMNS), unused_capacity_is_data=False,
            expected_maximum_bytes=expected_bytes)
        self.flush()

    def append(self, fields, clock):
        if self.count >= self.capacity:
            raise RuntimeError("Raw trace exceeded its declared physical/policy budget")
        if list(fields) != list(self.layout) or len(clock) != len(CLOCK_COLUMNS):
            raise ValueError("Trace field order or clock shape changed")
        packed = []
        for name, spec in self.layout.items():
            value = np.asarray(fields[name])
            if list(value.shape) != [self.lanes] + spec["shape"]:
                raise ValueError("Trace field shape changed: " + name)
            packed.append(value.reshape(self.lanes, spec["width"]))
        # NaN is retained in diagnostic values (e.g. never recovered_at). Clocks
        # must be finite; archive rows are not accepted by a permissive serializer.
        if not np.isfinite(clock).all():
            raise ValueError("Trace clock must be finite")
        self.values[self.count] = np.concatenate(packed, axis=1)
        self.clock[self.count] = clock
        self.count += 1

    def flush(self, status="recording"):
        self.values.flush()
        self.clock.flush()
        self.manifest.update(valid_frames=self.count, status=status)
        write_json(self.path / "manifest.json", self.manifest)

    def close(self, status="completed"):
        self.flush(status)
        self.values._mmap.close()
        self.clock._mmap.close()


def field_values(values, layout, name):
    spec = layout[name]
    return values[..., spec["offset"]:spec["offset"] + spec["width"]].reshape(
        *values.shape[:-1], *spec["shape"])


def contact_integrals(frames, body_names, *, window_s=.02):
    """Independent double-precision integration of piecewise-constant forces.

Each frame is (completed_end, actual_dt, [lane,body,xyz], scored_and_enabled).
Fractional oldest intervals are removed before each body's peak is reduced.
"""
    from .recovery_impact import GROUPS, body_groups
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError("Impulse window must be positive")
    running = force_peak = impulse_peak = seconds = first = last = None
    history = deque()
    previous_end = None
    frames_seen = 0
    for end, dt, force, enabled in frames:
        force = np.asarray(force, dtype=np.float64)
        enabled = np.asarray(enabled, dtype=bool)
        if (force.ndim != 3 or force.shape[1:] != (len(body_names), 3)
                or enabled.shape != force.shape[:1] or not np.isfinite(force).all()
                or not math.isfinite(end) or not math.isfinite(dt) or dt <= 0):
            raise ValueError("Malformed completed force interval")
        start = end - dt
        if previous_end is not None and not math.isclose(start, previous_end, abs_tol=1e-7, rel_tol=0):
            raise ValueError("Force intervals have a gap or overlap")
        previous_end = end
        if running is None:
            running = np.zeros(force.shape[:2])
            force_peak, impulse_peak = running.copy(), running.copy()
            seconds = np.zeros(len(force))
            first, last = np.full(len(force), np.nan), np.full(len(force), np.nan)
        elif running.shape != force.shape[:2]:
            raise ValueError("Force lane/body shape changed")
        normal = np.maximum(force[..., 2], 0.) * enabled[:, None]
        history.append([start, end, normal])
        running += normal * dt
        cutoff = end - window_s
        while history and history[0][1] <= cutoff + 1e-12:
            left, right, old = history.popleft()
            running -= old * (right - left)
        if history and history[0][0] < cutoff:
            left, right, old = history[0]
            running -= old * (cutoff - left)
            history[0][0] = cutoff
        force_peak = np.maximum(force_peak, normal)
        impulse_peak = np.maximum(impulse_peak, np.maximum(running, 0.))
        first = np.where(enabled & np.isnan(first), start, first)
        last = np.where(enabled, end, last)
        seconds += dt * enabled
        frames_seen += 1
    if running is None:
        raise ValueError("No physical frames were committed")
    groups = np.asarray(body_groups(body_names))
    group_force, group_impulse = [], []
    for group in range(len(GROUPS)):
        mask = groups == group
        group_force.append(force_peak[:, mask].max(1) if mask.any() else np.zeros(len(running)))
        group_impulse.append(impulse_peak[:, mask].max(1) if mask.any() else np.zeros(len(running)))
    return dict(group_order=list(GROUPS), frames=frames_seen,
        body_force_peak_n=force_peak.tolist(), body_impulse_peak_ns=impulse_peak.tolist(),
        group_force_peak_n=np.stack(group_force, 1).tolist(),
        group_impulse_peak_ns=np.stack(group_impulse, 1).tolist(),
        enabled_scored_seconds=seconds.tolist(),
        first_enabled_interval_start_s=[None if np.isnan(x) else float(x) for x in first],
        last_enabled_interval_end_s=[None if np.isnan(x) else float(x) for x in last])


def diagnostic_fields(diagnostics, lanes, prefix="diagnostics"):
    """Keep original tensor diagnostics, including named simultaneous faults."""
    fields = {}
    for name, value in sorted(diagnostics.items()):
        key = prefix + "/" + name
        if isinstance(value, dict):
            fields.update(diagnostic_fields(value, lanes, key))
        elif hasattr(value, "detach") and value.ndim and value.shape[0] == lanes:
            fields[key] = value
    return fields


class RecoveryTraceObserver:
    """Read-only hook for the original fixed-grid cold-evaluation lifecycle."""
    def __init__(self, env, output, *, seconds):
        import torch
        from .handoff_evaluation import CompletedPhysicsClock
        if not env.identified_wheels.enabled or env.getup_feedback is None:
            raise ValueError("Raw recovery traces require the identified fixed-grid drive and feedback")
        if bool(env.domain.enabled.any()):
            raise ValueError("Raw recovery trace native admission is restricted to the nominal domain")
        if hasattr(env.identified_wheels, "samples") or not hasattr(env.identified_wheels, "history"):
            raise ValueError("This cold trace admits the original fixed-grid wheel response only")
        self.env, self.torch, self.output = env, torch, Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.policy_capacity = round(seconds / env.policy_dt)
        self.physical_capacity = self.policy_capacity * env.decimation
        self.physical = self.policy = None
        self.last_native_step = env.sim.get_physics_step_count()
        self.manager = env.sim.physics_manager.step.__self__
        self.original_native = self.manager._physx_sim
        self.native_clock = CompletedPhysicsClock(self.original_native)
        self.manager._physx_sim = self.native_clock
        self.last_native_completed = 0
        self.policy_index = -1
        self.contact = torch.zeros_like(env.contact_force)
        self.scored = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.last_motor = torch.zeros(env.num_envs, 6, device=env.device)
        self.last_held = torch.zeros_like(self.last_motor)
        self.last_legs = torch.zeros(env.num_envs, 4, device=env.device)
        self.last_wheels = torch.zeros(env.num_envs, 2, device=env.device)
        self.target_valid = False
        self.identities = self._identities()
        self.original_pd = env.v5.motor_efforts
        self.original_wheel = env.identified_wheels.apply_torque
        self.original_updated = env.control_audit.physics_updated

        def pd(q, dq, legs, wheels):
            self.last_legs.copy_(legs)
            self.last_wheels.copy_(wheels)
            result = self.original_pd(q, dq, legs, wheels)
            self.last_held.copy_(result)
            self.target_valid = True
            return result

        def wheel(effort, velocity):
            result = self.original_wheel(effort, velocity)
            # The caller subsequently modifies its returned buffer in place.
            # Archive the motor output separately from passive physical forces.
            self.last_motor.copy_(result)
            return result

        def updated(dt):
            result = self.original_updated(dt)
            self.observe_physics(dt)
            return result

        env.v5.motor_efforts, env.identified_wheels.apply_torque = pd, wheel
        env.control_audit.physics_updated = updated
        self.manifest = dict(schema=SCHEMA, status="prepared", physical_rate=True,
            physics_dt_s=env.dt, policy_dt_s=env.policy_dt, feedback_dt_s=env.getup_feedback.dt,
            body_names=list(env.robot.body_names), joint_names=list(env.robot.joint_names),
            motor_joint_names=[env.robot.joint_names[i] for i in env.ids],
            action_coordinates=(env.encoder_increment_actions.config["policy_order"]
                if getattr(env, "encoder_increment_actions", None) is not None else
                ["left_hip", "left_auxiliary", "right_hip", "right_auxiliary", "left_wheel", "right_wheel"]),
            contact_scope="ground resultants per body; not self collision or internal structural loads",
            clock_scope="native simulate submission followed by completed fetch, not an engine time getter",
            controller_history_scope="listed complete fixed-grid buffers at every completed step; not a simulator checkpoint",
            controller_object_ids=self.identities,
            controller_configuration=dict(actuator_response=getattr(env.actuator_response, "cfg", None),
                fixed_grid_wheel=getattr(env.identified_wheels, "cfg", None),
                feedback=getattr(env.getup_feedback, "cfg", None), domain=getattr(env.domain, "cfg", None)),
            scoring_changes=False, original_terminal_retirement_resets_preserved=True,
            wheelbipe16001_handoff_tested=False, batches=[])
        write_json(self.output / "manifest.json", self.manifest)

    def _identities(self):
        e = self.env
        return {name: id(value) for name, value in dict(env=e, sim=e.sim, robot=e.robot, controller=e.v5,
            actuator_response=e.actuator_response, identified_wheels=e.identified_wheels,
            domain=e.domain, feedback=e.getup_feedback, gimbal=e.gimbal).items()}

    def _fields(self, *, contacts=True):
        e, torch, count = self.env, self.torch, self.env.num_envs
        d = e.robot.data
        fields = {"state/joint_position": d.joint_pos.torch, "state/joint_velocity": d.joint_vel.torch,
            "state/root_pose": d.root_link_pose_w.torch, "state/root_com_velocity": d.root_com_vel_w.torch,
            "state/gravity_body": d.projected_gravity_b.torch, "state/gyro_body": d.root_com_ang_vel_b.torch,
            "mask/enabled": ~(e.passive | e.finished), "mask/scored": self.scored,
            "mask/passive": e.passive, "mask/finished": e.finished,
            "control/raw_action": e.actions, "control/previous_raw_action": e.previous_actions}
        if contacts:
            fields.update({"contact/force_world_n": self.contact,
                "effort/physical_nm": d.applied_torque.torch,
                "effort/motor_nm": self.last_motor, "effort/held_pre_response_nm": self.last_held,
                "target/legs_rad": self.last_legs, "target/wheels_rad_s": self.last_wheels,
                "target/valid": torch.full((count,), self.target_valid, device=e.device),
                "effort/requested_nm": e.v5.requested_motor_effort,
                "effort/bounds_nm": e.v5.current_motor_bounds})
        specifications = {
            "actuator_response": ("state",),
            "identified_wheels": ("history", "passive_effort", "drive_bound", "head"),
            "getup_feedback": ("q", "dq", "q_history", "dq_history", "bias", "period", "delay",
                               "delay_fraction", "cursor", "tick"),
            "domain": ("enabled", "torque_history", "torque_state", "torque_cursor", "sensor_history",
                       "observed", "sensor_cursor", "last_sensor_tick", "strength", "kp", "kd", "spring",
                       "motor_alpha", "torque_delay", "torque_delay_fraction", "sensor_delay", "bias", "sigma"),
            "gimbal": ("time", "yaw_goal", "yaw_velocity", "phase", "period", "target", "effort"),
            "quiet_standing": ("active", "anchor", "target_slow", "velocity_slow"),
            "leg_friction": ("velocity", "passive_effort")}
        for component, names in specifications.items():
            obj = getattr(e, component, None)
            for name in names:
                value = getattr(obj, name, None)
                if value is None:
                    continue
                key = "history/" + component + "/" + name
                if hasattr(value, "detach"):
                    if name in ("history", "torque_history", "sensor_history", "q_history", "dq_history"):
                        value = value.movedim(1, 0)
                    if not value.ndim or value.shape[0] != count:
                        raise ValueError("Unexpected controller history shape: " + key)
                    fields[key] = value
                elif type(value) in (bool, int, float):
                    fields[key] = torch.full((count,), value, device=e.device)
        return fields

    def _cpu(self, fields):
        # One contiguous transfer per physical frame, including controller buffers.
        shapes = {k: tuple(v.shape) for k, v in fields.items()}
        packed = self.torch.cat([v.reshape(self.env.num_envs, -1).float() for v in fields.values()], 1)
        packed = packed.detach().cpu().numpy()
        result, offset = {}, 0
        for name, shape in shapes.items():
            width = math.prod(shape[1:])
            result[name] = packed[:, offset:offset + width].reshape(shape)
            offset += width
        return result

    def begin_batch(self, offset, cases, lane_cases):
        if self.physical is not None or self.policy is not None:
            raise RuntimeError("Previous raw trace batch was not closed")
        self.batch_path = self.output / f"batch_{offset:03d}"
        self.batch_path.mkdir(exist_ok=False)
        self.scored.zero_()
        self.scored[:len(cases)] = True
        self.target_valid = False
        self.batch_start = self.native_clock.time_s
        self.policy_index = -1
        initial = self._cpu(self._fields(contacts=False))
        np.savez_compressed(self.batch_path / "initial_state.npz", **initial)
        write_json(self.batch_path / "cases.json", dict(scored_cases=cases, lane_cases=lane_cases,
            duplicate_padding_lanes=list(range(len(cases), self.env.num_envs)),
            initial_state_has_measured_contacts=False, native_time_at_start_s=self.batch_start))
        self.manifest["batches"].append(str(self.batch_path.relative_to(self.output)))
        self.manifest["status"] = "recording"
        write_json(self.output / "manifest.json", self.manifest)
        self.terminal_flags = [set() for _ in cases]
        self.original_done = [False for _ in cases]

    def begin_policy_step(self, observation, action, active, policy_index):
        if policy_index != self.policy_index + 1:
            raise RuntimeError("Policy trace skipped or repeated an actor interval")
        self.policy_index = policy_index
        self.scored.zero_()
        self.scored[:len(active)] = self.torch.as_tensor(active, device=self.env.device)
        self.actor_fields = self._cpu({"actor/input_35": observation["policy"],
            "actor/output_6": action, "actor/previous_control_action": self.env.actions,
            "mask/scored": self.scored, "mask/passive_before": self.env.passive,
            "mask/finished_before": self.env.finished})

    def observe_physics(self, dt):
        import warp as wp
        e = self.env
        native_step = e.sim.get_physics_step_count()
        if (native_step != self.last_native_step + 1 or self.native_clock.steps != self.last_native_completed + 1
                or self.native_clock.last_dt != dt or e.sim.get_physics_dt() != dt):
            raise RuntimeError("Raw trace hook does not follow exactly one completed native step")
        self.last_native_step, self.last_native_completed = native_step, self.native_clock.steps
        if self.policy_index < 0:
            raise RuntimeError("Physical trace has no owning actor interval")
        for ids, view in e.contact_views:
            matrix = wp.to_torch(view.get_contact_force_matrix(dt=dt))
            self.contact[ids] = matrix.reshape(len(ids), e.body_count, -1, 3).sum(2)
        fields = self._cpu(self._fields())
        if self.physical is None:
            self.physical = ArrayTrace(self.batch_path / "physical", self.physical_capacity, e.num_envs, fields)
        self.physical.append(fields, self._clock(dt))

    def _clock(self, dt):
        return [self.native_clock.time_s - self.batch_start, dt, self.last_native_step,
                self.native_clock.time_s, self.native_clock.last_dt or 0., self.policy_index]

    def end_policy_step(self, diagnostics, done, time_outs):
        fields = dict(self.actor_fields)
        fields.update(self._cpu(diagnostic_fields(dict(diagnostics, original_done=done, time_outs=time_outs),
                                                   self.env.num_envs)))
        if self.policy is None:
            self.policy = ArrayTrace(self.batch_path / "policy", self.policy_capacity, self.env.num_envs, fields)
        self.policy.append(fields, self._clock(self.env.policy_dt))
        for lane in range(len(self.terminal_flags)):
            if not fields["mask/scored"][lane] or not bool(fields["diagnostics/original_done"][lane]):
                continue
            self.original_done[lane] = True
            for name, value in fields.items():
                flag = (name.startswith("diagnostics/task_failures/") or name in {
                    "diagnostics/mechanical", "diagnostics/closure_gap", "diagnostics/impact_bad",
                    "diagnostics/boundary_bad", "diagnostics/terminated", "diagnostics/time_outs"})
                if flag and np.asarray(value[lane]).ndim == 0 and bool(value[lane]):
                    self.terminal_flags[lane].add(name)
        self.physical.flush()
        self.policy.flush()

    def end_batch(self, rows):
        if self.physical is None or self.policy is None:
            raise RuntimeError("Cannot finish an empty raw evaluation")
        physical = self.physical
        def frames():
            for i in range(physical.count):
                values = physical.values[i]
                yield (physical.clock[i, 0], physical.clock[i, 1],
                    field_values(values, physical.layout, "contact/force_world_n"),
                    (field_values(values, physical.layout, "mask/enabled") > 0) &
                    (field_values(values, physical.layout, "mask/scored") > 0))
        audit = contact_integrals(frames(), self.manifest["body_names"],
                                  window_s=self.env.recovery_impact.config["window_seconds"])
        identity_preserved = self.identities == self._identities()
        audit.update(schema=SCHEMA, status="completed", same_controller_objects=identity_preserved,
            original_terminal_retirement_resets_preserved=True, physical_frames=physical.count,
            policy_frames=self.policy.count, policy_to_physics_frame_count_matches=(
                physical.count == self.policy.count * self.env.decimation), cases=[])
        for lane, row in enumerate(rows):
            raw_force = np.asarray(audit["group_force_peak_n"][lane])
            raw_impulse = np.asarray(audit["group_impulse_peak_ns"][lane])
            force_difference = np.abs(raw_force - row["force_peak_n"])
            impulse_difference = np.abs(raw_impulse - row["impulse_peak_ns"])
            receipt = dict(lane=lane, force_peak_n=raw_force.tolist(), impulse_peak_ns=raw_impulse.tolist(),
                original_force_max_abs_difference_n=float(force_difference.max()),
                original_impulse_max_abs_difference_ns=float(impulse_difference.max()),
                agrees_with_original_force=bool(np.allclose(raw_force, row["force_peak_n"], atol=2e-4, rtol=2e-4)),
                agrees_with_original_impulse=bool(np.allclose(raw_impulse, row["impulse_peak_ns"], atol=2e-4, rtol=2e-4)),
                enabled_scored_seconds=audit["enabled_scored_seconds"][lane],
                terminal_flags=sorted(self.terminal_flags[lane]),
                screen_stop=("original_done" if self.original_done[lane] else "passed" if row["passed"]
                             else "evaluation_window_exhausted"),
                actual_wheelbipe16001_handoff_time_s=None,
                terminal_flags_are_simultaneous_observations_not_unique_causal_attribution=True)
            row["raw_physics_audit"] = receipt
            audit["cases"].append(receipt)
        write_json(self.batch_path / "audit.json", audit)
        self.physical.close()
        self.policy.close()
        self.physical = self.policy = None
        if not identity_preserved or not audit["policy_to_physics_frame_count_matches"]:
            raise RuntimeError("Raw trace object identity or frame count differs from its contract")
        return audit

    def close(self, status):
        for trace in (self.physical, self.policy):
            if trace is not None:
                trace.close("failed")
        self.env.v5.motor_efforts = self.original_pd
        self.env.identified_wheels.apply_torque = self.original_wheel
        self.env.control_audit.physics_updated = self.original_updated
        self.manager._physx_sim = self.original_native
        self.manifest.update(status=status, same_controller_objects=self.identities == self._identities())
        write_json(self.output / "manifest.json", self.manifest)
