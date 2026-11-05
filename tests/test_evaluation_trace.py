import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from wheeled_tasks.chassis.evaluation_trace import (
    ArrayTrace, RecoveryTraceObserver, contact_integrals, diagnostic_fields, field_values,
)


def test_per_body_integral_precedes_group_maximum():
    force = np.zeros((1, 3, 3))
    frames = []
    for i in range(20):
        force = force.copy()
        force[0, :, 2] = [0., 100. if i < 10 else 0., 100. if i >= 10 else 0.]
        frames.append(((i + 1) * .001, .001, force, [True]))
    audit = contact_integrals(frames, ["base_link", "leg_a", "leg_b"])
    assert audit["body_impulse_peak_ns"][0] == pytest.approx([0., 1., 1.])
    # Taking the group's summed force first would incorrectly produce 2 N.s.
    assert audit["group_impulse_peak_ns"][0][2] == pytest.approx(1.)


def test_fractional_interval_is_removed_with_its_own_dt():
    forces = []
    end = 0.
    for dt, value in ((.007, 100.), (.013, 100.), (.005, 100.)):
        end += dt
        forces.append((end, dt, np.array([[[0., 0., value]]]), [True]))
    audit = contact_integrals(forces, ["base_link"])
    assert audit["body_impulse_peak_ns"] == [[pytest.approx(2.)]]
    assert audit["enabled_scored_seconds"] == [pytest.approx(.025)]
    assert audit["first_enabled_interval_start_s"] == [pytest.approx(0.)]


def test_disabled_retired_and_padding_intervals_do_not_add_impulse():
    frames = []
    for i in range(6):
        # The third lane simulates duplicated padding with much larger contacts.
        force = np.array([[[0., 0., 100.]], [[0., 0., -100.]], [[0., 0., 10000.]]])
        frames.append(((i + 1) * .005, .005, force, [i in (1, 2), True, False]))
    audit = contact_integrals(frames, ["base_link"])
    assert np.asarray(audit["body_impulse_peak_ns"]).flatten() == pytest.approx([1., 0., 0.])
    assert audit["enabled_scored_seconds"] == pytest.approx([.01, .03, 0.])
    assert audit["first_enabled_interval_start_s"][2] is None
    assert audit["last_enabled_interval_end_s"][0] == pytest.approx(.015)


@pytest.mark.parametrize("frames", [[], [(1., 0., np.zeros((1, 1, 3)), [True])],
    [(.01, .01, np.full((1, 1, 3), np.nan), [True])],
    [(.01, .01, np.zeros((1, 1, 3)), [True]), (.03, .01, np.zeros((1, 1, 3)), [True])]])
def test_invalid_physical_records_fail(frames):
    with pytest.raises(ValueError):
        contact_integrals(frames, ["base_link"])


def test_array_layout_committed_prefix_and_nan_diagnostics(tmp_path):
    fields = {"mask": np.array([True, False]), "history": np.arange(12).reshape(2, 2, 3),
              "diagnostic": np.array([np.nan, 0.])}
    trace = ArrayTrace(tmp_path / "trace", 3, 2, fields, reserve_bytes=0)
    trace.append(fields, [0.005, .005, 9007199, .005, .005, 0])
    assert json.loads((trace.path / "manifest.json").read_text())["valid_frames"] == 0
    trace.flush()
    assert trace.manifest["valid_frames"] == 1
    assert trace.clock[0, 2] == 9007199
    assert np.array_equal(field_values(trace.values[0], trace.layout, "history"), fields["history"])
    assert np.isnan(field_values(trace.values[0], trace.layout, "diagnostic")[0])
    with pytest.raises(ValueError, match="shape"):
        trace.append(dict(fields, history=np.zeros((2, 7))), [0.] * 6)
    trace.close("failed")
    manifest = json.loads((trace.path / "manifest.json").read_text())
    assert manifest["valid_frames"] == 1 and manifest["capacity"] == 3
    assert manifest["unused_capacity_is_data"] is False
    assert manifest["status"] == "failed"


def test_array_budget_overrun_is_not_silently_truncated(tmp_path):
    fields = {"a": np.ones((2, 1))}
    trace = ArrayTrace(tmp_path / "trace", 1, 2, fields, reserve_bytes=0)
    trace.append(fields, [0.] * 6)
    with pytest.raises(RuntimeError, match="budget"):
        trace.append(fields, [0.] * 6)
    trace.close()


def test_disk_budget_checked_before_simulation(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.disk_usage", lambda path: NS(free=1))
    with pytest.raises(OSError, match="disk"):
        ArrayTrace(tmp_path / "trace", 5, 2, {"state": np.ones((2, 6))})


def test_original_nested_flags_preserved_without_scalar_broadcasting():
    source = {"task_failures": {"rise_timeout": torch.tensor([True, False]),
                               "refall": torch.tensor([False, True])},
              "recovered_at": torch.tensor([float("nan"), .2]),
              "global_scalar": torch.tensor(3.)}
    fields = diagnostic_fields(source, 2)
    assert set(fields) == {"diagnostics/task_failures/rise_timeout", "diagnostics/task_failures/refall",
                           "diagnostics/recovered_at"}
    assert fields["diagnostics/task_failures/rise_timeout"] is source["task_failures"]["rise_timeout"]


def fake_env(monkeypatch):
    n = 2
    e = NS(num_envs=n, device="cpu", dt=.005, policy_dt=.01, decimation=2, ids=list(range(6)))
    e.passive, e.finished = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
    e.actions, e.previous_actions = torch.zeros(n, 6), torch.zeros(n, 6)
    data = NS(**{name: NS(torch=value) for name, value in {
        "joint_pos": torch.zeros(n, 6), "joint_vel": torch.zeros(n, 6),
        "root_link_pose_w": torch.zeros(n, 7), "root_com_vel_w": torch.zeros(n, 6),
        "projected_gravity_b": torch.zeros(n, 3), "root_com_ang_vel_b": torch.zeros(n, 3),
        "applied_torque": torch.zeros(n, 6)}.items()})
    e.robot = NS(data=data, body_names=["base_link"], joint_names=["q" + str(i) for i in range(6)])
    e.body_count, e.contact_force = 1, torch.full((n, 1, 3), -99.)
    def get_contacts(*, dt):
        assert dt == e.dt
        return torch.tensor([[[0., 0., 100.]], [[0., 0., 500.]]])
    e.contact_views = [(list(range(n)), NS(get_contact_force_matrix=get_contacts))]
    monkeypatch.setitem(sys.modules, "warp", NS(to_torch=lambda x: x))
    e.actuator_response = NS(state=torch.zeros(n, 6))
    e.domain = NS(enabled=torch.zeros(n, dtype=torch.bool), torque_history=torch.zeros(3, n, 6),
                  torque_state=torch.zeros(n, 6), sensor_history=torch.zeros(3, n, 18),
                  torque_cursor=0, sensor_cursor=0)
    e.getup_feedback = NS(q=torch.zeros(n, 6), dq=torch.zeros(n, 6), q_history=torch.zeros(8, n, 6),
        dq_history=torch.zeros(8, n, 6), cursor=0, tick=0, dt=.005)
    e.gimbal = None
    e.recovery_impact = NS(config={"window_seconds": .02})
    e.encoder_increment_actions = NS(config={"policy_order": ["cL", "rL", "cR", "rR", "wL", "wR"]})
    e.v5 = NS(requested_motor_effort=torch.zeros(n, 6), current_motor_bounds=torch.full((n, 6), 40.))
    def pd(q, dq, legs, wheels):
        e.getup_feedback.tick += 1
        e.getup_feedback.q.copy_(q)
        return torch.full_like(q, 3.)
    e.v5.motor_efforts = pd
    e.identified_wheels = NS(enabled=True, history=torch.zeros(3, n, 2), head=0,
                            drive_bound=torch.ones(n, 2), passive_effort=torch.zeros(n, 2))
    def wheel(effort, velocity):
        e.identified_wheels.history[e.identified_wheels.head] = effort[:, [2, 5]]
        e.identified_wheels.head = (e.identified_wheels.head + 1) % 3
        return effort.clone()
    e.identified_wheels.apply_torque = wheel
    e.original_updates = 0
    def updated(dt):
        e.original_updates += 1
        return "original_result"
    e.control_audit = NS(physics_updated=updated)
    class Native:
        def simulate(self, dt):
            return "simulate"
        def fetch_results(self):
            return "fetch"
    class Manager:
        def __init__(self):
            self._physx_sim = Native()
        def step(self):
            assert self._physx_sim.simulate(e.dt) == "simulate"
            assert self._physx_sim.fetch_results() == "fetch"
            e.sim.steps += 1
    manager = Manager()
    e.sim = NS(physics_manager=NS(step=manager.step), steps=0,
               get_physics_step_count=lambda: e.sim.steps, get_physics_dt=lambda: e.dt)
    return e, manager


def test_observer_captures_copies_preserves_callbacks_and_original_scoring(tmp_path, monkeypatch):
    e, manager = fake_env(monkeypatch)
    pd, wheel, updated, native = e.v5.motor_efforts, e.identified_wheels.apply_torque, e.control_audit.physics_updated, manager._physx_sim
    rng = torch.random.get_rng_state().clone()
    observer = RecoveryTraceObserver(e, tmp_path / "raw", seconds=.02)
    observer.begin_batch(0, [{"family": "belly_down"}], [{"family": "belly_down"}] * 2)
    observation = {"policy": torch.arange(70).reshape(2, 35).float()}
    action = torch.ones(2, 6)
    observer.begin_policy_step(observation, action, [True], 0)
    for _ in range(2):
        effort = e.v5.motor_efforts(torch.ones(2, 6), torch.zeros(2, 6), torch.ones(2, 4), torch.zeros(2, 2))
        motor = e.identified_wheels.apply_torque(effort, torch.zeros(2, 6))
        motor.zero_()  # Caller mutation must not erase the archived motor output.
        manager.step()
        assert e.control_audit.physics_updated(e.dt) == "original_result"
    diagnostics = {"task_failures": {"rise_timeout": torch.tensor([True, False]),
                                     "refall": torch.tensor([True, False])},
                   "terminated": torch.tensor([True, False])}
    observer.end_policy_step(diagnostics, torch.tensor([True, False]), torch.tensor([False, False]))
    rows = [dict(passed=False, force_peak_n=[100., 0., 0., 0.], impulse_peak_ns=[1., 0., 0., 0.])]
    audit = observer.end_batch(rows)
    assert audit["same_controller_objects"] and audit["policy_to_physics_frame_count_matches"]
    assert audit["cases"][0]["agrees_with_original_impulse"]
    assert rows[0]["passed"] is False
    assert rows[0]["raw_physics_audit"]["actual_wheelbipe16001_handoff_time_s"] is None
    assert rows[0]["raw_physics_audit"]["terminal_flags"] == [
        "diagnostics/task_failures/refall", "diagnostics/task_failures/rise_timeout", "diagnostics/terminated"]
    physical_path = tmp_path / "raw/batch_000/physical"
    manifest = json.loads((physical_path / "manifest.json").read_text())
    values = np.load(physical_path / "values.npy", mmap_mode="r")[:manifest["valid_frames"]]
    assert np.all(field_values(values, manifest["fields"], "effort/motor_nm") == 3.)
    assert np.all(field_values(values, manifest["fields"], "mask/scored")[:, 1] == 0.)
    assert audit["body_impulse_peak_ns"][1] == [0.]
    initial = np.load(tmp_path / "raw/batch_000/initial_state.npz")
    assert not any(k.startswith("contact/") or k.startswith("target/") for k in initial.files)
    assert e.getup_feedback.tick == 2 and e.original_updates == 2
    assert torch.all(e.contact_force == -99.)  # Independent query never overwrites the task buffer.
    assert torch.equal(torch.random.get_rng_state(), rng)
    observer.close("completed_unqualified")
    assert e.v5.motor_efforts is pd and e.identified_wheels.apply_torque is wheel
    assert e.control_audit.physics_updated is updated and manager._physx_sim is native


def test_observer_rejects_uncompleted_or_wrong_native_step(tmp_path, monkeypatch):
    e, _ = fake_env(monkeypatch)
    observer = RecoveryTraceObserver(e, tmp_path / "raw", seconds=.01)
    observer.begin_batch(0, [{"family": "belly_down"}], [{}] * 2)
    observer.begin_policy_step({"policy": torch.zeros(2, 35)}, torch.zeros(2, 6), [True], 0)
    with pytest.raises(RuntimeError, match="completed native"):
        e.control_audit.physics_updated(e.dt)
    observer.close("failed")
    assert json.loads((tmp_path / "raw/manifest.json").read_text())["status"] == "failed"


def test_nominal_only_and_fixed_grid_admission(tmp_path, monkeypatch):
    e, _ = fake_env(monkeypatch)
    e.domain.enabled[0] = True
    with pytest.raises(ValueError, match="nominal"):
        RecoveryTraceObserver(e, tmp_path / "raw", seconds=.01)
    e.domain.enabled.zero_()
    e.identified_wheels.samples = []
    with pytest.raises(ValueError, match="fixed-grid"):
        RecoveryTraceObserver(e, tmp_path / "raw2", seconds=.01)


def test_formal_contract_and_default_screen_keep_original_gate():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "contracts/wheel_leg_recovery_physical_trace_v1.json").read_text())
    assert contract["scoring_changes"] is False
    assert contract["original_terminal_retirement_resets_preserved"] is True
    source = (root / "scripts/evaluate_self_righting.py").read_text()
    assert 'parser.add_argument("--raw-physics", action="store_true"' in source
    assert 'row["active_time_s"] <= 20.' in source
    assert 'row["stable_hold_s"] >= 3. - 1e-5' in source
