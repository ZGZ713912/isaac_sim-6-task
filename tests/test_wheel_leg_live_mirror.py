"""Geometry, ROS CDR synchronization, recording and read-only transport checks."""
from dataclasses import replace
import asyncio
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import threading
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wheeled_algo.wheel_leg_mirror import MirrorModel, ROOT, load_telemetry_reader


@pytest.fixture(scope="module")
def protocol():
    return load_telemetry_reader(ROOT.parent / "RMCS")


@pytest.fixture
def model():
    return MirrorModel()


@pytest.fixture(scope="module")
def geometry():
    spec = importlib.util.spec_from_file_location("mirror_reference_geometry", ROOT / "tools/full_robot_v202.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(model, protocol, q=None, rotation=None, stamp=1_000_000_001):
    q = dict(model.spec["nominal_joint_pos"] if q is None else q)
    rotation = Rotation.identity() if rotation is None else rotation
    xyzw = rotation.as_quat()
    frames = model.fk(q)
    knees, lengths = [], []
    for side in model.mapping["legs"]:
        h, k, w = (frames[n][:3, 3] for n in side["points"])
        a, b = h - k, w - k
        knees.append(math.acos(float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))))
        lengths.append(float(np.linalg.norm(w - h)))
    by_source = {e["source"]: (q[e["target"]] - e["offset_rad"]) / e["sign"] for e in model.entries}
    model_by_source = {e["source"]: q[e["target"]] for e in model.entries}
    offsets = {e["source"]: e["offset_rad"] for e in model.entries}
    return protocol.Snapshot(stamp, tuple(by_source[n] for n in protocol.MOTOR_NAMES),
        tuple(xyzw[[3, 0, 1, 2]]), (.1, -.2, .3), tuple(rotation.inv().apply([0., 0., -1.])), (.1, -.2, .3),
        velocities=(.01, .02, .03, .04, .05, .06), rmcs_knee=tuple(knees), rmcs_length=tuple(lengths),
        rmcs_model_positions=tuple(model_by_source[n] for n in protocol.MOTOR_NAMES),
        model_offsets=tuple(offsets[n] for n in protocol.MOTOR_NAMES[:4]),
        sensor_age_s=(.002,) * 7, diagnostic_status="valid", **model.mapping["calibration"])


def ros_messages(snapshot, protocol, *, order=(0, 1, 2, 3, 4, 5)):
    from rosbags.typesys import Stores, get_typestore
    store = get_typestore(Stores.ROS2_JAZZY)
    t = store.types
    header = t["std_msgs/msg/Header"](t["builtin_interfaces/msg/Time"](
        snapshot.stamp_ns // 1_000_000_000, snapshot.stamp_ns % 1_000_000_000), "chassis_body")
    vector = lambda v: t["geometry_msgs/msg/Vector3"](*v)
    w, x, y, z = snapshot.quaternion_wxyz
    joints = t["sensor_msgs/msg/JointState"](header, [protocol.MOTOR_NAMES[i] for i in order],
        np.asarray([snapshot.positions[i] for i in order]), np.asarray([snapshot.velocities[i] for i in order]),
        np.array([], dtype=float))
    imu = t["sensor_msgs/msg/Imu"](header, t["geometry_msgs/msg/Quaternion"](x, y, z, w),
        np.zeros(9), vector(snapshot.gyro_body), np.zeros(9), vector((0., 0., 0.)), np.array([-1.] + [0.] * 8))
    values = {"schema": "wheel_leg_telemetry_v1", "coordinate": "rmcs_hardware_output_shaft_v1",
        "feedback_fresh": str(snapshot.feedback_fresh).lower(), "state_valid": str(snapshot.state_valid).lower(),
        "reset_count": str(snapshot.reset_count), "imu_age_s": str(snapshot.sensor_age_s[-1]),
        "control_enabled": "unknown" if snapshot.control_enabled is None else str(snapshot.control_enabled).lower(),
        "dm_control_ready": "unknown" if snapshot.dm_control_ready is None else str(snapshot.dm_control_ready).lower(),
        "calibration_sha256": snapshot.calibration_sha256,
        "calibration_model_spec_sha256": snapshot.calibration_model_spec_sha256}
    for i, name in enumerate(protocol.MOTOR_NAMES):
        values["motor_age_s/" + name] = str(snapshot.sensor_age_s[i])
        values["model_position_rad/" + name] = str(snapshot.rmcs_model_positions[i])
        if i < 4:
            values["model_offset_rad/" + name] = str(snapshot.model_offsets[i])
    for i, side in enumerate(("left", "right")):
        values[side + "/passive_knee_rad"] = str(snapshot.rmcs_knee[i])
        values[side + "/leg_length_m"] = str(snapshot.rmcs_length[i])
    status = t["diagnostic_msgs/msg/DiagnosticStatus"](0, "wheel_leg/telemetry", "valid", "wheel_leg",
        [t["diagnostic_msgs/msg/KeyValue"](k, v) for k, v in values.items()])
    return store, dict(zip(protocol.TOPICS, [joints, imu,
        t["geometry_msgs/msg/Vector3Stamped"](header, vector(snapshot.gravity_rl)),
        t["geometry_msgs/msg/Vector3Stamped"](header, vector(snapshot.gyro_rl)),
        t["diagnostic_msgs/msg/DiagnosticArray"](header, [status])]))


@pytest.mark.parametrize("changes,angles", [
    ({}, [0., 0., 0.]),
    ({"L_joint1": -.12, "LL_joint1": .42}, [.15, -.25, .7]),
    ({"R_joint1": .75, "RR_joint1": -.2, "L_joint3": 1.7, "R_joint3": -2.}, [-.4, .2, -1.2]),
    ({"L_joint1": -1.2, "LL_joint1": -.3, "R_joint1": -.9, "RR_joint1": -1.7}, [.6, -.5, 1.4]),
])
def test_complete_geometry_matches_existing_independent_asset_solver(model, protocol, geometry, changes, angles):
    active = {n: model.spec["nominal_joint_pos"][n] for n in model.targets}
    q = geometry.solve_pose(model.spec, {**active, **changes, "M_joint1": 0., "M_joint2": 0.},
                            model.spec["nominal_joint_pos"])
    snapshot = sample(model, protocol, q, Rotation.from_euler("xyz", angles))
    result = model.reconstruct(snapshot)
    root = np.eye(4)
    root[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    root[:3, 3] = (0., 0., .65)
    expected = geometry.fk(model.spec, q, root)
    for name, frame in expected.items():
        np.testing.assert_allclose(result["body_frames"][name], frame, atol=2e-5, rtol=0.)
    assert result["closure_error_m"] < 2e-6
    np.testing.assert_allclose(result["coordinate_error_rad"], 0., atol=1e-12)
    assert result["gravity_error"] < 1e-12


def test_independent_encoder_turns_do_not_change_pose(model, protocol):
    original = sample(model, protocol)
    expected = model.reconstruct(original)
    turns = np.array([3, -2, -4, 7, 1000, -1000]) * (2 * math.pi)
    wrapped = replace(original, positions=tuple(np.asarray(original.positions) + turns))
    actual = model.reconstruct(wrapped)
    for name in expected["body_frames"]:
        np.testing.assert_allclose(actual["body_frames"][name], expected["body_frames"][name], atol=1e-9)


def test_measured_leg_motion_across_pi_remains_on_the_same_assembly(model, protocol, geometry):
    active = {n: model.spec["nominal_joint_pos"][n] for n in model.targets}
    active.update(M_joint1=0., M_joint2=0.)
    for delta in (3.50, 3.55, 3.60, 3.65):
        changed = dict(active)
        for name in ("L_joint1", "LL_joint1"):
            changed[name] += delta
        q = geometry.solve_pose(model.spec, changed, model.spec["nominal_joint_pos"])
        state = model.reconstruct(sample(model, protocol, q))
        expected = geometry.fk(model.spec, q)
        for name, pose in expected.items():
            np.testing.assert_allclose(state["body_frames"][name][:3, :3], pose[:3, :3], atol=2e-5)


@pytest.mark.parametrize("endpoints", [(0, 1), (1, 0)])
def test_reachable_knee_limits_do_not_trigger_a_false_closure_failure(model, protocol, geometry, endpoints):
    active = {n: model.spec["nominal_joint_pos"][n] for n in model.targets}
    active.update(M_joint1=0., M_joint2=0.)
    # First visit the opposite limits: a large valid move exercises warm starts
    # as well as the scaled-gradient stopping condition at a hard joint bound.
    for indices in (endpoints[::-1], endpoints):
        for side, index in zip(model.mapping["legs"], indices):
            active[side["auxiliary"]] = (active[side["hip"]] + side["relative_sign"]
                                         * side["relative_limits_rad"][index])
        q = geometry.solve_pose(model.spec, active, model.spec["nominal_joint_pos"])
        state = model.reconstruct(sample(model, protocol, q))
        assert state["closure_error_m"] < 2e-7
        np.testing.assert_allclose(state["knee_error_rad"], 0., atol=5e-4)


def test_mapping_order_is_not_the_hardware_or_model_array_order(model, protocol, tmp_path):
    snapshot = sample(model, protocol)
    expected = model.reconstruct(snapshot)
    mapping = model.mapping
    mapping["joints"] = mapping["joints"][::-1]
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(mapping))
    actual = MirrorModel(path).reconstruct(snapshot)
    for name in expected["body_frames"]:
        np.testing.assert_allclose(actual["body_frames"][name], expected["body_frames"][name], atol=1e-9)


@pytest.mark.parametrize("replacement,match", [
    ({"sensor_age_s": (.05,) * 7}, "40 ms"),
    ({"calibration_sha256": "different"}, "calibration"),
    ({"model_offsets": (0.,) * 4}, "offsets"),
    ({"rmcs_knee": ()}, "missing mirror reference"),
])
def test_bad_feedback_does_not_advance_the_last_valid_geometry(model, protocol, replacement, match):
    original = sample(model, protocol)
    model.reconstruct(original)
    previous = dict(model.previous)
    with pytest.raises(ValueError, match=match):
        model.reconstruct(replace(original, **replacement))
    assert model.previous == previous


@pytest.mark.parametrize("enabled", [False, True])
def test_hand_moved_pose_is_independent_of_actuator_enable_and_gravity(model, protocol, geometry, enabled):
    active = {n: model.spec["nominal_joint_pos"][n] for n in model.targets}
    model.reconstruct(sample(model, protocol))
    active.update(L_joint1=-1.2, LL_joint1=-.3, R_joint1=-.9, RR_joint1=-1.7, M_joint1=0., M_joint2=0.)
    q = geometry.solve_pose(model.spec, active, model.spec["nominal_joint_pos"])
    rotation = Rotation.from_euler("xyz", [2.6, .8, -1.7])
    snapshot = replace(sample(model, protocol, q, rotation), control_enabled=enabled,
                       dm_control_ready=False, gyro_body=(20., -30., 40.),
                       gravity_rl=(0., 0., 0.), feedback_fresh=False, state_valid=False)
    actual = model.reconstruct(snapshot, base_position=(0., 0., -.5))
    root = np.eye(4)
    root[:3, :3] = rotation.as_matrix()
    root[:3, 3] = (0., 0., -.5)
    for name, frame in geometry.fk(model.spec, q, root).items():
        np.testing.assert_allclose(actual["body_frames"][name], frame, atol=2e-5)
    assert actual["reference_valid"] is False
    assert actual["knee_error_rad"] is None and actual["gravity_error"] is None
    assert actual["diagnostic_warnings"]
    # Zero gravity from an invalid estimator must survive the CDR path as well.
    store, messages = ros_messages(snapshot, protocol)
    assembler = protocol.SnapshotAssembler()
    for topic, message in messages.items():
        schema = protocol.TOPICS[topic][1]
        decoded = assembler.accept(topic, store.deserialize_cdr(store.serialize_cdr(message, schema), schema))
    assert decoded == snapshot


def test_reference_gravity_disagreement_is_displayed_without_changing_measured_pose(model, protocol):
    snapshot = sample(model, protocol, rotation=Rotation.from_euler("xyz", [math.pi, .2, -.7]))
    snapshot = replace(snapshot, gravity_rl=(0., 0., -4.), control_enabled=False, dm_control_ready=False)
    state = model.reconstruct(snapshot)
    assert state["gravity_error"] > 4.
    np.testing.assert_allclose(state["body_frames"]["base_link"][:3, :3],
                               Rotation.from_euler("xyz", [math.pi, .2, -.7]).as_matrix(), atol=1e-12)


def test_measured_pose_outside_nominal_leg_limits_is_reported_without_clamping(model, protocol, geometry):
    active = {n: model.spec["nominal_joint_pos"][n] for n in model.targets}
    for side in model.mapping["legs"]:
        active[side["auxiliary"]] = active[side["hip"]] + side["relative_sign"] * (side["relative_limits_rad"][1] + .02)
    active.update(M_joint1=0., M_joint2=0.)
    q = geometry.solve_pose(model.spec, active, model.spec["nominal_joint_pos"])
    state = model.reconstruct(sample(model, protocol, q))
    assert any("outside nominal" in warning for warning in state["diagnostic_warnings"])
    assert np.rad2deg(state["knee_rad"]).min() > 110.
    for name, frame in geometry.fk(model.spec, q).items():
        np.testing.assert_allclose(state["body_frames"][name][:3, :3], frame[:3, :3], atol=2e-5)


def test_cdr_roundtrip_name_permutation_and_exact_timestamp_pairing(model, protocol):
    snapshot = sample(model, protocol)
    store, messages = ros_messages(snapshot, protocol, order=(5, 2, 4, 0, 3, 1))
    assembler = protocol.SnapshotAssembler()
    for topic in list(messages)[::-1]:
        schema = protocol.TOPICS[topic][1]
        message = store.deserialize_cdr(store.serialize_cdr(messages[topic], schema), schema)
        result = assembler.accept(topic, message)
    assert result == snapshot
    # A single later IMU must never be paired with the older four topics.
    _, later = ros_messages(replace(snapshot, stamp_ns=snapshot.stamp_ns + 1), protocol)
    mixed = protocol.SnapshotAssembler()
    for topic, message in messages.items():
        assert mixed.accept(topic, later[topic] if "imu_body" in topic else message) is None


def test_wrong_frame_duplicate_names_and_missing_topics_are_rejected(model, protocol):
    _, messages = ros_messages(sample(model, protocol), protocol)
    topic = next(iter(messages))
    messages[topic].header.frame_id = "sensor_frame"
    with pytest.raises(ValueError, match="frame"):
        protocol.SnapshotAssembler().accept(topic, messages[topic])
    _, messages = ros_messages(sample(model, protocol), protocol)
    messages[topic].name[0] = messages[topic].name[1]
    assembler = protocol.SnapshotAssembler()
    with pytest.raises(ValueError, match="duplicate"):
        for name, msg in messages.items():
            assembler.accept(name, msg)


def test_recording_replay_identity_timing_and_no_overwrite(model, protocol, tmp_path):
    path = tmp_path / "session.jsonl"
    latest = protocol.LatestSnapshot(path, metadata=model.metadata)
    snapshot = sample(model, protocol)
    latest.put(snapshot)
    latest.close()
    rows = list(protocol.read_record(path, model.metadata))
    assert rows[0][1] == snapshot
    with pytest.raises(FileExistsError):
        protocol.LatestSnapshot(path)
    with pytest.raises(ValueError, match="mapping_sha256"):
        list(protocol.read_record(path, {"mapping_sha256": "other"}))
    target = protocol.LatestSnapshot()
    replay = protocol.ReplayReader(path, target, metadata=model.metadata)
    replay.start()
    replay.join(2)
    assert target.snapshot == snapshot and target.status == "replay complete"
    target.received -= 1
    assert target.get()[1] > .2  # GUI can distinguish an old state from live reception.


def test_empty_and_negative_time_recordings_are_rejected(model, protocol, tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        list(protocol.read_record(path))
    path = tmp_path / "negative.jsonl"
    latest = protocol.LatestSnapshot(path, metadata=model.metadata)
    latest.put(sample(model, protocol))
    latest.close()
    lines = path.read_text().splitlines()
    row = json.loads(lines[1])
    row["record_time_s"] = -1.
    path.write_text(lines[0] + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="nonnegative"):
        list(protocol.read_record(path, model.metadata))


def test_foxglove_cdr_reader_reconnects_and_only_subscribes(model, protocol):
    import websockets
    ready = threading.Event()
    stopped = threading.Event()
    received, connections, failures = [], [], []
    endpoint = []
    first = sample(model, protocol)
    second = replace(first, stamp_ns=first.stamp_ns + 1)
    async def handler(socket):
        index = len(connections)
        connections.append(index)
        store, messages = ros_messages(first if index == 0 else second, protocol)
        base = 10 if index == 0 else 100
        channels = [{"id": base + i, "topic": topic, "encoding": "cdr", "schemaName": protocol.TOPICS[topic][1]}
                    for i, topic in enumerate(messages)]
        await socket.send(json.dumps({"op": "advertise", "channels": channels}))
        request = json.loads(await socket.recv())
        received.append(request)
        for i, (topic, message) in enumerate(messages.items()):
            payload = store.serialize_cdr(message, protocol.TOPICS[topic][1])
            await socket.send(bytes([1]) + struct.pack("<IQ", base + i, first.stamp_ns) + bytes(payload))
        if index == 0:
            await asyncio.sleep(.05)
            await socket.close()
        else:
            while not stopped.is_set():
                await asyncio.sleep(.01)
    async def serve():
        async with websockets.serve(handler, "127.0.0.1", 0, subprotocols=["foxglove.websocket.v1"] ) as server:
            endpoint.append(server.sockets[0].getsockname()[1])
            ready.set()
            while not stopped.is_set():
                await asyncio.sleep(.01)
    def run():
        try:
            asyncio.run(serve())
        except Exception as error:
            failures.append(error)
            ready.set()
    server = threading.Thread(target=run, daemon=True)
    server.start()
    assert ready.wait(3) and not failures
    latest = protocol.LatestSnapshot()
    reader = protocol.FoxgloveReader(f"ws://127.0.0.1:{endpoint[0]}", latest)
    reader.start()
    try:
        deadline = time.monotonic() + 6
        while latest.count < 2 and time.monotonic() < deadline:
            time.sleep(.02)
        assert latest.count >= 2, latest.status
        assert latest.snapshot.stamp_ns == second.stamp_ns
        assert len(connections) >= 2
        assert all(m["op"] == "subscribe" and len(m["subscriptions"]) == 5 for m in received)
    finally:
        stopped.set()
        reader.close()
        server.join(3)
    assert not failures
