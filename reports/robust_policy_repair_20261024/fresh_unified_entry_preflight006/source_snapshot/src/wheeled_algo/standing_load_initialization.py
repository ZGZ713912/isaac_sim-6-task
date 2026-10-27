"""Authenticate a nominal mechanics prior used only for fresh output bias.

No policy, tensor checkpoint, optimizer or teacher is an input. The returned
six values initialize ordinary trainable weights, never a runtime controller.
"""
import hashlib
import json
import math
from pathlib import Path


CONTROL_ORDER = ["L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3"]
POLICY_FROM_CONTROL = (0, 1, 3, 4, 2, 5)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_values(reference, control):
    if (reference.get("schema") != "scratch-standing-nominal-static-load-diagnostic-v1"
            or reference.get("training_modified") is not False
            or reference.get("control_joint_order") != CONTROL_ORDER
            or not math.isclose(reference["height_m"], .305, rel_tol=0., abs_tol=1e-8)):
        raise ValueError("Need the unmodified nominal mid-height static reference")
    kp, scale = control["leg_kp"], control["leg_position_scale"]
    if not math.isfinite(kp) or kp <= 0. or scale != .25:
        raise ValueError("Static initialization requires the authenticated leg PD units")
    torque = reference["static_load"]["static_hold_torque_nm"]
    offset = reference["hypothetical_static_load_offset_rad"]
    geometry = reference["initial_geometry_action_control"]
    biased = reference["hypothetical_biased_action_control"]
    for vector in (torque, offset, geometry, biased):
        if len(vector) != 6 or not all(math.isfinite(value) for value in vector):
            raise ValueError("Static initialization needs finite six-axis values")
    for i in range(6):
        if not math.isclose(offset[i] * kp, torque[i], rel_tol=0., abs_tol=2e-5):
            raise ValueError("Static target offset does not equal holding torque divided by Kp")
        if not math.isclose(biased[i], geometry[i] + offset[i] / scale, rel_tol=0., abs_tol=2e-7):
            raise ValueError("Static target offset has inconsistent action units")
        if i in (2, 5):
            if torque[i] != 0. or offset[i] != 0. or geometry[i] != 0. or biased[i] != 0.:
                raise ValueError("A leg load prior must not initialize wheel propulsion")
        elif abs(torque[i]) >= 40. or abs(biased[i]) >= control["action_clip"]:
            raise ValueError("Static initialization exceeds a native motor or action limit")
    return dict(geometry_policy_raw=[geometry[i] for i in POLICY_FROM_CONTROL],
                offset_policy_raw=[offset[i] / scale for i in POLICY_FROM_CONTROL],
                biased_policy_raw=[biased[i] for i in POLICY_FROM_CONTROL])


def load_prior(reference_path, pair_path, *, mechanism, bank_path, root):
    reference_path, pair_path, root = Path(reference_path), Path(pair_path), Path(root)
    reference = json.loads(reference_path.read_text())
    values = validate_values(reference, mechanism["v5_control"])
    for name, expected in reference["sources_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Static reference source changed: " + name)
    source_configs = [root / name for name in reference["sources_sha256"] if name.endswith("config/contract.json")]
    source_banks = [expected for name, expected in reference["sources_sha256"].items() if name.endswith("pose_bank.json")]
    if len(source_configs) != 1 or source_banks != [sha(bank_path)]:
        raise ValueError("Static reference must bind the current reset geometry")
    original = json.loads(source_configs[0].read_text())
    for key in ("asset_directory", "asset_manifest_sha256", "v5_control", "control_math_sha256", "contact_material",
                "physics_dt", "policy_dt"):
        if original[key] != mechanism[key]:
            raise ValueError("Static prior mechanics differ: " + key)
    pair = json.loads(pair_path.read_text())
    if (pair.get("schema") != "scratch-static-load-native-pair-v1" or pair.get("all_checks_pass") is not True
            or not pair.get("checks") or not all(pair["checks"].values()) or pair.get("new_PPO") != 0
            or len(pair.get("diagnostic_result_sha256", {})) != 2):
        raise ValueError("Need the completed, matched zero-PPO native validation")
    modes = []
    for name, expected in pair["diagnostic_result_sha256"].items():
        result_path = root / name
        if sha(result_path) != expected:
            raise ValueError("Native static-load evidence changed")
        result = json.loads(result_path.read_text())
        if result["diagnostic_reference_sha256"] != sha(reference_path) or result["completed_updates"] != 0:
            raise ValueError("Native validation belongs to a different static reference")
        modes.append(result["diagnostic_target"])
    if sorted(modes) != ["constant_geometry", "constant_geometry_plus_offline_static_load_over_Kp"]:
        raise ValueError("Need both native constant-target arms")
    return dict(schema="scratch-static-load-initialization-v1", reference_sha256=sha(reference_path),
        native_pair_sha256=sha(pair_path), **values, runtime_compensation=False, all_outputs_trainable=True,
        source="nominal mechanism static torque divided by leg Kp; fresh trainable bias only",
        scope="Startup initialization prior; no standing, robustness or hardware qualification")


def initial_action(nominal, prior):
    expected = nominal.new_tensor(prior["geometry_policy_raw"])
    if nominal.shape != (6,) or not (nominal - expected).abs().max() < 2e-6:
        raise ValueError("Fresh initialization geometry differs from the validated static calculation")
    return nominal + nominal.new_tensor(prior["offset_policy_raw"])
