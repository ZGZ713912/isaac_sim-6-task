"""Bind the user's 110-degree zero and 40–110 degree stops to the v2.0.2 plant.

Run on this task's copied asset only. CAD frames, masses and measured force
knots are preserved; the original source workspace is never written.
"""
from copy import deepcopy
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

import numpy as np

from wheeled_tasks.chassis.full_robot_asset import ACTIVE, SPRINGS, closure_residual, digest, fk, solve_pose, write_json
from wheeled_tasks.chassis.traditional_model import virtual_work

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/"model/wheel_leg_v2.0.2/isaac"
INNER_DEG = [40., 110.]
ENCODER_ZERO_INNER_DEG = 110.


def main():
    manifest = json.loads((BUNDLE/"manifest.json").read_text())
    before = {name:digest(BUNDLE/name) for name in manifest["files_sha256"]}
    if before != manifest["files_sha256"]:
        raise ValueError("Input asset differs from authenticated manifest")
    old_manifest = digest(BUNDLE/"manifest.json")
    spec = json.loads((BUNDLE/"model_spec.json").read_text())
    original = deepcopy(spec)
    knees = ("L_joint2", "R_jonit2")
    for name, sign in zip(knees, (1., -1.)):
        joint = next(j for j in spec["joints"] if j["name"] == name)
        limits = sorted((np.deg2rad(INNER_DEG)-spec["passive_knee_zero_inner_rad"][name])/sign)
        joint["limit"].update(lower=str(limits[0]), upper=str(limits[1]))
    assert spec["bodies"] == original["bodies"] and spec["constraints"] == original["constraints"]
    write_json(BUNDLE/"model_spec.json", spec)
    spec_hash = digest(BUNDLE/"model_spec.json")
    # Change only the two knee limit attributes; preserve all supplied USD frames.
    # The free-gimbal variant references robot.usda and only overrides its gimbal.
    for file in ("robot.usda",):
        text = (BUNDLE/file).read_text()
        for name in knees:
            joint = next(j for j in spec["joints"] if j["name"] == name)
            pattern = r'(def PhysicsRevoluteJoint "'+name+r'"\s*\{)(.*?)(\n\s*\})'
            def replace(match):
                body = match[2]
                for key in ("lower", "upper"):
                    body, count = re.subn(r'(float physics:'+key+r'Limit = )[^\n]+',
                        lambda m:m[1]+format(np.rad2deg(float(joint["limit"][key])), ".17g"), body)
                    assert count == 1
                return match[1]+body+match[3]
            text, count = re.subn(pattern, replace, text, flags=re.S)
            assert count == 1
        (BUNDLE/file).write_text(text)
    tree = ET.parse(BUNDLE/"robot.urdf")
    for name in knees:
        joint = next(j for j in spec["joints"] if j["name"] == name)
        tree.find(f"joint[@name='{name}']/limit").attrib.update(joint["limit"])
    ET.indent(tree, space="  ")
    tree.write(BUNDLE/"robot.urdf", encoding="utf-8", xml_declaration=True)

    curve = json.loads((BUNDLE/"gas_spring_model.json").read_text())
    encoder = json.loads((BUNDLE/"encoder_geometry.json").read_text())
    rows, q, gap = [], spec["nominal_joint_pos"], 0.
    for beta in np.linspace(*INNER_DEG, 281):
        q = solve_pose(spec, dict(L_joint1=.42, R_joint1=-.42, L_joint3=0., R_joint3=0.,
            L_joint2=np.deg2rad(beta)-spec["passive_knee_zero_inner_rad"][knees[0]],
            R_jonit2=-(np.deg2rad(beta)-spec["passive_knee_zero_inner_rad"][knees[1]])), q)
        _, mapping, _ = virtual_work(spec, q)
        gap = max(gap, float(np.linalg.norm(closure_residual(spec, q).reshape(-1, 3), axis=1).max()))
        rows.append((float(beta), dict(q), mapping, fk(spec, q)))
    for word, side, columns in (("left", "L", slice(0, 2)), ("right", "R", slice(3, 5))):
        hip, crank, slide = side+"_joint1", side*2+"_joint1", side+"_spring_slide"
        sign = 1. if side == "L" else -1.
        delta = [(r[1][crank]-r[1][hip])*sign for r in rows]
        assert np.all(np.diff(delta)>0.)
        curve["sides"][word] = dict(motor_names=[hip, crank], delta_rad=delta,
            beta_deg=[r[0] for r in rows],
            compression_m=[spec["spring_binding"][slide]["compression_at_q_zero_m"]-r[1][slide] for r in rows],
            length_jacobian_m_per_rad=[r[2][SPRINGS.index(slide), columns].tolist() for r in rows], closure_max_m=gap)
        ordered = sorted(rows, key=lambda r:r[1][crank]-r[1][hip])
        wheel = next(b for b in spec["bodies"] if b["name"] == side+"_link3")["collisions"][0]
        poses = [r[3][side+"_link3"]@np.array(wheel["origin"]) for r in ordered]
        encoder["sides"][word].update(delta_rad=[r[1][crank]-r[1][hip] for r in ordered],
            passive_inner_deg=[r[0] for r in ordered], wheel_center_b_m=[p[:3,3].tolist() for p in poses],
            wheel_axis_b=poses[0][:3,2].tolist())
    signed_low = max(row["delta_rad"][0] for row in curve["sides"].values())
    signed_high = min(row["delta_rad"][-1] for row in curve["sides"].values())
    soft = dict(minimum=signed_low, maximum=signed_high, margin=.03, projection="outward_relative_torque_v1")
    curve.update(model_spec_sha256=spec_hash, geometry_beta_domain_deg=INNER_DEG,
        soft_limit_active_difference_rad=[signed_low, signed_high],
        right_leg_symmetry_assumed=True, right_leg_symmetry_basis="user accepted 2026-10-30")
    encoder.update(model_spec_sha256=spec_hash, passive_knee_domain_deg=INNER_DEG,
        hardware_zero_reference_inner_deg=ENCODER_ZERO_INNER_DEG, max_closure_gap_m=gap)
    write_json(BUNDLE/"gas_spring_model.json", curve)
    write_json(BUNDLE/"encoder_geometry.json", encoder)
    wheel = json.loads((BUNDLE/"wheel_model.json").read_text())
    wheel["asset_rebinding"]["model_spec_sha256"] = spec_hash
    write_json(BUNDLE/"wheel_model.json", wheel)
    runtime = json.loads((BUNDLE/"runtime.json").read_text())
    runtime["v5_control"]["gas_spring_compensation"].update(model_sha256=digest(BUNDLE/"gas_spring_model.json"), relative_soft_limits=soft)
    runtime["identified_wheel_response"]["model_sha256"] = digest(BUNDLE/"wheel_model.json")
    runtime["right_gas_spring_symmetry_basis"] = "user accepted 2026-10-30"
    write_json(BUNDLE/"runtime.json", runtime)
    limits = json.loads((BUNDLE/"mechanical_limits.json").read_text())
    limits.update(knee_inner_limits_deg=INNER_DEG, relative_soft_limits=soft,
        hardware_zero_reference_inner_deg=ENCODER_ZERO_INNER_DEG,
        knee_limit_source="user corrected encoder-zero reference to 110 and limits to 40-110 degrees 2026-10-30",
        knee_limit_coordinate="physical included angle between thigh and shank; not two independent motor stops")
    write_json(BUNDLE/"mechanical_limits.json", limits)
    manifest.update(knee_inner_limits_deg=INNER_DEG, hardware_zero_reference_inner_deg=ENCODER_ZERO_INNER_DEG,
        asset_revision="full_robot_v202_zero110_knee_40_110",
        files_sha256={name:digest(BUNDLE/name) for name in manifest["files_sha256"]})
    write_json(BUNDLE/"manifest.json", manifest)
    receipt = json.loads((BUNDLE.parent/"build_receipt.json").read_text())
    receipt.update(new_model_spec_sha256=spec_hash, asset_manifest_sha256=digest(BUNDLE/"manifest.json"),
        geometry_max_closure_gap_m=gap, traditional_measured_limits_override=True)
    write_json(BUNDLE.parent/"build_receipt.json", receipt)
    write_json(BUNDLE.parent/"measured_limits_receipt.json", dict(schema="v202-measured-limits-override/v1",
        source="direct user instruction 2026-10-30", knee_inner_limits_deg=INNER_DEG,
        encoder_zero_reference_inner_deg=ENCODER_ZERO_INNER_DEG,
        right_gas_spring_symmetry="accepted modeling convention", upstream_manifest_sha256=old_manifest,
        result_manifest_sha256=digest(BUNDLE/"manifest.json"), model_spec_sha256=spec_hash,
        generator_sha256=digest(__file__), geometry_max_closure_gap_m=gap,
        changes={name:dict(before=before[name], after=digest(BUNDLE/name)) for name in before if before[name]!=digest(BUNDLE/name)},
        original_workspace_modified=False, cad_frames_and_mass_properties_unchanged=True,
        measured_force_knots_unchanged=True, force_measurement_domain_deg=curve["measured_beta_domain_deg"],
        dynamic_behavior_qualification=False))
    print("Applied 110-degree zero reference and 40–110 degree stops; closure gap", gap, flush=True)


if __name__ == "__main__":
    main()
