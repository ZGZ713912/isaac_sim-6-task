#!/usr/bin/env python3
"""Reconstruct the exact model specification declared by the supplied flat actor."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "model/纯底盘_v6_232mm_stop105/urdf"
TARGET = ROOT / "model/纯底盘_v7_flat_base107/urdf"
MODEL_SHA = "0ac82045ab584f20091538aa782d93ea545382dc701faca1f8ae92aeb7cd1997"
PARENT_SHA = "875fa71e89d4d669af0a5471b2334f878d302e06f1b354fc245b95cb5d6f29ba"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def build(output=TARGET):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new asset directory; an existing bundle is never overwritten")
    if digest(PARENT / "manifest.json") != PARENT_SHA:
        raise ValueError("Parent asset manifest changed")
    parent = json.loads((PARENT / "manifest.json").read_text())
    for name, expected in parent["files_sha256"].items():
        if digest(PARENT / name) != expected:
            raise ValueError(f"Parent asset dependency changed: {name}")
    # The complete resulting byte stream, including every body and frame, has
    # the hash declared inside the user's ONNX and the deployed winding code.
    original = (PARENT / "model_spec.json").read_text()
    old, new = "1.048303061004253", "1.0832096460441396"
    if original.count(old) != 2:
        raise ValueError("Expected exactly two mirrored passive-knee limits")
    spec_bytes = original.replace(old, new).encode()
    if hashlib.sha256(spec_bytes).hexdigest() != MODEL_SHA:
        raise ValueError("Reconstruction does not match the ONNX's exact model specification")
    expected = deepcopy(json.loads(original))
    for joint in expected["joints"]:
        if joint["name"] == "L_joint2":
            joint["limit"]["upper"] = new
        elif joint["name"] == "R_jonit2":
            joint["limit"]["lower"] = "-" + new
    if json.loads(spec_bytes) != expected:
        raise ValueError("Unexpected geometry change")

    # Parent validation receipts remain in the parent; they are not new results.
    omit = {"README.md", "asset_revision.json", "build_manifest.json", "parent_comparison.json",
            "validation.json", "cpu_validation.json", "dynamics_validation.json",
            "physx_input_manifest.json", "physx_validation.json", "mechanical_limits.json"}
    output.mkdir(parents=True)
    for name in parent["files_sha256"]:
        if name in omit:
            continue
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PARENT / name, dest)
    (output / "model_spec.json").write_bytes(spec_bytes)
    for name in ("robot.urdf", "robot.xml"):
        path = output / name
        data = path.read_text()
        if data.count(old) != 2:
            raise ValueError(f"Expected two passive-knee limits in {name}")
        path.write_text(data.replace(old, new))
    path = output / "robot.usda"
    data = path.read_text()
    for before, after in (("float physics:upperLimit = 60.06334", "float physics:upperLimit = 62.06334"),
                          ("float physics:lowerLimit = -60.06334", "float physics:lowerLimit = -62.06334")):
        if data.count(before) != 1:
            raise ValueError("Expected one USD bound per passive knee")
        data = data.replace(before, after)
    path.write_text(data)
    changes = {"schema": "flat-base107-exact-reconstruction-v1", "parent_directory": str(PARENT.relative_to(ROOT)),
               "parent_manifest_sha256": PARENT_SHA, "model_spec_sha256": MODEL_SHA,
               "physical_changes": {"L_joint2.upper": [old, new], "R_jonit2.lower": ["-" + old, "-" + new]},
               "all_other_spec_fields_exact": True, "knee_inner_limits_deg": [40., 107.],
               "source_of_target_identity": "user_ONNX_metadata_and_RMCS_paired_branch107_v1",
               "hardware_measurements_changed": False, "new_physics_validation": "pending"}
    (output / "reconstruction.json").write_bytes(encode(changes))
    (output / "README.md").write_text(
        "# Matched flat-base 107 model\n\nThe model_spec.json matches the input ONNX's SHA-256 exactly.\n"
        "Only the two passive knee limits differ from the preserved stop105 parent.\n"
        "Reconstruction provenance is in reconstruction.json. Old validation receipts are not reused.\n"
        "The copied parent tool sources retain their original provenance; use scripts/build_flat_base_asset.py to rebuild.\n")
    manifest = deepcopy(parent)
    manifest.update(asset_revision="v7_flat_base_matched107_v1", parent_manifest_sha256=PARENT_SHA,
        physical_change_scope="Only the two passive-knee coordinate bounds, 105 to 107 degrees; exact ONNX model-spec identity",
        knee_inner_limits_deg=[40., 107.], knee_limits_source="exact_ONNX_declared_model_spec_reconstruction",
        static_validation_passed=False, dynamics_validation="not_run_for_this_asset", physx_validation="pending",
        cpu_geometry_validation_passed=False, validation_receipts=[], training_ready=False)
    manifest.pop("build_manifest_sha256", None)
    manifest["files_sha256"] = {str(path.relative_to(output)): digest(path)
                                for path in sorted(output.rglob("*")) if path.is_file()}
    (output / "manifest.json").write_bytes(encode(manifest))
    return {"asset": str(output), "manifest_sha256": digest(output / "manifest.json"), **changes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=TARGET)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False))
