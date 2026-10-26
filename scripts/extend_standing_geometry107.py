#!/usr/bin/env python3
"""Append authenticated 105..107 degree geometry without changing old knots."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_robust_getup_bank import Builder
from build_v6_height_lookup import digest
from wheeled_algo.rolling_velocity import frozen_table_path


def extend(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Geometry generation never overwrites an existing table")
    original = frozen_table_path()
    table = json.loads(original.read_text())
    bundle = ROOT / "model/纯底盘_v7_flat_base107/urdf"
    builder = Builder(bundle, digest(bundle / "manifest.json"))
    if (table["asset_manifest_sha256"] != builder.manifest_sha256
            or table["model_spec_sha256"] != digest(bundle / "model_spec.json")
            or table["solver_sha256"] != digest(builder.solver_path)
            or table["fk_sha256"] != digest(bundle / "tools/v6_mechanism.py")
            or table["passive_knee_domain_deg"] != [40., 105.]):
        raise ValueError("Original table does not match this sealed mechanism")
    result = deepcopy(table)
    for name, side in (("left", "L"), ("right", "R")):
        old = table["sides"][name]
        q = dict(builder.nominal)
        nominal_beta = builder.knees(q)[0 if side == "L" else 1]
        for beta in np.linspace(nominal_beta, 105., 100)[1:]:
            q = builder.solve_knee(side, beta, q)
            builder.validate_q(q)
        hip, auxiliary = side + "_joint1", side * 2 + "_joint1"
        reference = builder.nominal[hip]
        boundary = 0 if side == "R" else -1
        if (abs((q[auxiliary] - q[hip]) - old["delta_rad"][boundary]) > 1e-9
                or np.linalg.norm(builder.wheel(side, q)[0] - old["wheel_center_b_m"][boundary]) > 1e-9):
            raise ValueError("105 degree continuation selected a different closure branch")
        extra = []
        states = [q]
        for beta in np.linspace(105., 107., 17)[1:]:
            q = builder.solve_knee(side, beta, q)
            builder.validate_q(q)
            extra.append((float(q[auxiliary] - q[hip]), float(beta), builder.wheel(side, q)[0].tolist()))
            states.append(q)
        checks = []
        for before, after in zip(states[:-1], states[1:], strict=True):
            delta = .5 * (before[auxiliary] + after[auxiliary]) - reference
            exact = builder.solve({**builder.active, auxiliary: reference + delta}, before)
            builder.validate_q(exact)
            estimate = .5 * (builder.wheel(side, before)[0] + builder.wheel(side, after)[0])
            checks.append(dict(kind="appended_midpoint", delta_rad=float(delta),
                               error_mm=float(np.linalg.norm(estimate - builder.wheel(side, exact)[0]) * 1000)))
        origin, axis = np.asarray(old["hip_origin_b_m"]), np.asarray(old["hip_axis_b"])
        for index in (0, 8, 16):
            start = states[index]
            point, wheel_axis, _ = builder.wheel(side, start)
            for common in (-np.pi, -np.pi / 2, np.pi / 2, np.pi):
                q = dict(start)
                for fraction in np.linspace(0., 1., 33)[1:]:
                    prescribed = dict(builder.active)
                    prescribed[hip] = start[hip] + fraction * common
                    # Fix the physical knee during this independent FK check.
                    # Solving two rounded active coordinates at the hard stop
                    # otherwise turns solver residual into a tiny limit breach.
                    del prescribed[auxiliary]
                    prescribed[builder.passive_name(side)] = start[builder.passive_name(side)]
                    q = builder.solve(prescribed, q)
                    builder.validate_q(q)
                vector = point - origin
                estimate = (origin + np.cos(common) * vector + np.sin(common) * np.cross(axis, vector)
                            + (1 - np.cos(common)) * axis * (axis @ vector))
                actual, actual_axis, _ = builder.wheel(side, q)
                if np.linalg.norm(actual_axis - wheel_axis) > 1e-8:
                    raise ValueError("Appended geometry changed the wheel axle")
                checks.append(dict(kind="appended_common_rotation", beta_deg=105. + index / 8.,
                                   common_rotation_rad=float(common),
                                   error_mm=float(np.linalg.norm(estimate - actual) * 1000)))
        maximum = max(row["error_mm"] for row in checks)
        if maximum > .1:
            raise ValueError("Appended interpolation/direct FK exceeds 0.1 mm")
        extra.sort()
        new = result["sides"][name]
        for key, column in (("delta_rad", 0), ("passive_inner_deg", 1), ("wheel_center_b_m", 2)):
            values = [row[column] for row in extra]
            new[key] = values + old[key] if side == "R" else old[key] + values
            retained = new[key][16:] if side == "R" else new[key][:-16]
            if retained != old[key]:
                raise ValueError("An original lookup value was changed")
        if not np.all(np.diff(new["delta_rad"]) > 0):
            raise ValueError("Extended relative coordinate is not monotonic")
        new["validation"]["appended_checks"] = checks
        new["validation"]["max_error_mm"] = max(maximum, old["validation"]["max_error_mm"])
    result.update(schema="paired107_standing_geometry_extension_v1",
                  passive_knee_domain_deg=[40., 107.], knots_per_side=273,
                  generator_sha256=digest(Path(__file__)), max_closure_gap_m=builder.max_gap_m,
                  extension=dict(source=str(original.relative_to(ROOT)), source_sha256=digest(original),
                                 original_knots_unchanged=True, appended_knots_per_side=16,
                                 purpose="Cover the existing 107 degree stop, retain invalid-outside-table handling",
                                 hardware_calibrated=False, dynamics_validated=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(output), sha256=digest(output),
                          max_closure_gap_m=builder.max_gap_m,
                          interpolation_mm={k:v["validation"]["max_error_mm"] for k,v in result["sides"].items()})))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    extend(parser.parse_args().output)
