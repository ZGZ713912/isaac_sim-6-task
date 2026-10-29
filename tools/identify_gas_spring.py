#!/usr/bin/env python3
"""Geometry-constrained LEFT gas-spring identification, with PhysX calibration.

Run geometry first; run scripts/probe_identified_gas_spring.py in Isaac Sim;
then fit using --native-probe. Real data NPZ is exported with the audited RMCS
processor. No production asset, training contract or existing fit is overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "model/纯底盘_v6_232mm_stop105/urdf"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(bundle, angles, *, side="left"):
    if side not in ("left", "right"):
        raise ValueError("Expected left or right leg")
    sys.path.insert(0, str(bundle / "tools"))
    from build_v6_closedchain import closure_error, solve_pose, POSE_COORDINATES
    spec = json.loads((bundle / "model_spec.json").read_text())
    manifest = json.loads((bundle / "manifest.json").read_text())
    names = [j["name"] for j in spec["joints"]]
    active = [names.index(n) for n in manifest["control_joint_names"]]
    passive = [i for i in range(len(names)) if i not in active]
    slide_name = "L_spring_slide" if side == "left" else "R_spring_slide"
    slide = names.index(slide_name)
    motor_columns = slice(0, 2) if side == "left" else slice(3, 5)
    q = spec["nominal_joint_pos"]
    poses, compressions, mappings, gaps = [], [], [], []
    for angle in angles:
        knee = math.radians(angle) - (math.pi - 2.3573)
        prescribed = dict(zip(POSE_COORDINATES, [.42, knee, 0., -.42, -knee, 0.]))
        q = solve_pose(spec, prescribed, q)
        columns = []
        for name in names:
            plus, minus = dict(q), dict(q)
            plus[name] += 1e-6
            minus[name] -= 1e-6
            columns.append((closure_error(spec, plus)-closure_error(spec, minus))/2e-6)
        jac = np.column_stack(columns)
        mapping = np.zeros((len(names), len(active)))
        mapping[active] = np.eye(len(active))
        mapping[passive] = np.linalg.lstsq(jac[:, passive], -jac[:, active], rcond=1e-8)[0]
        assert np.max(abs(jac @ mapping)) < 1e-7
        poses.append([q[n] for n in names])
        compressions.append(spec["spring_binding"][slide_name]["compression_at_q_zero_m"]-q[slide_name])
        mappings.append(mapping[slide, motor_columns])
        gaps.append(float(np.max(abs(closure_error(spec, q)))))
    return dict(beta_deg=np.asarray(angles), compression_m=np.array(compressions),
                jacobian_m_per_rad=np.array(mappings), poses=np.array(poses),
                names=np.array(names), closure_m=np.array(gaps))


def linear_basis(x, knots):
    x = np.asarray(x)
    assert np.isfinite(x).all() and (x >= knots[0]-1e-12).all() and (x <= knots[-1]+1e-12).all()
    index = np.clip(np.searchsorted(knots, x, side="right")-1, 0, len(knots)-2)
    fraction = (x-knots[index])/(knots[index+1]-knots[index])
    basis = np.zeros((len(x), len(knots)))
    basis[np.arange(len(x)), index] = 1-fraction
    basis[np.arange(len(x)), index+1] = fraction
    return basis


def force(model, compression):
    compression = np.asarray(compression)
    lo, hi = model["compression_domain_m"]
    if np.any(compression < lo-1e-8) or np.any(compression > hi+1e-8):
        raise ValueError(f"Compression outside identified domain [{lo}, {hi}]")
    if model["kind"] == "monotone_cubic":
        return np.polynomial.polynomial.polyval(compression/.08, model["coefficients_n"])
    return np.interp(compression, model["compression_knots_m"], model["force_knots_n"])


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("geometry", "fit"))
    p.add_argument("--bundle", type=Path, default=BUNDLE)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--native-probe", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "geometry":
        g = geometry(args.bundle, np.arange(48., 104.01, .5))
        np.savez_compressed(args.out / "geometry.npz", **g)
        (args.out / "geometry.json").write_text(json.dumps(dict(
            bundle=str(args.bundle.resolve()), source_sha256=digest(__file__),
            files_sha256={n: digest(args.bundle / n) for n in ("manifest.json", "model_spec.json", "robot.usda", "fit_10mpa.json")},
            beta_domain_deg=[48., 104.], left_motor_order=["L_joint1", "LL_joint1"],
            mapping="dl/dq_sim; tau_holding=-J_length.T*F; V6 motor angle differences match RMCS geometry table",
            max_closure_m=float(g["closure_m"].max())), indent=2)+"\n")
        print("GEOMETRY_COMPLETE",len(g["beta_deg"]),flush=True)
        return
    g = np.load(args.out / "geometry.npz")
    d = np.load(args.out / "real_data.npz")
    probe = json.loads(args.native_probe.read_text())
    assert probe["status"] == "completed" and probe["mode"] == "unit"
    assert probe["all_cases_settled"], "Reject dynamic/saturated sensitivity calibration"
    assert probe["gravity"] == [0., 0., 0.]
    # Infer sensitivity from free native static equilibrium, with no spring FF.
    # Unit cases have constant 100 N force; use the final ACTUAL pose, not target.
    factors = []
    for row in probe["cases"]:
        if row["constant_force_n"] != 100.:
            continue
        beta = row["actual_beta_deg"]
        j = np.array([np.interp(beta, g["beta_deg"], g["jacobian_m_per_rad"][:, i]) for i in range(2)])
        factors.append(np.asarray(row["holding_torque_nm"])/(-100*j))
    factors = np.array(factors)
    scale = np.median(factors, axis=0)
    assert np.all(abs(scale-1) < .1), (scale, factors)
    beta = d["beta_deg"]
    good = ((beta >= 56) & (beta <= 101) & (abs(d["relative_velocity_rad_s"]) > .06)
            & (abs(d["relative_acceleration_rad_s2"]) < 3) & (abs(d["hip_v"]) < .18))
    rows = {k:d[k][good] for k in d.files}
    b = rows["beta_deg"]
    s = np.interp(b, g["beta_deg"], g["compression_m"])
    j = np.column_stack([np.interp(b, g["beta_deg"], g["jacobian_m_per_rad"][:, i])*scale[i] for i in range(2)])
    lever = (-j[:,0]+j[:,1])/2
    y = rows["relative_holding_torque_nm"]
    direction = np.sign(rows["relative_velocity_rad_s"])
    # Repeat-pass averaging and equal angle/direction/source weights.
    keys = np.column_stack((rows["source"], rows["stage"], np.round(b*2), direction))
    train = rows["validation"] == 0
    pass_rows = []
    for key in np.unique(keys[train], axis=0):
        mask = train & np.all(keys == key, axis=1)
        pass_rows.append([key[0], key[2], key[3], b[mask].mean(), s[mask].mean(),
                          lever[mask].mean(), y[mask].mean()])
    passes = np.array(pass_rows)
    cells = []
    for key in np.unique(passes[:,:3], axis=0):
        cells.append(passes[np.all(passes[:,:3] == key, axis=1)].mean(axis=0))
    cells = np.array(cells)
    knots_beta = np.array([56,58,60,65,70,75,80,85,90,94,96,97,98,99,100,101.])
    knots_s = np.interp(knots_beta, g["beta_deg"], g["compression_m"])[::-1]
    domains = [float(knots_s[0]), float(knots_s[-1])]
    # Allow direction-dependent nuisance load in MOTOR Nm. This is not exported
    # as spring force or attributed uniquely to motor vs seal friction.
    nuisance = -linear_basis(cells[:,3], knots_beta)*cells[:,2,None]
    models = {}
    for kind in ("monotone_cubic", "effective_monotone_table"):
        if kind == "monotone_cubic":
            u = cells[:,4]/.08
            fbase = np.column_stack((np.ones(len(u)),u,u*u,u*u*u))
            reconstruction = None
        else:
            # Nonnegative increments enforce force nondecreasing in compression.
            reconstruction = np.tril(np.ones((len(knots_s),len(knots_s))))
            fbase = linear_basis(cells[:,4], knots_s) @ reconstruction
        X = np.column_stack((fbase*cells[:,5,None], nuisance))
        fit = lsq_linear(X, cells[:,6], bounds=(0., np.inf), tol=1e-12, max_iter=1000)
        assert fit.success
        n = fbase.shape[1]
        entry = dict(kind=kind, compression_domain_m=domains, beta_domain_deg=[56.,101.],
                     direction_half_width_nm=fit.x[n:].tolist(), direction_beta_knots_deg=knots_beta.tolist(),
                     physical_pressure_identified=False, side="left", training_cells=len(cells))
        if reconstruction is None:
            entry["coefficients_n"] = fit.x[:n].tolist()
        else:
            entry.update(compression_knots_m=knots_s.tolist(), force_knots_n=(reconstruction@fit.x[:n]).tolist())
        models[kind] = entry
    catalogue = json.loads((args.bundle/"fit_10mpa.json").read_text())
    models["catalogue"] = dict(kind="monotone_cubic", coefficients_n=catalogue["monomial_coefficients_n"],
                                compression_domain_m=domains, beta_domain_deg=[56.,101.])
    prior_residual = cells[:,6]-cells[:,5]*force(models["catalogue"],cells[:,4])
    prior_width = lsq_linear(nuisance,prior_residual,bounds=(0.,np.inf),tol=1e-12)
    assert prior_width.success
    models["catalogue"].update(direction_beta_knots_deg=knots_beta.tolist(),
                                direction_half_width_nm=prior_width.x.tolist())
    validation = {}
    for name, model in models.items():
        gas = lever*force(model,s)
        width = np.interp(b,model["direction_beta_knots_deg"],model["direction_half_width_nm"])
        prediction = gas-width*direction
        groups = {}
        for source in (0,1):
            mask = (rows["source"]==source) & (rows["validation"]==1)
            groups[str(source)] = dict(rows=int(mask.sum()), mean_load_residual_rms_nm=rms(y[mask]-gas[mask]),
                                      with_direction_residual_rms_nm=rms(y[mask]-prediction[mask]),
                                      mean_joint_residual_rms_nm=rms(np.r_[rows["hip_torque_nm"][mask]-gas[mask],rows["knee_torque_nm"][mask]+gas[mask]]))
        validation[name] = groups
    # Fixed criteria: flexible model represents mounted effective force; ordinary
    # pressure/cubic family is retained as the physically simpler comparison.
    result = dict(status="fitted_pending_native_curve_validation", asset_bundle=str(args.bundle),
                  native_unit_probe=str(args.native_probe), native_unit_probe_sha256=digest(args.native_probe),
                  native_sensitivity_scale=scale.tolist(), native_sensitivity_individual=factors.tolist(),
                  models=models, validation=validation, selected="effective_monotone_table",
                  training_rule="only slow stages1..6 from both files; stage7+ held out",
                  limitations=["left leg only; fixed 232 mm mounting and logged beta calibration assumed",
                               "effective mounted force, not unique gas pressure or seal friction identification",
                               "no extrapolation outside compression/beta domain; no right-leg identification",
                               "direction half-width is a diagnostic nuisance load, not enabled in the passive gas model",
                               "no production model/contract overwritten; native nonlinear validation is required"])
    (args.out / "identified_model.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(args.out/"fit_data.npz", **rows, compression_m=s, effective_lever_m_per_rad=lever)
    print(json.dumps(dict(native_scale=scale.tolist(),validation=validation),indent=2),flush=True)


if __name__ == "__main__":
    main()
