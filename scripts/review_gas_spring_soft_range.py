#!/usr/bin/env python3
"""Export static model coverage and a figure; no simulator, learner or weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from wheeled_tasks.chassis.encoder_spring_feedforward import EncoderGasSpringFeedforward, resolve
from wheeled_tasks.chassis.training_gas_spring import resolve as resolve_spring


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v1.json").read_text())
    cfg = json.loads((ROOT/"contracts/scratch_standing_base_mechanism_v2.json").read_text())
    spec_path = ROOT/cfg["asset_directory"]/"model_spec.json"
    spring = resolve_spring(ROOT, plan["gas_spring_model"], json.loads(spec_path.read_text()))
    ff = EncoderGasSpringFeedforward(resolve(ROOT, plan["gas_spring_feedforward"], spring, spec_path))
    left, right = ff.get(0, "delta_rad"), ff.get(1, "delta_rad")
    delta = torch.linspace(float(left[0]), float(left[-1]), 4001, dtype=torch.float64)
    mirror = torch.linspace(float(right[-1]), float(right[0]), 4001, dtype=torch.float64)
    q = torch.stack((delta*0.-.42, delta-.42, delta*0.+.42, mirror+.42), -1)
    holding = ff(q)
    data = {k:v.detach().numpy() for k,v in ff.last.items()}
    checks = dict(full_soft_range_geometry_valid=bool(data["geometry_valid"].all()),
        finite_nonzero_compensation_every_motor=bool(np.isfinite(data["holding_effort_nm"]).all()
            and (np.abs(data["holding_effort_nm"]) > .01).all()),
        holding_alone_within40nm=bool((holding.abs() < 40.).all()),
        positive_force=bool((data["axial_force_n"] > 0.).all()),
        no_measurement_claim_for_tails=spring["resolved_trend_extension"]["extrapolation_validated_by_measurement"] is False)
    if not all(checks.values()):
        raise RuntimeError("Spring model does not cover the complete soft range: "+str(checks))
    beta = np.degrees(data["knee_inner_rad"])
    samples = []
    for angle in (40.,45.,50.,56.,65.,85.,101.,103.,107.):
        samples.append(dict(knee_deg=angle, measured=56. <= angle <= 101.,
            estimated_axial_force_n=[float(np.interp(angle,beta[:,i],data["axial_force_n"][:,i])) for i in range(2)],
            holding_nm=[float(np.interp(angle,beta[:,i//2],data["holding_effort_nm"][:,i])) for i in range(4)]))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays = output/"model_curve_arrays.npz"
    np.savez_compressed(arrays, knee_deg=beta, **data)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1,figsize=(9,7),sharex=True,layout="constrained")
    for ax in axes:
        ax.axvspan(40.,56.,color="tab:orange",alpha=.12)
        ax.axvspan(101.,107.,color="tab:orange",alpha=.12)
        ax.axvline(56.,color="grey",lw=.8,ls=":")
        ax.axvline(101.,color="grey",lw=.8,ls=":")
        ax.grid(alpha=.25)
        ax.set_xlim(40.,107.)
    axes[0].plot(beta[:,0],data["axial_force_n"][:,0],label="Shared axial-force law (both legs)")
    knots = np.array(spring["model"]["compression_knots_m"])
    measured_beta = np.interp(knots,data["compression_m"][::-1,0],beta[::-1,0])
    axes[0].scatter(measured_beta,spring["model"]["force_knots_n"],s=22,color="black",label="Unchanged measured knots",zorder=3)
    axes[0].set_ylabel("Axial force [N]")
    axes[0].set_title("Gas-spring model over 40-107 deg soft range\nOrange: endpoint-trend extrapolation, unmeasured")
    axes[0].legend(loc="upper right")
    for i,name in enumerate(("L hip", "L auxiliary", "R hip", "R auxiliary")):
        axes[1].plot(beta[:,i//2],data["holding_effort_nm"][:,i],ls="-" if i<2 else "--",label=name)
    axes[1].set_ylabel("Holding feedforward [N m]")
    axes[1].set_xlabel("Passive knee included angle [deg]")
    axes[1].legend(ncol=2)
    png = output/"gas_spring_full_soft_range.png"
    fig.savefig(png,dpi=180)
    plt.close(fig)
    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(schema="gas-spring-soft-range-static-review-v1", checks=checks,
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        model_sha256=spring["sha256"],extension_contract=plan["gas_spring_model"]["trend_extension"],
        geometry_sha256=ff.settings["geometry_sha256"],soft_knee_deg=[40.,107.],measured_knee_deg=[56.,101.],
        samples_per_side=4001,samples=samples,axial_force_range_n=[float(data["axial_force_n"].min()),float(data["axial_force_n"].max())],
        holding_peak_nm=float(holding.abs().max()),arrays=str(arrays),arrays_sha256=sha(arrays),figure=str(png),figure_sha256=sha(png),
        scope="Static model/geometry coverage only; figure is not a training curve, native trajectory, pressure identification or behavior qualification.",
        new_PPO=0,hardware_ready=False)
    (output/"review.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({k:report[k] for k in ("checks","axial_force_range_n","holding_peak_nm","figure")},ensure_ascii=False))


if __name__ == "__main__":
    main()
