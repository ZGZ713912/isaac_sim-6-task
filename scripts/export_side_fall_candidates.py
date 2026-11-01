#!/usr/bin/env python3
"""Export full measured candidate states without granting RL reset admission."""
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts"),str(ROOT/"tools")]
from wheeled_tasks.chassis.fallen_pose_bank import digest
from summarize_fallen_pose_bank import classify, group_modes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--banks",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Preserve existing catalogues")
    poses,excluded,sources,group_records = [],[],{},[]
    identity = None
    for path in args.banks:
        bank = json.loads(path.read_text())
        key = (bank["asset_manifest_sha256"],bank["model_spec_sha256"],bank["leg_motor_friction"]["model_sha256"])
        if bank["status"] != "completed" or (identity and key != identity) or bank["motor_effort_peak_nm"] != 0.:
            raise ValueError("Need matching completed, disabled new-friction banks")
        identity = key
        cad_path = path.parent/"cad_contacts.json"
        cad = json.loads(cad_path.read_text())
        comparison_path = path.parent/"analysis001/comparison.json"
        comparison = json.loads(comparison_path.read_text())
        if cad["source_bank_sha256"] != digest(path) or comparison["source_bank_sha256"] != digest(path):
            raise ValueError("Endpoint audit/analysis changed")
        metrics = {m["record_index"]:m for pair in comparison["pairs"] for m in (pair["off"],pair["moving"])}
        audits = {r["index"]:r for r in cad["records"]}
        trajectory = {}
        for sample in sorted(path.parent.glob("trajectory_screen*/summary.json")):
            audit = json.loads(sample.read_text())
            if audit["source_bank_sha256"] != digest(path):
                raise ValueError("Trajectory audit changed")
            for r in audit["records"]:
                trajectory[r["native_record"]] = dict(summary_path=str(sample.resolve()),summary_sha256=digest(sample),**r)
        sources[str(path.resolve())] = dict(bank_sha256=digest(path),cad_sha256=digest(cad_path),analysis_sha256=digest(comparison_path),
            dt_s=bank["physics_dt_s"],duration_s=bank["actual_timeout_s"],leg_friction=bank["leg_motor_friction"])
        for start,record in zip(bank["starts"],bank["records"]):
            if not start["leg_friction_enabled"]:
                continue
            i = record["index"]
            if record["status"] not in ("settled","not_settled") or audits[i]["review_required"]:
                excluded.append(dict(source_bank=str(path.resolve()),record_index=i,pair_id=start["pair_id"],
                    native_status=record["status"],cad_audit=audits[i]))
                continue
            pose = deepcopy(record["final_pose"])
            pose["name"] = f"side_candidate_{len(poses):03d}"
            pose.update(source_bank=str(path.resolve()),source_record=i,pair_id=start["pair_id"],
                source_bank_sha256=digest(path),source_placement_sha256=start["placement_sha256"],
                initial_knee_pair_deg=start["knee_pair_deg"],initial_orbital_pair_deg=start["orbital_pair_deg"],
                initial_roll_deg=start["signed_roll_deg"],placement_clearance_m=start["min_collision_clearance_m"],
                actual_physics_dt_s=bank["physics_dt_s"],observation_seconds=record["elapsed_s"],
                metrics=metrics[i],cad_endpoint_audit=audits[i],
                sampled_fk_trajectory_audit=trajectory.get(i),
                static_equilibrium_proven=False,admitted_as_training_reset=False,
                reset_velocity_instruction="Measured residual velocities are preserved. Zeroing them changes the state and needs a separate replacement check.")
            record_copy = deepcopy(record)
            record_copy["index"] = len(poses)
            record_copy["final_pose"] = pose
            record_copy["final_pose"]["cad_contact_audit"] = audits[i]
            pose["window_evidence"] = classify(record_copy) or "moving"
            if trajectory.get(i,{}).get("flagged_samples",0):
                pose["window_evidence"] = "endpoint_only_trajectory_flagged"
                record_copy["final_pose"]["cad_contact_audit"] = dict(review_required=True)
            poses.append(pose)
            group_records.append(record_copy)
    groups = group_modes(group_records)
    for g in groups:
        g.pop("representative_pose",None)
    output = dict(schema="wheel-leg-side-fall-candidate-catalogue-v1",source_banks=sources,
        exporter_sha256=digest(__file__),asset_manifest_sha256=identity[0],model_spec_sha256=identity[1],
        leg_motor_friction_model_sha256=identity[2],all_eight_motor_servos_disabled=True,motor_effort_peak_nm=0.,
        temporary_nonwheel_friction=.7,poses=poses,excluded=excluded,
        candidate_snapshot_count=len(poses),excluded_endpoint_count=len(excluded),
        evidence_counts=dict(Counter(p["window_evidence"] for p in poses)),
        body_orientation_navigation_groups=groups,orientation_group_angle_deg=12.,orientation_group_height_range_m=.035,
        grouping_scope="Finite selected snapshots. Orientation groups are navigation aids,not an exhaustive equilibrium count;full joint/gimbal states remain separate.",
        training_reset_admission=False,static_equilibrium_qualified=False,ppo_updates=0,hardware_motion=False,
        scope="Measured, mechanically/CAD-screened endpoint candidates with residual motion and trajectory exclusions retained. No learner or existing pose-bank integration.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps(dict(candidates=len(poses),excluded=len(excluded),body_orientation_groups=len(groups),evidence=output["evidence_counts"]),ensure_ascii=False))


if __name__ == "__main__":
    main()
