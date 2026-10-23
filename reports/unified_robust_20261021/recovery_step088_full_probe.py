"""Check every active recovery input in079 full/nominal recordings for seams."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration045_runner import write
from wheeled_algo.robust_getup import tilt_gate


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    checkpoint = RUNS / "pilot033_dynamic_recovery079/model_final.pt"
    mean, _ = load_mean(checkpoint)
    candidates = {step: deepcopy(mean) for step in (.2, .15)}
    for step, model in candidates.items():
        model.recovery_leg_step = step
    result = {}
    for run in ("dynamic_recovery_full079", "dynamic_recovery_target079"):
        directory = RUNS / run / "robust"
        report = json.loads((directory / "report.json").read_text())
        assert sha(directory / "trajectory.npz") == report["trajectory_sha256"]
        with np.load(directory / "trajectory.npz") as data:
            all_obs = torch.from_numpy(data["observations"].reshape(-1,35).copy())
            active = torch.from_numpy((data["alive"] & ~data["passive"]).ravel().copy())
            gates = tilt_gate(all_obs, mean.stance_aware_recovery, mean.recovery_shape_aware).ravel()
            indices = torch.where(active & (gates > 0.))[0]
            rows = {str(step): dict(checked=0, branch_rows=0, partial_branch_rows=0,
                maximum_circular_target_delta_rad=0., examples=[]) for step in candidates}
            with torch.inference_mode():
                for ids in indices.split(2048):
                    obs, gate = all_obs[ids], gates[ids]
                    original = mean(obs)
                    for step, model in candidates.items():
                        changed = model(obs)
                        assert torch.equal(changed[:,4:],original[:,4:])
                        delta = changed[:,:4] - original[:,:4]
                        circular = torch.atan2((delta*.25).sin(),(delta*.25).cos()).abs().amax(-1)
                        branch = delta.abs().amax(-1)>12.
                        partial = branch & (gate<1.)
                        row = rows[str(step)]
                        row["checked"] += len(ids)
                        row["branch_rows"] += int(branch.sum())
                        row["partial_branch_rows"] += int(partial.sum())
                        row["maximum_circular_target_delta_rad"] = max(row["maximum_circular_target_delta_rad"],float(circular.max()))
                        for local in torch.where(partial)[0][:max(0,3-len(row["examples"]))]:
                            flat = int(ids[local]);tick,lane=divmod(flat,len(report["lanes"]))
                            row["examples"].append(dict(lane=lane,case=report["lanes"][lane]["name"],
                                time_s=float(data["time"][tick,lane]),authority=float(gate[local]),
                                circular_delta_rad=float(circular[local])))
            result[run] = dict(trajectory_sha256=report["trajectory_sha256"],
                total_recorded_frames=len(all_obs), active_frames=int(active.sum()),
                active_recovery_frames=len(indices), steps=rows)
    value = dict(schema="recovery088-full-recorded-scope-v1", checkpoint_sha256=sha(checkpoint),
        recordings=result,new_macro_updates=0,scope="All recorded active recovery frames; no new closed-loop or physical evidence",
        native_behavior_proven=False,full_training_admitted=False)
    write(REPORTS/"recovery_step088_full_cpu.json",value)
    print(json.dumps(value),flush=True)


if __name__=="__main__":
    main()
