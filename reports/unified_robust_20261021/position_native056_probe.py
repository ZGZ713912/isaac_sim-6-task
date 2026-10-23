"""Measure the actual new native observation, without hypothetical extra inputs."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path.insert(0,str(REPORTS))
from iteration036_compare import read, CASES


def main():
    directory,report,_,data = read("position_zero_static056")
    # Recorded observation is before this action, diagnostic position after it.
    # Therefore obs[t+1] is compared to diagnostic position[t], not position[t+1].
    estimate = data["observations"][1:,:,14:16]/20.
    time = data["time"][:-1]
    position = data["local_position"][:-1,:,:2].astype(np.float64)
    rows = []
    for row in report["lanes"]:
        i = row["lane"]
        start = np.flatnonzero(time[:,i] >= .2)[0]
        displacement = position[:,i]-position[start:start+1,i]
        selected = (time[:,i] >= .2) & data["alive"][:-1,i]
        unsaturated = selected & (np.abs(estimate[:,i]).max(-1) < .249)
        error = estimate[:,i].mean(-1)-displacement[:,0]
        crossing = np.flatnonzero(selected & (np.linalg.norm(displacement,axis=-1) > .05))
        item = dict(lane=i,case=row["name"],unsaturated_frames=int(unsaturated.sum()),
            max_unsaturated_error_m=float(np.abs(error[unsaturated]).max()),
            mae_unsaturated_m=float(np.abs(error[unsaturated]).mean()),
            actual_final_forward_m=float(displacement[-1,0]),actual_final_lateral_m=float(displacement[-1,1]),
            estimated_final_forward_m=float(estimate[-1,i].mean()),
            saturated_frames=int((selected & ~unsaturated).sum()),crossed_true_5cm=bool(len(crossing)))
        if len(crossing):
            at=crossing[0]
            item.update(first_crossing_seconds=float(time[at,i]),
                estimated_at_crossing_m=float(estimate[at,i].mean()),
                actual_forward_at_crossing_m=float(displacement[at,0]),
                actual_lateral_at_crossing_m=float(displacement[at,1]))
        rows.append(item)
    summary = {case:dict(worst_unsaturated_error_m=max(x["max_unsaturated_error_m"] for x in rows if x["case"]==case),
        mean_episode_mae_m=float(np.mean([x["mae_unsaturated_m"] for x in rows if x["case"]==case])),
        crossed_5cm=sum(x["crossed_true_5cm"] for x in rows if x["case"]==case),
        saturated_frames=sum(x["saturated_frames"] for x in rows if x["case"]==case)) for case in CASES}
    result = dict(schema="position056-actual-native-observation-v1",trajectory_sha256=report["trajectory_sha256"],
        actual_frames=int(data["observations"].shape[0]),summary=summary,lanes=rows,new_macro_updates=0,
        scope="Completed native zero-head trajectory. Truth only labels error; no policy or controller input uses it.")
    output=REPORTS/"position_native056_probe.json"
    output.write_text(json.dumps(result,indent=2)+"\n")
    writer=SummaryWriter(str(directory.parent),filename_suffix=".position_native056")
    for case,values in summary.items():
        for key,value in values.items():
            writer.add_scalar(f"Diagnosis/position_native056/{case}/{key}",value,441)
    writer.flush();writer.close()
    print(json.dumps({k:result[k] for k in ("schema","actual_frames","summary")}))


if __name__ == "__main__":
    main()
