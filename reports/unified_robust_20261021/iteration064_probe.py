"""Read-only attribution of the remaining high-height startup excursion."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration036_compare import read, paired
from analyze_robust_getup import hf_rms


def main():
    torch.set_num_threads(1)
    parent, _ = load_mean(RUNS / "quiet_position_reference062/model_migrated.pt")
    tuned, _ = load_mean(RUNS / "pilot032_joint_quiet064/model_final.pt")
    old_head = deepcopy(tuned)
    old_head.quiet_velocity.high.load_state_dict(parent.quiet_velocity.high.state_dict())
    old_mlp = deepcopy(tuned)
    old_mlp.quiet_high.load_state_dict(parent.quiet_high.state_dict())
    anchor = torch.zeros(1, 35)
    anchor[:, 3], anchor[:, 9] = .42 * 5., -1.
    feature_anchor = torch.tanh(anchor * tuned.feature_scale)
    with torch.inference_mode():
        anchor_shift = tuned.quiet_high(feature_anchor) - parent.quiet_high(feature_anchor)
    recentered = deepcopy(tuned)
    with torch.no_grad():
        recentered.quiet_high[-1].bias.sub_(anchor_shift[0])
    result = dict(schema="joint064-high-startup-attribution-v1", runs={},
                  high_position_weight_before=parent.quiet_velocity.high.weight.tolist(),
                  high_position_weight_after=tuned.quiet_velocity.high.weight.tolist(),
                  canonical_anchor_observation=anchor[0].tolist(),
                  high_MLP_canonical_logit_shift=anchor_shift[0].tolist(),
                  new_macro_updates=0,
                  limitation="Same-input actor attribution is not closed-loop causal evidence")
    baseline = read("quiet_velocity_baseline036")
    for name in ("position_reference_static062", "joint_quiet_static064"):
        directory, report, analysis, data = evidence = read(name)
        matched = paired(baseline, evidence)
        lane_rows = []
        for row in report["lanes"]:
            i = row["lane"]
            if row["name"] != "stand_high":
                continue
            t = data["time"][:, i]
            full = data["alive"][:, i] & ~data["passive"][:, i]
            xy = data["local_position"][:, i, :2]
            relative = xy - xy[np.flatnonzero(full)[:1]]
            radius = np.linalg.norm(relative, axis=-1)
            at = int(np.where(full, radius, -1.).argmax())
            item = dict(lane=i, excursion_m=float(radius[at]), at_s=float(t[at]),
                        xy_at_peak_m=relative[at].tolist(),
                        estimated_position_before_peak_action_m=(data["observations"][at, i, 14:16] / 20.).tolist())
            x = torch.from_numpy(data["observations"][:, i].copy())
            with torch.inference_mode():
                actual = tuned(x)
                alternatives = dict(old_high_position_head=old_head(x), old_high_quiet_MLP=old_mlp(x),
                                    canonical_high_MLP_bias_recentered=recentered(x))
            for label, output in alternatives.items():
                delta = (output - actual).numpy()
                item[label] = {}
                for window, mask in (("startup", t < 1.), ("steady", t >= 3.)):
                    item[label][window] = dict(
                        leg_target_delta_rms_rad=float(np.sqrt(np.mean((.25 * delta[mask, :4]) ** 2))),
                        wheel_target_delta_rms_rad_s=float(np.sqrt(np.mean((10. * delta[mask, 4:]) ** 2))))
            lane_rows.append(item)
        worst = max(lane_rows, key=lambda row: row["excursion_m"])
        result["runs"][name] = dict(pairing=matched, trajectory_sha256=report["trajectory_sha256"],
                                    worst=worst, lanes=lane_rows)
        step = json.loads((directory.parent / "progress.json").read_text())["completed_updates"]
        writer = SummaryWriter(str(directory.parent), filename_suffix=".high_startup064")
        for key in ("lane", "excursion_m", "at_s"):
            writer.add_scalar("Diagnosis/high_startup064/" + key, worst[key], step)
        writer.close()
    (REPORTS / "iteration064_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: row["worst"] for name, row in result["runs"].items()}))


if __name__ == "__main__":
    main()
