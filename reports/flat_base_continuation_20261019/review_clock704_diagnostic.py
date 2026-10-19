"""Compare shared raw trace windows; the short diagnostic cannot qualify a release."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/flat_base_20261019"
HERE = Path(__file__).resolve().parent
names = dict(pd200="rmcs704_pd200_diagnostic001", pd1000="rmcs704_envelope001")
records, traces = {}, {}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for name, folder in names.items():
    path = RUN / folder / "evaluation.json"
    records[name] = json.loads(path.read_text())
    if records[name]["status"] != "evaluated":
        raise SystemExit("Clock counterpart is still running")
    traces[name] = np.load(RUN / folder / "traces.npz")
a, b = traces["pd200"], traces["pd1000"]
count = min(len(a["active"]), len(b["active"]))
joint = a["active"][:count] & b["active"][:count]
checks = dict(same_checkpoint=records["pd200"]["checkpoint_sha256"] == records["pd1000"]["checkpoint_sha256"],
    same_seed=records["pd200"]["seed"] == records["pd1000"]["seed"],
    same_case_order=np.array_equal(a["groups"], b["groups"]),
    same_policy50=records["pd200"]["policy_hz"] == records["pd1000"]["policy_hz"] == 50,
    declared_physics=(records["pd200"]["physics_hz"], records["pd1000"]["physics_hz"]) == (200, 1000),
    first_observations_bit_exact=np.array_equal(a["observations"][0], b["observations"][0]),
    first_actions_bit_exact=np.array_equal(a["actions"][0], b["actions"][0]),
    raw_requests_bit_exact_shared_active=np.array_equal(a["raw_requested_commands"][:count][joint],
                                                       b["raw_requested_commands"][:count][joint]),
    diagnostic_ineligible=not records["pd200"]["diagnostic_counterfactual"]["qualification_eligible"])

def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        found = []
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                found.append(prefix + "/" + key)
            else:
                found.extend(differences(left[key], right[key], prefix + "/" + key))
        return found
    return [prefix] if left != right else []


control_paths = [ROOT / (RUN / folder / "config/contract.json").relative_to(ROOT) for folder in names.values()]
controls = [json.loads((ROOT / json.loads(p.read_text())["control_math_source"]).read_text()) for p in control_paths]
control_diff = differences(*controls)
checks["only_control_timing_and_provenance_differ"] = set(control_diff) == {
    "/control_id", "/simulation_timing_approximation", "/timing/decimation", "/timing/pc_control_dt", "/timing/physics_dt"}
windows = {}
for case in ("stand_230mm", "stand_305mm", "stand_430mm", "forward_3", "backward_3", "curve_low", "curve_low_reverse"):
    windows[case] = {}
    for lo, hi in ((0., 1.), (1., 2.), (2., 3.), (3., 9.)):
        rows = {}
        for clock, data in traces.items():
            mask = data["active"].copy()
            mask &= (data["groups"] == case)[None, :]
            ticks = np.arange(len(mask)) * .02
            mask &= ((ticks >= lo) & (ticks < hi))[:, None]
            vx, target = data["velocity"][:, :, 0][mask], data["commands"][:, :, 0][mask]
            yaw, yaw_target = data["omega"][:, :, 2][mask], data["commands"][:, :, 1][mask]
            rows[clock] = dict(sample_frames=int(mask.sum()), vx_mean_m_s=float(vx.mean()),
                command_vx_mean_m_s=float(target.mean()), vx_mae_m_s=float(np.abs(vx-target).mean()),
                yaw_mean_rad_s=float(yaw.mean()), yaw_mae_rad_s=float(np.abs(yaw-yaw_target).mean()))
        windows[case][f"{lo:g}-{hi:g}s"] = rows
report = dict(schema="flat-clock704-diagnostic-review-v1", recorded_at=datetime.now(timezone.utc).isoformat(),
    evaluation_sha256={k:sha(RUN / v / "evaluation.json") for k,v in names.items()},
    checkpoint_sha256=records["pd200"]["checkpoint_sha256"], checks=checks,
    matched_initial_inputs_and_requests=all(checks.values()), control_diff=control_diff, windows=windows,
    new_optimizer_updates=0, qualification_eligible=False, hardware_motion_started=False,
    scope="paired native feedback-clock intervention under the same RMCS command math; no hardware claim")
(HERE / "clock704_diagnostic_review001.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(checks=checks, forward3=windows["forward_3"], backward3=windows["backward_3"])))
