"""Bounded single-variable interventions in the existing quiet policy."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ["src", "scripts"]
import numpy as np
import torch
from export_robust_getup import load_mean
from wheeled_algo.robust_getup import quiet_gate

torch.set_num_threads(1)
base = Path("runs/robust_policy_repair_20261024")
source = base / "low_wheel_diagonal_restoring_probe001/model_probe.pt"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
original = torch.load(source, map_location="cpu", weights_only=False)
old, info = load_mean(source)
trace = base / "diagnosis013_low_diagonal_fixed/evaluation/trajectory.npz"
frames = np.load(trace)["observations"].reshape(-1, 35)
key = "mlp.mean.quiet_velocity.low.weight"

for name in ("quiet_high_prediction010_probe001", "low_diagonal_gain2_probe001"):
    output = base / name
    output.mkdir(exist_ok=False)
    state = deepcopy(original)
    if name.startswith("quiet_high"):
        assert state["infos"]["settings"]["quiet_high_leg_prediction_s"] == .005
        assert not state["infos"]["settings"].get("quiet_leg_prediction_all_heights", False)
        state["infos"]["settings"]["quiet_high_leg_prediction_s"] = .010
        hypothesis = "Test whether the existing causal 10ms leg-angle prediction prevents the onset of high-stance pitch oscillation. No new observer state or sensor."
        gate = dict(fixed_standing_at_least=20, development_standing_at_least=21,
                    all_high_cases_required=True, no_lost_previous_pass=True)
    else:
        state["actor_state_dict"][key][2:4] *= 2.
        hypothesis = "Double only the two diagonal restoring wheel-displacement gains to test braking of the first0.75s low-stance translation. Leg balance gains and every other expert are retained."
        gate = dict(fixed_standing_at_least=21, low_all7_required=True,
                    no_lost_previous_pass=True)
    state["infos"]["bounded_quiet_gain_probe"] = dict(
        source=str(source), source_sha256=sha(source), hypothesis=hypothesis,
        new_PPO=0, optimizer="Original slots retained; this is not a coordinate-equivalent migration")
    state["infos"]["hardware_ready"] = False
    checkpoint = output / "model_probe.pt"
    torch.save(state, checkpoint)
    new, _ = load_mean(checkpoint)
    checks = dict(unrelated_actor_tensors_exact=all(
        torch.equal(v, state["actor_state_dict"][k])
        for k, v in original["actor_state_dict"].items()
        if name.startswith("quiet_high") or k != key))
    changed = outside_changed = leg_changed = 0
    with torch.inference_mode():
        for obs in torch.tensor(frames).split(2048):
            a, b = old(obs), new(obs)
            difference = (a != b).any(-1)
            eligible = quiet_gate(obs).ravel() > 0.
            if name.startswith("quiet_high"):
                eligible &= obs[:, 3] / 5. > .35
            else:
                eligible &= obs[:, 3] / 5. < .305
                leg_changed += int((a[:, :4] != b[:, :4]).any(-1).sum())
            changed += int(difference.sum())
            outside_changed += int(difference[~eligible].sum())
    checks["outside_declared_expert_exact"] = outside_changed == 0
    if not name.startswith("quiet_high"):
        checks["all_leg_targets_exact"] = leg_changed == 0
        checks["common_leg_feedback_weights_exact"] = torch.equal(
            original["actor_state_dict"][key][:2], state["actor_state_dict"][key][:2])
    assert all(checks.values()), checks
    receipt = dict(schema="bounded-quiet-single-variable-probe-v1",
        checkpoint=str(checkpoint), sha256=sha(checkpoint), source_sha256=sha(source),
        checks=checks, frames=len(frames), changed_frames=changed,
        replay_sha256=sha(trace), hypothesis=hypothesis, gate=gate,
        motion_MAE_at_most=.135, new_PPO=0, experiment_consumed=2587,
        source_clock=info["completed_updates"], qualification=None,
        hardware_started=False, created_unix=time.time())
    checkpoint.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
