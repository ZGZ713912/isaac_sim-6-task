"""Compose retained079 recovery with110 standing after user's high10cm change."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration044_selective_memory import equal
from iteration045_runner import write
from wheeled_algo.robust_getup import quiet_gate, tilt_gate
from wheeled_tasks.chassis.quiet_standing import standing_request_mask


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    recovery = RUNS / "pilot033_dynamic_recovery079/model_final.pt"
    quiet = RUNS / "quiet_position_gain110/model_migrated.pt"
    assert sha(recovery) == "5fcc13518d9fdd1cb4a3745ef84b38892bffbaeb90719336f66f5a15e6091018"
    assert sha(quiet) == "c38a376b240c5e04a282e24a8357b9dfb68f092b2aa77b0552c11b69ad12bf2a"
    assert json.loads((REPORTS / "iteration112_pipeline.json").read_text())["status"] == "completed"
    negative = json.loads((REPORTS / "iteration112_final.json").read_text())
    assert negative["after"]["practical_moving_fall_recoveries"] == 0
    assert json.loads((REPORTS / "iteration114_decision.json").read_text())["status"] == "withdrawn_before_native"
    old = torch.load(recovery, map_location="cpu", weights_only=False)
    q = torch.load(quiet, map_location="cpu", weights_only=False)
    state = deepcopy(old)
    keys = ["mlp.mean.quiet_high.2.bias", "mlp.mean.quiet_velocity.low.weight", "mlp.mean.quiet_velocity.high.weight"]
    for key in keys:
        state["actor_state_dict"][key] = q["actor_state_dict"][key].clone()
    changed = [k for k in old["actor_state_dict"] if not equal(old["actor_state_dict"][k], state["actor_state_dict"][k])]
    assert set(changed) == set(keys)
    for key in ("critic_state_dict", "optimizer_state_dict"):
        assert equal(old[key], state[key])
    for key in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"):
        assert equal(old["infos"][key], state["infos"][key])
    task = state["infos"]["settings"]
    assert task["standing_position_filter_seconds"] == .15
    task.update(version="robust-v2.52-composed115", quiet_output_wheel_alpha=.32)
    revision = dict(schema="composed115-state-v1", recovery_parent_sha256=sha(recovery), quiet_parent_sha256=sha(quiet),
        change="110 quiet high bias, low/high position weights and wheel outputalpha.32; retain079 recovery, not096/098",
        reason="112 full randomized096 loses sole moving recovery and right completion; user now permits high10cm with recentering",
        protocol_sha256=sha(REPORTS / "acceptance_protocol004.json"), new_macro_updates=0, new_offline_steps=0,
        candidate_ppo_lineage=569, candidate_offline_steps=0, robust_consumed=1384, actual_experiment_offline_steps=128,
        candidate_adopted=False, full_training_admitted=False, hardware_ready=False)
    state["infos"].update(control_revision=revision, migrated_from_sha256=sha(recovery), hardware_ready=False)
    output = RUNS / "composed_getup115"
    output.mkdir(exist_ok=False)
    shutil.copytree(recovery.parent / "source", output / "source")
    sources = set(old["infos"]["source_sha256"]) | set(q["infos"]["source_sha256"]) | {str(Path(__file__).relative_to(ROOT))}
    for relative in sorted(sources):
        path, target = ROOT / relative, output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        state["infos"]["source_sha256"][relative] = sha(target)
    contract = json.loads((recovery.parent / "task_contract.json").read_text())
    contract.update(settings=task, control_revision=revision)
    write(output / "task_contract.json", contract)
    state["infos"]["task_contract_sha256"] = sha(output / "task_contract.json")
    checkpoint = output / "model_migrated.pt"
    torch.save(state, checkpoint)
    merged, _ = load_mean(checkpoint)
    quiet_mean, _ = load_mean(quiet)
    recovery_mean, _ = load_mean(recovery)
    source = RUNS / "position_gain_tuning110/robust/trajectory.npz"
    with np.load(source) as data:
        observations = torch.from_numpy(data["observations"].reshape(-1, 35).copy())
    with torch.inference_mode():
        assert torch.count_nonzero(tilt_gate(observations, merged.stance_aware_recovery, merged.recovery_shape_aware)) == 0
        for x in observations.split(2048):
            assert torch.equal(merged(x), quiet_mean(x))
        source2 = RUNS / "dynamic_recovery_full079/robust/trajectory.npz"
        with np.load(source2) as data:
            x = torch.from_numpy(data["observations"].reshape(-1, 35)[::311].copy())
        commands = [(v, w) for v in (-2.5, -.75, -.12, 0., .12, .75, 1., 2., 3.)
                    for w in (-2., -1., -.25, 0., .25, 1., 2.) if abs(v) >= .12 or abs(w) >= .25]
        moving_count = 0
        for v, w in commands:
            for height in (.23, .305, .42):
                z = x.clone()
                z[:, 0], z[:, 2], z[:, 3] = v, w, height * 5.
                z[:, 4:7], z[:, 7:10], z[:, 14:16] = 0., torch.tensor([0., 0., -1.]), 0.
                altered = z.clone()
                altered[:, 14:16] = torch.tensor([5., -5.])
                false = torch.zeros(len(z), dtype=torch.bool)
                assert quiet_gate(z).count_nonzero() == 0
                assert standing_request_mask(torch.tensor([[v, w, height]]).expand(len(z), -1), false, false, false).count_nonzero() == 0
                expected = recovery_mean(z)
                assert torch.equal(merged(z), expected) and torch.equal(merged(altered), expected)
                moving_count += len(z)
        with np.load(source2) as data:
            fallen = torch.from_numpy(data["observations"].reshape(-1, 35)[::23].copy())
        nonquiet = quiet_gate(fallen).ravel() == 0.
        assert torch.equal(merged(fallen[nonquiet]), recovery_mean(fallen[nonquiet]))
    result = dict(revision, checkpoint_sha256=sha(checkpoint), strict_load_valid=True,
        changed_actor_relative079=changed, other_actor_std_critic_Adam_fourRNG_exact079=True,
        quiet110_recorded_inputs=len(observations), quiet110_outputs_bitwise_exact=True,
        actual_nonquiet079_inputs=int(nonquiet.sum()), actual_nonquiet079_outputs_bitwise_exact=True,
        locomotion_scope_inputs=moving_count, locomotion_commands=len(commands),
        locomotion_actions_and_position_perturbations_bitwise079=True, moving_standing_reward_mask_zero=True,
        source_static_trajectory_sha256=sha(source), source_dynamic_trajectory_sha256=sha(source2),
        native_composed_behavior_verified=False)
    write(REPORTS / "iteration115_migration.json", result)
    writer = SummaryWriter(str(output), filename_suffix=".migration115")
    writer.add_scalar("Diagnosis/Composed115/motion_action_peak_difference_raw", 0., 115)
    writer.add_scalar("Diagnosis/Composed115/quiet_action_peak_difference_raw", 0., 115)
    writer.add_scalar("Diagnosis/Composed115/locomotion_scope_inputs", moving_count, 115)
    writer.close()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
