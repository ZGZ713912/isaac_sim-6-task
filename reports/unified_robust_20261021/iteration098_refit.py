"""Bounded offline fit of successful sampled recovery targets, with retention.

This is application-target regression, not PPO or a policy-gradient estimator.
The existing PPO Adam state is preserved explicitly; the separate optimizer is
saved with its own step count. Native mean behavior is the only acceptance.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
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
from wheeled_algo.robust_getup import tilt_gate

PARENT = RUNS / "offline_recovery096/model_migrated.pt"
DATA_TEACHER = RUNS / "pilot033_dynamic_recovery079/model_final.pt"
OUTPUT = RUNS / "offline_recovery098"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    directory = RUNS / relative
    report = json.loads((directory / "report.json").read_text())
    assert sha(directory / "trajectory.npz") == report["trajectory_sha256"]
    return directory, report, np.load(directory / "trajectory.npz")


def prepare_data(mean):
    source, report, data = read("recovery_sample_exposure092/robust")
    analysis = json.loads((source / "analysis.json").read_text())
    candidates = [row for row in report["lanes"] if row["name"] == "moving_push_right"
                  and analysis["practical_followup_by_lane"][row["lane"]]]
    assert len(candidates) == 1
    lane = candidates[0]["lane"]
    obs = torch.from_numpy(data["observations"][:, lane].copy())
    with torch.inference_mode():
        gate = tilt_gate(obs, mean.stance_aware_recovery, mean.recovery_shape_aware).ravel().numpy()
    keep = (data["alive"][:, lane] & (gate >= .5) & data["in_motion_fall"][:, lane]
            & ~data["recovered_after_push"][:, lane])
    elite_x = obs[keep].clone()
    elite_y = torch.from_numpy(data["actions"][keep, lane][:, [0, 1, 3, 4, 2, 5]].copy())
    assert len(elite_x) >= 200 and torch.isfinite(elite_y).all()
    # Uniform coverage per old successful lane, not just ordinary upright data.
    retention, lineage = [], []
    for relative in ("dynamic_recovery_target079/robust", "dynamic_recovery_full079/robust"):
        directory, report, data = read(relative)
        analysis = json.loads((directory / "analysis.json").read_text())
        count = 0
        for row in report["lanes"]:
            lane = row["lane"]
            if not analysis["practical_followup_by_lane"][lane]:
                continue
            x = torch.from_numpy(data["observations"][:, lane].copy())
            with torch.inference_mode():
                gate = tilt_gate(x, mean.stance_aware_recovery, mean.recovery_shape_aware).ravel().numpy()
            ids = np.flatnonzero(data["alive"][:, lane] & (gate > .1) & ~data["passive"][:, lane])
            if len(ids):
                selected = ids[np.linspace(0, len(ids) - 1, min(64, len(ids))).astype(int)]
                retention.append(x[selected].clone())
                count += len(selected)
        lineage.append(dict(source=relative, trajectory_sha256=report["trajectory_sha256"], retained_frames=count))
    retain_x = torch.cat(retention)
    with torch.inference_mode():
        retain_y = mean(retain_x).clone()
    assert len(retain_x) > 1000
    lineage.insert(0, dict(source="recovery_sample_exposure092/robust", lane=candidates[0]["lane"],
        trajectory_sha256=sha(source / "trajectory.npz"), supervised_frames=len(elite_x),
        scope="Only postfall recovery-gate>=.5 frames before first recovered-after-push; no prepush targets"))
    return (elite_x, elite_y, retain_x, retain_y), lineage


def discrepancy(prediction, target):
    delta = prediction - target
    leg = delta[:, :4] * .25
    leg = torch.atan2(leg.sin(), leg.cos()) / .25
    return torch.cat((leg, delta[:, 4:]), -1)


def plan(lineage):
    return dict(schema="offline-recovery098-plan-v1", parent_sha256=sha(PARENT),
        method="Continue096 exactly64 offline steps with same316 targets, original079 teacher3883 frames and restored offline Adam; not PPO",
        data_teacher_sha256=sha(DATA_TEACHER), preceding_offline_steps=64, cumulative_offline_steps=128,
        learned_parameters="Only six recovery MLP tensors; all other actor/std/critic/PPO Adam/four saved RNG remain exact",
        steps=64, learning_rate=.00003, optimizer="restore096 separate offline Adam64->128; parent PPO Adam stays exact",
        retention_weight=4., gradient_norm_clip=1., seed=211496, retention_batch=512,
        elite_batch="all postfall successful right frames, uniformly weighted", sources=lineage,
        scope_checks=dict(max_retention_leg_target_change_rad=.05, max_retention_wheel_target_change_rad_s=3.,
                          nonrecovery_exact=True, strict_checkpoint_load=True),
        behavioral_next="Only if scope checks pass: nominal paired24 lanes, three cases x4 x2 models,12s; candidate must recover all12 practically",
        implementation_sha256={str(p.relative_to(ROOT)): sha(p) for p in (
            Path(__file__), ROOT / "scripts/evaluate_robust_getup_batch.py",
            ROOT / "src/wheeled_algo/robust_getup.py", ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py")},
        heldout_seeds=[211131, 211132], source092_retired_from_heldout=True,
        new_macro_updates=0, new_offline_steps=64, robust_consumed=1384,
        selected_ppo_lineage=505, candidate_ppo_lineage=569, full_training_admitted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert sha(PARENT) == "eb49ecdf871d5d2ca218586b75072c62f61771fc8431171f363c95d8bbb7a50f"
    mean, _ = load_mean(PARENT)
    teacher, _ = load_mean(DATA_TEACHER)
    samples, lineage = prepare_data(teacher)
    value = plan(lineage)
    plan_path = REPORTS / "iteration098_plan.json"
    if args.prepare:
        if plan_path.exists():
            assert json.loads(plan_path.read_text()) == value
        else:
            write(plan_path, value)
        print(json.dumps(dict(elite_frames=len(samples[0]), retention_frames=len(samples[2]), plan_sha256=sha(plan_path))), flush=True)
        return
    assert json.loads(plan_path.read_text()) == value
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *value["implementation_sha256"],
                    str(plan_path.relative_to(ROOT))], cwd=ROOT, check=True)
    assert not OUTPUT.exists()
    OUTPUT.mkdir()
    old = torch.load(PARENT, map_location="cpu", weights_only=False)
    state = deepcopy(old)
    elite_x, elite_y, retain_x, retain_y = samples
    mean.requires_grad_(False)
    mean.recovery.requires_grad_(True)
    parameters = list(mean.recovery.parameters())
    assert len(parameters) == 6 and sum(p.numel() for p in parameters) == 21894
    optimizer = torch.optim.Adam(parameters, lr=value["learning_rate"])
    assert old["infos"]["offline_refit_steps"] == 64
    optimizer.load_state_dict(old["infos"]["offline_recovery_optimizer"])
    assert all(int(item["step"]) == 64 for item in optimizer.state.values())
    generator = torch.Generator().manual_seed(value["seed"])
    for _ in range(value["preceding_offline_steps"]):
        torch.randint(len(retain_x), (value["retention_batch"],), generator=generator)
    writer = SummaryWriter(str(OUTPUT), filename_suffix=".offline098")
    log_std = old["actor_state_dict"]["mlp.log_std"].clamp(-3., -.22314355)

    def objective(x, target):
        gate = tilt_gate(x, mean.stance_aware_recovery, mean.recovery_shape_aware)
        scale = (log_std.exp() * (.12 + .88 * gate)).clamp(.003, .8)
        return (discrepancy(mean(x), target) / scale).square().mean()

    with torch.no_grad():
        initial_loss = float(objective(elite_x, elite_y))
    prior_audit = json.loads((REPORTS / "iteration096_state.json").read_text())
    assert abs(initial_loss - prior_audit["final_elite_loss"]) < 1e-7
    for step in range(value["steps"]):
        ids = torch.randint(len(retain_x), (value["retention_batch"],), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        elite_loss = objective(elite_x, elite_y)
        retention_loss = objective(retain_x[ids], retain_y[ids])
        loss = elite_loss + value["retention_weight"] * retention_loss
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, value["gradient_norm_clip"], error_if_nonfinite=True)
        optimizer.step()
        for key, scalar in dict(total_loss=loss, elite_loss=elite_loss,
                                retention_loss=retention_loss, gradient_norm=norm).items():
            writer.add_scalar("OfflineRecovery098/" + key, float(scalar.detach()), step + 1)
    with torch.no_grad():
        final_loss = float(objective(elite_x, elite_y))
        delta = discrepancy(mean(retain_x), retain_y)
        leg_peak = float(delta[:, :4].abs().max() * .25)
        wheel_peak = float(delta[:, 4:].abs().max() * 10.)
    for key, tensor in mean.state_dict().items():
        state["actor_state_dict"]["mlp.mean." + key] = tensor.detach().clone()
    changed = [k for k in old["actor_state_dict"] if not equal(old["actor_state_dict"][k], state["actor_state_dict"][k])]
    assert len(changed) == 6 and all(k.startswith("mlp.mean.recovery.") for k in changed)
    for key in ("critic_state_dict", "optimizer_state_dict"):
        assert equal(old[key], state[key])
    for key in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"):
        assert equal(old["infos"][key], state["infos"][key])
    state["infos"]["settings"]["version"] = "robust-v2.47-offline-success-recovery098"
    revision = dict(value, actual_offline_steps=value["steps"],
        optimizer_note="Parent PPO Adam is preserved, deliberately not described as continued PPO; offline Adam saved separately",
        source_plan_sha256=sha(plan_path), parent_ppo_clock=old["infos"]["completed_updates"])
    assert all(int(item["step"]) == 128 for item in optimizer.state.values())
    state["infos"].update(control_revision=revision, offline_recovery_optimizer=optimizer.state_dict(),
        offline_refit_steps=value["cumulative_offline_steps"], hardware_ready=False, migrated_from_sha256=sha(PARENT))
    shutil.copytree(PARENT.parent / "source", OUTPUT / "source")
    for relative in sorted(set(state["infos"]["source_sha256"]) | {str(Path(__file__).relative_to(ROOT))}):
        target = OUTPUT / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
        state["infos"]["source_sha256"][relative] = sha(target)
    contract = json.loads((PARENT.parent / "task_contract.json").read_text())
    contract.update(settings=state["infos"]["settings"], control_revision=revision)
    write(OUTPUT / "task_contract.json", contract)
    state["infos"]["task_contract_sha256"] = sha(OUTPUT / "task_contract.json")
    checkpoint = OUTPUT / "model_migrated.pt"
    torch.save(state, checkpoint)
    restored, _ = load_mean(checkpoint)
    baseline, _ = load_mean(PARENT)
    _, _, data = read("high_recenter_full066/robust")
    x = torch.from_numpy(data["observations"].reshape(-1, 35)[::23].copy())
    with torch.inference_mode():
        zero_gate = tilt_gate(x, baseline.stance_aware_recovery, baseline.recovery_shape_aware).ravel() == 0.
        nonrecovery_exact = torch.equal(restored(x[zero_gate]), baseline(x[zero_gate]))
    scope_pass = (nonrecovery_exact and final_loss < initial_loss and leg_peak <= .05 and wheel_peak <= 3.)
    result = dict(schema="offline-recovery098-state-v1", status="completed", checkpoint_sha256=sha(checkpoint),
        changed_actor=changed, frozen_actor_std_critic_PPOAdam_fourRNG_exact=True, strict_load_valid=True,
        elite_frames=len(elite_x), retention_frames=len(retain_x), initial_elite_loss=initial_loss,
        final_elite_loss=final_loss, retention_leg_target_change_peak_rad=leg_peak,
        retention_wheel_target_change_peak_rad_s=wheel_peak, nonrecovery_inputs=int(zero_gate.sum()),
        nonrecovery_actions_exact=nonrecovery_exact, scope_pass=scope_pass,
        new_macro_updates=0, actual_offline_steps=value["steps"], cumulative_offline_steps=128, robust_consumed=1384,
        candidate_ppo_lineage=569, full_training_admitted=False,
        caveat="Lower supervised loss cannot establish native recovery improvement")
    for key in ("initial_elite_loss", "final_elite_loss", "retention_leg_target_change_peak_rad", "retention_wheel_target_change_peak_rad_s"):
        writer.add_scalar("OfflineRecovery098/" + key, result[key], value["steps"])
    writer.close()
    write(REPORTS / "iteration098_state.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
