"""Bounded recovery-only exposure pilot after077 disproved gate-only repair."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from iteration045_runner import native, write
from long_training070_plan import task_settings, training_arguments
from long_training070_runner import checked_training

PARENT = RUNS / "quiet_severe_reference065/model_migrated.pt"
TRAIN = RUNS / "pilot033_dynamic_recovery079"
UPDATES = 64
SEED = 211431


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    import torch
    from export_robust_getup import load_mean
    torch.set_num_threads(1)
    mean, info = load_mean(PARENT)
    with tempfile.TemporaryDirectory(prefix="recovery079-plan-") as folder:
        expected = task_settings(PARENT, Path(folder) / "prepare", SEED)
    assert expected["motion_fraction"] == .4 and expected["motion_push_fraction"] == 1.
    assert expected["recovery_family_weights"] == [0., .1, .1, .2, .2, .3, .1]
    assert not expected["side_recovery_birth_focus"] and not expected["quiet_training"]
    assert info["completed_updates"] == 505 and sum(p.numel() for p in mean.recovery.parameters()) == 21894
    return dict(schema="recovery079-bounded-exposure-plan-v1", parent_sha256=sha(PARENT),
        reason="077 matched dynamics disproved authority-only changes; prior032 recovery optimization focused80%fallen with70%side births, quiet adaptations since then never trained recovery",
        hypothesis="Broader ordinary-motion push plus inverted exposure can improve the existing recovery head without modifying stable motion/quiet feedback",
        rejected_gate_versions_not_installed=True, expected_settings=expected,
        new_macro_updates_target=UPDATES, num_envs=128, rollout_steps=128, seed=SEED,
        actor_lr=1e-5, critic_lr=1e-3, trainable_actor_values=21894,
        trained="existing recovery head only", frozen="all parent/arc/quiet/position/std parameters and named Adam",
        reset_mix="40%upright;60%fallen;front/back/left/right/inverted/compound=.1/.1/.2/.2/.3/.1",
        command_mix="stand30/straight20/spin10/arcs40 percent; original high-speed range retained",
        disturbances="Existing80N/32Nm/.3s at5s in100%upright lanes, both directions; no pose writes after reset",
        gate_action_reward_physics_changed=False, domain_scale=.35, ground_friction=[.3, .5],
        policy_hz=200, physics_hz=1000, origin="isolated_overlap_v1",
        lineage_before=505, candidate_lineage_if_complete=569, robust_charged_before=1192,
        robust_charged_if_complete=1256, long_training_admitted=False,
        evaluation="140 full randomized suite and96 same-batch nominal before/after targeted dynamic pushes plus inverted_asymmetric",
        scope="Bounded diagnostic64, not the conditional5120 run; no automatic extra PPO",
        revision="v2:078 unpaired domain result retained; new position diagnostic rejects random batching; ordinary train/full/nominal paths unchanged",
        sources={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / "scripts/train_robust_getup.py", ROOT / "scripts/export_robust_getup.py",
            ROOT / "scripts/evaluate_robust_getup_batch.py", ROOT / "src/wheeled_algo/robust_getup.py",
            ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py")})


def analyze(name, variants):
    output = RUNS / name
    for variant in variants:
        with (REPORTS / (name + "_" + variant + "_analysis.log")).open("w") as log:
            subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / variant), "--tensorboard"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def evaluate(name, targeted=False):
    output = RUNS / name
    command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(TRAIN / "model_final.pt"),
        "--output", str(output), "--variants", "robust", *( ["reference"] if targeted else []),
        "--replicas", "16" if targeted else "4", "--seed", "211322", "--domain-scale", "0" if targeted else ".35",
        "--friction-min", ".5" if targeted else ".3", "--seconds", "12", "--isolated-origins"]
    if targeted:
        command += ["--reference-checkpoint", str(PARENT), "--case-names", "moving_push_left", "moving_push_right", "inverted_asymmetric"]
    native(name, command, output)
    analyze(name, ["robust", "reference"] if targeted else ["robust"])


def result():
    import numpy as np
    from iteration036_compare import read, paired, standing
    full = read("dynamic_recovery_full079")
    before = read("high_recenter_full066")
    target = RUNS / "dynamic_recovery_target079"
    evidence = {}
    for variant in ("reference", "robust"):
        directory = target / variant
        report = json.loads((directory / "report.json").read_text())
        analysis = json.loads((directory / "analysis.json").read_text())
        assert report["trajectory_sha256"] == sha(directory / "trajectory.npz")
        evidence[variant] = (directory, report, analysis, np.load(directory / "trajectory.npz"))
    target_pair = paired(evidence["reference"], evidence["robust"])
    summary = {variant: {case: dict(episodes=row["episodes"], survived=row["survived"],
            practical_completed=sum(info[2]["practical_followup_by_lane"][lane["lane"]]
                                    for lane in info[1]["lanes"] if lane["name"] == case))
        for case, row in info[2]["cases"].items()} for variant, info in evidence.items()}
    value = dict(schema="recovery079-result-v1", status="completed", new_macro_updates=UPDATES,
        candidate_checkpoint_sha256=sha(TRAIN / "model_final.pt"), candidate_lineage=569, robust_consumed=1256,
        full_pairing=paired(before, full), before_recovery=before[2]["actual_settled_attitudes"],
        after_recovery=full[2]["actual_settled_attitudes"], full_standing=standing(full),
        motion=dict(survived=full[2]["motion_survived"], episodes=full[2]["motion_episodes"],
            practical_moving_fall_recoveries=full[2]["practical_moving_fall_recoveries"]),
        targeted_pairing=target_pair, targeted=summary,
        closure_m=full[2]["max_gap_m"], passive_motor_peak_nm=full[2]["passive_motor_peak_nm"],
        adoption_decided=False, full_training_admitted=False, long_new_updates=0)
    write(REPORTS / "iteration079_final.json", value)
    print("079 RESULT", json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    value = plan()
    path = REPORTS / "iteration079_plan_v2.json"
    if args.prepare:
        if path.exists():
            assert json.loads(path.read_text()) == value
        else:
            write(path, value)
        print(json.dumps({key: row for key, row in value.items() if key != "expected_settings"}), flush=True)
        return
    assert json.loads(path.read_text()) == value
    # This runner is launched explicitly after078 completes, never in parallel.
    finished078 = json.loads((REPORTS / "position_release_static078_process.json").read_text())
    assert finished078["status"] == "completed" and finished078["returncode"] == 0
    assert finished078["native_status"] == "completed"
    # Its matching audit failed and cannot qualify a policy.079 uses only065.
    assert json.loads((REPORTS / "iteration078_independent_result.json").read_text())["admissible_pair"] is False
    pipeline = dict(status="running", phase="bounded_training", started=time.time(),
        new_macro_updates=0, robust_charged_before=1192, long_training_admitted=False)
    write(REPORTS / "iteration079_pipeline.json", pipeline)
    try:
        checked_training("iteration079_train", training_arguments(PARENT, TRAIN, UPDATES, SEED), TRAIN,
                         PARENT, value["expected_settings"], UPDATES)
        pipeline.update(phase="full_evaluation", new_macro_updates=UPDATES, robust_consumed=1256)
        write(REPORTS / "iteration079_pipeline.json", pipeline)
        evaluate("dynamic_recovery_full079")
        pipeline["phase"] = "targeted_before_after"
        write(REPORTS / "iteration079_pipeline.json", pipeline)
        evaluate("dynamic_recovery_target079", targeted=True)
        result()
        pipeline.update(status="completed", phase="result_sealed", finished=time.time())
    except BaseException:
        pipeline.update(status="failed_no_retry", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(REPORTS / "iteration079_pipeline.json", pipeline)


if __name__ == "__main__":
    main()
