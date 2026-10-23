"""Bounded128 recovery continuation supported by079 nominal closed-loop gains."""
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

PARENT = RUNS / "pilot033_dynamic_recovery079/model_final.pt"
TRAIN = RUNS / "pilot034_dynamic_recovery083"
UPDATES = 128
SEED = 211432


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    import torch
    from export_robust_getup import load_mean
    torch.set_num_threads(1)
    _, info = load_mean(PARENT)
    previous = json.loads((REPORTS / "iteration079_final.json").read_text())
    assert info["completed_updates"] == 569 and previous["candidate_checkpoint_sha256"] == sha(PARENT)
    rejected = json.loads((REPORTS / "iteration084_final.json").read_text())
    assert rejected["status"] == "completed" and rejected["static_pass"] is False
    for case in ("moving_push_left", "inverted_asymmetric"):
        assert previous["targeted"]["reference"][case]["practical_completed"] == 0
        assert previous["targeted"]["robust"][case]["practical_completed"] == 16
    with tempfile.TemporaryDirectory(prefix="recovery083-plan-") as folder:
        expected = task_settings(PARENT, Path(folder) / "prepare", SEED)
    return dict(schema="recovery083-continuation-plan-v1", parent_sha256=sha(PARENT),
        evidence="079 matched nominal left-moving fall and inverted-asymmetric improve0/16 to16/16; ordinary84/84 retained. Right-moving fall and broad random recovery remain unresolved.",
        hypothesis="Additional bounded recovery exposure can extend the demonstrated learned gains without another gate, reward or action-coordinate change",
        expected_settings=expected, new_macro_updates_target=UPDATES,
        num_envs=128, rollout_steps=128, seed=SEED, actor_lr=1e-5, critic_lr=1e-3,
        trained="existing21894 recovery values plus critic", frozen="all other actor/std/named Adam",
        exposure_and_rewards_same_as079=True, new_native_episode_reset_declared=True,
        gyro_filter084_rejected=True, gyro_filter084_installed=False,
        prerequisite084_sha256=sha(REPORTS / "iteration084_final.json"),
        lineage_before=569, candidate_lineage_if_complete=697,
        robust_charged_before=1256, robust_charged_if_complete=1384,
        evaluation="140 random full suite then96 nominal before/after targeted push pair plus inverted-asymmetric",
        new_long_updates=0, full_training_admitted=False,
        sources={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / "scripts/train_robust_getup.py", ROOT / "scripts/export_robust_getup.py",
            ROOT / "scripts/evaluate_robust_getup_batch.py", ROOT / "src/wheeled_algo/robust_getup.py",
            ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py")})


def evaluate(name, targeted=False):
    output = RUNS / name
    variants = ["robust", "reference"] if targeted else ["robust"]
    command = [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(TRAIN / "model_final.pt"),
        "--output", str(output), "--variants", *variants, "--replicas", "16" if targeted else "4",
        "--seed", "211322", "--domain-scale", "0" if targeted else ".35", "--friction-min", ".5" if targeted else ".3",
        "--seconds", "12", "--isolated-origins"]
    if targeted:
        command += ["--reference-checkpoint", str(PARENT), "--case-names", "moving_push_left", "moving_push_right", "inverted_asymmetric"]
    native(name, command, output)
    for variant in variants:
        with (REPORTS / (name + "_" + variant + "_analysis.log")).open("w") as log:
            subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(output / variant), "--tensorboard"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def result():
    import numpy as np
    from iteration036_compare import read, paired, standing
    before, after = read("dynamic_recovery_full079"), read("dynamic_recovery_full083")
    targeted = {}
    for variant in ("reference", "robust"):
        directory = RUNS / "dynamic_recovery_target083" / variant
        report = json.loads((directory / "report.json").read_text())
        analysis = json.loads((directory / "analysis.json").read_text())
        assert report["trajectory_sha256"] == sha(directory / "trajectory.npz")
        targeted[variant] = (directory, report, analysis, np.load(directory / "trajectory.npz"))
    summary = {variant: {case: dict(episodes=row["episodes"], survived=row["survived"],
        practical_completed=sum(info[2]["practical_followup_by_lane"][lane["lane"]]
            for lane in info[1]["lanes"] if lane["name"] == case))
        for case, row in info[2]["cases"].items()} for variant, info in targeted.items()}
    value = dict(schema="recovery083-result-v1", status="completed", new_macro_updates=UPDATES,
        checkpoint_sha256=sha(TRAIN / "model_final.pt"), candidate_lineage=697, robust_consumed=1384,
        full_pairing=paired(before, after), before_recovery=before[2]["actual_settled_attitudes"],
        after_recovery=after[2]["actual_settled_attitudes"], full_standing=standing(after),
        motion={key: after[2][key] for key in ("motion_survived", "motion_episodes", "practical_moving_fall_recoveries")},
        targeted_pairing=paired(targeted["reference"], targeted["robust"]), targeted=summary,
        closure_m=after[2]["max_gap_m"], passive_motor_peak_nm=after[2]["passive_motor_peak_nm"],
        adoption_decided=False, full_training_admitted=False, new_long_updates=0)
    write(REPORTS / "iteration083_final.json", value)
    print("083 RESULT", json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    value = plan()
    path = REPORTS / "iteration083_plan.json"
    if args.prepare:
        if path.exists():
            assert json.loads(path.read_text()) == value
        else:
            write(path, value)
        print(json.dumps({key: row for key, row in value.items() if key != "expected_settings"}), flush=True)
        return
    assert json.loads(path.read_text()) == value
    assert json.loads((REPORTS / "iteration082_pipeline.json").read_text())["status"] == "completed"
    assert json.loads((REPORTS / "iteration084_pipeline.json").read_text())["status"] == "completed"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tracked = [*value["sources"], str(path.relative_to(ROOT)), str(Path(__file__).relative_to(ROOT))]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *tracked], cwd=ROOT, check=True)
    archive = REPORTS / ("iteration083_source_" + commit[:12] + ".tar.gz")
    if archive.exists():
        raise ValueError("Refuse duplicate iteration083 launch archive")
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(archive), "HEAD", "src", "scripts", "contracts",
        str(REPORTS.relative_to(ROOT))], cwd=ROOT, check=True)
    write(REPORTS / "iteration083_launch.json", dict(source_commit=commit, source_archive=str(archive.relative_to(ROOT)),
        source_archive_sha256=sha(archive), plan_sha256=sha(path), parent_sha256=sha(PARENT),
        new_macro_updates_limit=UPDATES, launch_time=time.time(), hardware_motion=False, kaiser_changed=False))
    pipeline = dict(status="running", phase="bounded_training", started=time.time(), new_macro_updates=0,
                    robust_charged_before=1256, full_training_admitted=False)
    write(REPORTS / "iteration083_pipeline.json", pipeline)
    try:
        checked_training("iteration083_train", training_arguments(PARENT, TRAIN, UPDATES, SEED), TRAIN,
                         PARENT, value["expected_settings"], UPDATES)
        pipeline.update(phase="full_evaluation", new_macro_updates=UPDATES, robust_consumed=1384)
        write(REPORTS / "iteration083_pipeline.json", pipeline)
        evaluate("dynamic_recovery_full083")
        pipeline["phase"] = "targeted_before_after"
        write(REPORTS / "iteration083_pipeline.json", pipeline)
        evaluate("dynamic_recovery_target083", targeted=True)
        result()
        pipeline.update(status="completed", phase="result_sealed", finished=time.time())
    except BaseException:
        pipeline.update(status="failed_no_retry", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(REPORTS / "iteration083_pipeline.json", pipeline)


if __name__ == "__main__":
    main()
