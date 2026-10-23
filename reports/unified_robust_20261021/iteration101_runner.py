"""Same-batch no-PPO dynamic roll authority counterfactual."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from iteration045_runner import native, write

NAME = "early_roll_push101"
CHECKPOINT = RUNS / "offline_recovery096/model_migrated.pt"
VARIANTS = ["robust", "early_roll20_to_45", "early_roll20_to_25"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan():
    from early_roll075_prototype import EarlyRollPrototype
    from wheeled_algo.evaluation_early_roll import EarlyRollEvaluation
    assert inspect.getsource(EarlyRollPrototype.forward) == inspect.getsource(EarlyRollEvaluation.forward)
    return dict(schema="roll101-native-grid-plan-v1", checkpoint_sha256=sha(CHECKPOINT),
        parent="096:079 learned64PPO plus64offline; old077 used065 before these recovery gains",
        parent_ppo_lineage=569,parent_offline_steps=64,previous_negative_checkpoint="065 in077",
        cpu_scope_sha256=sha(REPORTS / "iteration101_cpu.json"),
        hypothesis="Learned096 recovery now has useful left/capture actions; earlier lateral authority may reduce right preimpact velocity",
        variants=VARIANTS, start_degrees=20., full_degrees=[45., 25.], cases=["moving_push_left", "moving_push_right"],
        replicas=8, environments=48, seed=211322, seconds=12, domain_scale=0., ground_friction=.5,
        force_n=80., torque_nm=32., duration_s=.3, push_start_s=5., policy_hz=200, physics_hz=1000,
        origin="isolated_overlap_v1", source_prototype_forward_exact=True,
        protection_thresholds_changed=False, learned_weights_changed=False,
        sources={str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT / "scripts/evaluate_robust_getup_batch.py", ROOT / "src/wheeled_algo/evaluation_early_roll.py",
            ROOT / "src/wheeled_algo/robust_getup.py", ROOT / "src/wheeled_tasks/chassis/robust_getup_env.py")},
        new_macro_updates=0, robust_consumed=1384, selected_lineage=505, full_training_admitted=False,
        interpretation="Nominal causal diagnostic; repeated identical nominal cases are not randomized coverage; distinguish prevention from completed post-fall recovery")


def analyze(output):
    import numpy as np
    from iteration036_compare import paired
    evidence, results = {}, {}
    for variant in VARIANTS:
        directory = output / variant
        with (REPORTS / (NAME + "_" + variant + "_analysis.log")).open("w") as log:
            subprocess.run([sys.executable, "scripts/analyze_robust_getup.py", str(directory), "--tensorboard"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((directory / "report.json").read_text())
        metrics = json.loads((directory / "analysis.json").read_text())
        assert report["checkpoint_sha256"] == sha(CHECKPOINT) and report["trajectory_sha256"] == sha(directory / "trajectory.npz")
        evidence[variant] = (directory, report, metrics, np.load(directory / "trajectory.npz"))
        summary = {}
        for case in ("moving_push_left", "moving_push_right"):
            rows = [row for row in report["lanes"] if row["name"] == case]
            summary[case] = dict(episodes=len(rows), contact_impulse_peak_ns=max(row["max_contact_impulse_ns"] for row in rows), survived=sum(row["survived"] for row in rows),
                actual_falls=sum(row["in_motion_fall"] for row in rows),
                recovered_after_push=sum(row["recovered_after_push"] for row in rows),
                practical_complete_after_fall=sum(row["in_motion_fall"] and row["recovered_after_push"] and metrics["practical_followup_by_lane"][row["lane"]]
                                                  for row in rows),
                terminations=[dict(lane=row["lane"], time=row["termination_time_s"], causes=row["termination_causes"])
                              for row in rows if not row["survived"]])
        results[variant] = dict(summary=summary, max_closure_m=metrics["max_gap_m"],
            passive_motor_peak_nm=metrics["passive_motor_peak_nm"],
            trajectory_sha256=report["trajectory_sha256"],
            analysis_sha256=sha(directory / "analysis.json"))
    baseline = evidence["robust"]
    initial_pairing = {name: paired(baseline, item) for name, item in evidence.items()}
    before_push = {name: {key: bool(np.array_equal(baseline[3][key][:1000], item[3][key][:1000]))
                         for key in ("observations", "actions", "motor_position", "motor_velocity", "local_position", "alive")}
                   for name, item in evidence.items()}
    assert all(all(row.values()) for row in before_push.values()), before_push
    result = dict(schema="roll101-native-counterfactual-v1", status="completed", variants=results,
        initial_pairing=initial_pairing, prepush1000frames_bitwise=before_push,
        attribution_supported=all(all(row.values()) for row in before_push.values()),
        new_macro_updates=0, robust_consumed=1384, full_training_admitted=False,
        scope="One nominal batch; no pose resets during push; protection limits unchanged")
    write(REPORTS / "iteration101_final.json", result)
    print("101 RESULT", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    plan_path = REPORTS / "iteration101_plan.json"
    plan = make_plan()
    if args.prepare:
        if plan_path.exists():
            assert json.loads(plan_path.read_text()) == plan
        else:
            write(plan_path, plan)
        print(json.dumps(plan), flush=True)
        return
    assert json.loads(plan_path.read_text()) == plan
    assert json.loads((REPORTS / "iteration100_pipeline.json").read_text())["status"] == "completed"
    subprocess.run(["git","diff","--exit-code","HEAD","--",*plan["sources"],str(Path(__file__).relative_to(ROOT)),str(plan_path.relative_to(ROOT))],cwd=ROOT,check=True)
    pipeline = dict(status="running", phase="dynamic_roll_counterfactual", started=time.time(), new_macro_updates=0)
    write(REPORTS / "iteration101_pipeline.json", pipeline)
    try:
        output = RUNS / NAME
        native(NAME, [sys.executable, "scripts/evaluate_robust_getup_batch.py", "--checkpoint", str(CHECKPOINT),
            "--output", str(output), "--variants", "robust", "--replicas", "8", "--seed", "211322",
            "--domain-scale", "0", "--seconds", "12", "--isolated-origins", "--case-names",
            "moving_push_left", "moving_push_right", "--early-roll-full-degrees", "45", "25"], output)
        analyze(output)
        pipeline.update(status="completed", phase="result_sealed", finished=time.time())
    except BaseException:
        pipeline.update(status="failed", error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        write(REPORTS / "iteration101_pipeline.json", pipeline)


if __name__ == "__main__":
    main()
