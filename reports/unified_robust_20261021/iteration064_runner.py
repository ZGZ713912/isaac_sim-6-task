"""Bounded low-LR joint quiet adaptation after the leg-position ablation failed."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT/"runs/unified_robust_20261022"
PARENT = RUNS/"quiet_position_reference062/model_migrated.pt"
TRAIN = RUNS/"pilot032_joint_quiet064"
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts"),str(REPORTS)]
from iteration045_runner import native, write
from iteration036_compare import read, paired, standing
from iteration048_runner import static_failures
from iteration064_guard import preflight, checked_native


def evaluation(name,replicas,static):
    output = RUNS/name
    command = [sys.executable,"scripts/evaluate_robust_getup_batch.py","--checkpoint",str(TRAIN/"model_final.pt"),
        "--output",str(output),"--variants","robust","--replicas",str(replicas),"--seed","211322",
        "--domain-scale",".35","--friction-min",".3","--seconds","12","--isolated-origins"]
    if static:
        command += ["--case-names","stand_low","stand","stand_high"]
    native(name,command,output)
    with (REPORTS/(name+"_analysis.log")).open("w") as log:
        subprocess.run([sys.executable,"scripts/analyze_robust_getup.py",str(output/"robust"),"--tensorboard"],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    assert not json.loads((REPORTS/"iteration063_final.json").read_text())["static_target_passed"]
    assert all(row["full_window_within_5cm"] == 16 for row in json.loads((REPORTS/"iteration062_final.json").read_text())["after"].values())
    plan = json.loads((REPORTS/"iteration064_plan.json").read_text())
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == plan["parent_sha256"]
    expected_settings = preflight(PARENT)
    pipeline = dict(schema="position064-pipeline-v1",status="running",phase="training",started=time.time(),
        long_training_admitted=False)
    write(REPORTS/"iteration064_pipeline.json",pipeline)
    try:
        command = [sys.executable,"scripts/train_robust_getup.py","--output",str(TRAIN),"--resume",str(PARENT),
            "--mode","train","--updates","32","--num-envs","128","--rollout-steps","128",
            "--seed","211329","--actor-lr",".00001","--domain-scale",".35","--friction-min",".3",
            "--isolated-origins","--quiet-only","--standing-support-training",
            "--early-standing-position","--standing-drift-objective","--eval-seconds","2",
            "--quiet-exploration-fraction",".06","--quiet-wheel-exploration-fraction",".35"]
        progress = checked_native("iteration064_train",command,TRAIN,expected_settings)
        assert progress["new_optimizer_updates"] == 32
        assert progress["optimizer_learning_rates"] == dict(actor=1e-5,critic=1e-3)
        subprocess.run([sys.executable,str(REPORTS/"iteration064_native_audit.py")],cwd=ROOT,check=True)
        pipeline["phase"] = "extended_static"
        write(REPORTS/"iteration064_pipeline.json",pipeline)
        evaluation("joint_quiet_static064",16,True)
        before,after = read("quiet_velocity_baseline036"),read("joint_quiet_static064")
        rows = standing(after)
        failures = static_failures(rows,standing(before))
        assert not after[1]["evaluation_overrides"]
        result = dict(schema="position064-static-v1",pairing=paired(before,after),
            parent_pairing=paired(read("position_reference_static062"),after),before=standing(before),after=rows,
            static_failures=failures,new_macro_updates=32,robust_consumed=1160,lineage_clock=505,
            checkpoint_sha256=after[1]["checkpoint_sha256"],trajectory_sha256=after[1]["trajectory_sha256"],
            evaluation_overrides={},full_suite_complete=False,long_training_admitted=False,installed_in_rmcs=False)
        if not failures:
            pipeline["phase"] = "full_randomized_suite"
            write(REPORTS/"iteration064_pipeline.json",pipeline)
            evaluation("joint_quiet_full064",4,False)
            full = read("joint_quiet_full064")
            result.update(full_suite_complete=True,full_suite_pairing=paired(read("side_confirmation032"),full),full_analysis=full[2])
        result["status"] = "static_target_not_met" if failures else "requires_fresh_formal_admission"
        write(REPORTS/"iteration064_final.json",result)
        pipeline.update(status="completed",phase=result["status"],finished=time.time())
        write(REPORTS/"iteration064_pipeline.json",pipeline)
        print(json.dumps(result),flush=True)
    except BaseException:
        pipeline.update(status="failed",error=traceback.format_exc(),finished=time.time())
        write(REPORTS/"iteration064_pipeline.json",pipeline)
        raise


if __name__ == "__main__":
    main()
