"""Bounded position-feedback learning, only after native zero-function parity."""
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
PARENT = RUNS/"quiet_position056_sealed/model_migrated.pt"
TRAIN = RUNS/"pilot030_standing_position057"
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts"),str(REPORTS)]
from iteration045_runner import native, write
from iteration036_compare import read, paired, standing
from iteration048_runner import static_failures


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
    assert json.loads((REPORTS/"iteration056_pipeline.json").read_text())["status"] == "completed"
    assert json.loads((REPORTS/"iteration056_final.json").read_text())["native_zero_parity"]
    plan = json.loads((REPORTS/"iteration057_plan.json").read_text())
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == plan["parent_sha256"]
    pipeline = dict(schema="position057-pipeline-v1",status="running",phase="training",started=time.time(),
        long_training_admitted=False)
    write(REPORTS/"iteration057_pipeline.json",pipeline)
    try:
        command = [sys.executable,"scripts/train_robust_getup.py","--output",str(TRAIN),"--resume",str(PARENT),
            "--mode","train","--updates","32","--num-envs","128","--rollout-steps","128",
            "--seed","211329","--actor-lr",".001","--domain-scale",".35","--friction-min",".3",
            "--isolated-origins","--quiet-only","--quiet-feedback-only","--standing-support-training",
            "--early-standing-position","--standing-drift-objective","--eval-seconds","2"]
        progress = native("iteration057_train",command,TRAIN)
        assert progress["new_optimizer_updates"] == 32 and progress["trainable_actor_parameters"] == 24
        subprocess.run([sys.executable,str(REPORTS/"iteration057_native_audit.py")],cwd=ROOT,check=True)
        pipeline["phase"] = "extended_static"
        write(REPORTS/"iteration057_pipeline.json",pipeline)
        evaluation("learned_position_static057",16,True)
        before,after = read("quiet_velocity_baseline036"),read("learned_position_static057")
        rows = standing(after)
        failures = static_failures(rows,standing(before))
        assert not after[1]["evaluation_overrides"]
        result = dict(schema="position057-static-v1",pairing=paired(before,after),
            zero_abi_pairing=paired(read("position_zero_static056"),after),before=standing(before),after=rows,
            static_failures=failures,new_macro_updates=32,robust_consumed=1121,lineage_clock=473,
            checkpoint_sha256=after[1]["checkpoint_sha256"],trajectory_sha256=after[1]["trajectory_sha256"],
            evaluation_overrides={},full_suite_complete=False,long_training_admitted=False,installed_in_rmcs=False)
        if not failures:
            pipeline["phase"] = "full_randomized_suite"
            write(REPORTS/"iteration057_pipeline.json",pipeline)
            evaluation("learned_position_full057",4,False)
            full = read("learned_position_full057")
            result.update(full_suite_complete=True,full_suite_pairing=paired(read("side_confirmation032"),full),full_analysis=full[2])
        result["status"] = "static_target_not_met" if failures else "requires_fresh_formal_admission"
        write(REPORTS/"iteration057_final.json",result)
        pipeline.update(status="completed",phase=result["status"],finished=time.time())
        write(REPORTS/"iteration057_pipeline.json",pipeline)
        print(json.dumps(result),flush=True)
    except BaseException:
        pipeline.update(status="failed",error=traceback.format_exc(),finished=time.time())
        write(REPORTS/"iteration057_pipeline.json",pipeline)
        raise


if __name__ == "__main__":
    main()
