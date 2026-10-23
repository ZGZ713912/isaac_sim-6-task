"""Equal-index randomized static confirmation of migrated constant alpha.32."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
REPORTS=Path(__file__).parent
RUNS=ROOT/"runs/unified_robust_20261022"
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts"),str(REPORTS)]
from iteration045_runner import native,write

CHECKPOINT=RUNS/"quiet_whole_wheel087/model_migrated.pt"
NAME="constant_wheel_random087"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plan():
    migration=json.loads((REPORTS/"iteration087_migration.json").read_text())
    assert migration["alpha"]==.32 and migration["checkpoint_sha256"]==sha(CHECKPOINT)
    return dict(schema="quiet087-random-confirmation-plan-v1",checkpoint_sha256=sha(CHECKPOINT),
        migration_sha256=sha(REPORTS/"iteration087_migration.json"),alpha=.32,
        seed=211322,replicas=16,environments=48,seconds=12,domain_scale=.35,friction=[.3,.5],
        cases=["stand_low","stand","stand_high"],policy_hz=200,physics_hz=1000,
        original_domain_pair="quiet_whole_static074/robust",actual_vibration_threshold="quiet_velocity_baseline036",
        protections_and_weights_changed=False,native_origin_mode="isolated_overlap_v1",
        new_macro_updates=0,robust_consumed=1384,selected_lineage=505,full_training_admitted=False,
        sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/"scripts/evaluate_robust_getup_batch.py",
            ROOT/"src/wheeled_algo/quiet_output_actor.py",ROOT/"src/wheeled_algo/robust_getup.py",
            ROOT/"src/wheeled_tasks/chassis/robust_getup_env.py"]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args();value=plan();path=REPORTS/"iteration087_plan.json"
    if args.prepare:
        if path.exists():assert json.loads(path.read_text())==value
        else:write(path,value)
        print(json.dumps(value),flush=True);return
    assert json.loads(path.read_text())==value
    assert json.loads((REPORTS/"iteration086_pipeline.json").read_text())["status"]=="completed"
    subprocess.run(["git","diff","--exit-code","HEAD","--",*value["sources"],
        str(path.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT))],cwd=ROOT,check=True)
    pipeline=dict(status="running",phase="random_static",started=time.time(),new_macro_updates=0)
    write(REPORTS/"iteration087_pipeline.json",pipeline)
    try:
        output=RUNS/NAME
        native(NAME,[sys.executable,"scripts/evaluate_robust_getup_batch.py","--checkpoint",str(CHECKPOINT),
            "--output",str(output),"--variants","robust","--replicas","16","--seed","211322",
            "--domain-scale",".35","--friction-min",".3","--seconds","12","--isolated-origins",
            "--case-names","stand_low","stand","stand_high"],output)
        with (REPORTS/(NAME+"_analysis.log")).open("w") as log:
            subprocess.run([sys.executable,"scripts/analyze_robust_getup.py",str(output/"robust"),"--tensorboard"],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        from iteration036_compare import read,paired,standing
        from iteration048_runner import static_failures
        old,after=read("quiet_whole_static074"),read(NAME)
        rows=standing(after); failures=static_failures(rows,standing(read("quiet_velocity_baseline036")))
        result=dict(schema="quiet087-random-result-v1",status="completed",initial_pairing=paired(old,after),
            before=standing(old),after=rows,static_failures=failures,static_pass=not failures,
            trajectory_sha256=after[1]["trajectory_sha256"],analysis_sha256=sha(after[0]/"analysis.json"),
            checkpoint_sha256=sha(CHECKPOINT),new_macro_updates=0,robust_consumed=1384,
            full_training_admitted=False,scope="One paired random seed; fresh admission and combined recovery candidate still separate")
        write(REPORTS/"iteration087_final.json",result)
        print("087 RESULT",json.dumps(result),flush=True)
        pipeline.update(status="completed",phase="result_sealed",finished=time.time())
    except BaseException:
        pipeline.update(status="failed_no_retry",error=traceback.format_exc(),finished=time.time());raise
    finally:write(REPORTS/"iteration087_pipeline.json",pipeline)


if __name__=="__main__":main()
