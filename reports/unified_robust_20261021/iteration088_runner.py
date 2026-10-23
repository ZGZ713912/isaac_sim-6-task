"""Zero-PPO nominal test of smaller recovery common-angle increments.

Run after087 exits. Original079, knee/wheel equations and mechanical guards
remain fixed; a favorable nominal result still needs random confirmation.
"""
import argparse
from copy import deepcopy
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
from wheeled_algo.evaluation_recovery_step import RecoveryStepEvaluation,recovery_step_variants

CHECKPOINT=RUNS/"pilot033_dynamic_recovery079/model_final.pt"
NAME="recovery_common_step088"
STEPS=[.2,.15]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plan():
    rejected=json.loads((REPORTS/"iteration083_decision.json").read_text())
    assert rejected["status"]=="rejected"
    return dict(schema="recovery088-nominal-step-plan-v1",checkpoint_sha256=sha(CHECKPOINT),
        parent_lineage=569,variants=recovery_step_variants(STEPS),seed=211322,replicas=4,environments=36,
        cases=["moving_push_left","moving_push_right","inverted_asymmetric"],seconds=12,
        domain_scale=0.,ground_friction=[.5,.5],policy_hz=200,physics_hz=1000,
        native_origin_mode="isolated_overlap_v1",new_macro_updates=0,robust_consumed=1384,
        reason="083 extra exposure lost079 left recovery;079 nominal right collides after full recovery authority with fast leg rotation",
        hypothesis="Smaller learned common-angle increments reduce collision without changing knee or wheel equations",
        maximum_common_increment_rad=[.075,.05,.0375],weights_and_protections_changed=False,
        guards=["equal initial frames","moving prepush1000frames exact","original baseline .3",
            "report complete12s and final2s practical motion","impulse20Ns and closure3mm unchanged"],
        selection="Nominal only; do not adopt or launch PPO from an isolated survival/impulse gain",
        full_training_admitted=False,
        cpu_scope_sha256=sha(REPORTS/"recovery_step088_full_cpu.json"),
        sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/"scripts/evaluate_robust_getup_batch.py",
            ROOT/"src/wheeled_algo/evaluation_recovery_step.py",ROOT/"src/wheeled_algo/robust_getup.py",
            ROOT/"src/wheeled_tasks/chassis/robust_getup_env.py"]})


def cpu_probe():
    import numpy as np
    import torch
    from export_robust_getup import load_mean
    torch.set_num_threads(1)
    mean,_=load_mean(CHECKPOINT)
    source=RUNS/"dynamic_recovery_full079/robust/trajectory.npz"
    with np.load(source) as data:x=torch.from_numpy(data["observations"].reshape(-1,35)[::23].copy())
    rng=torch.get_rng_state(); checks={}
    with torch.inference_mode():
        for step in [.3,*STEPS]:
            expected=deepcopy(mean);expected.recovery_leg_step=step
            actual=RecoveryStepEvaluation(mean,step)(x)
            assert torch.equal(actual,expected(x))
            checks[str(step)]=True
    assert mean.recovery_leg_step==.3 and torch.equal(rng,torch.get_rng_state())
    write(REPORTS/"iteration088_cpu.json",dict(schema="recovery088-installed-scope-v1",frames=len(x),
        installed_wrapper_exact_to_prototype=checks,original_unchanged=True,rng_unchanged=True,
        checkpoint_sha256=sha(CHECKPOINT),native_behavior_proven=False,new_macro_updates=0,
        helper_sha256=sha(ROOT/"src/wheeled_algo/evaluation_recovery_step.py")))


def analyze(output):
    import numpy as np
    from iteration036_compare import paired
    evidence,results={},{}
    for name,step in recovery_step_variants(STEPS).items():
        directory=output/name
        with (REPORTS/(NAME+"_"+name+"_analysis.log")).open("w") as log:
            subprocess.run([sys.executable,"scripts/analyze_robust_getup.py",str(directory),"--tensorboard"],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((directory/"report.json").read_text())
        metrics=json.loads((directory/"analysis.json").read_text())
        assert report["trajectory_sha256"]==sha(directory/"trajectory.npz")
        assert report["checkpoint_sha256"]==sha(CHECKPOINT) and report["recovery_leg_step_override"]==step
        data=np.load(directory/"trajectory.npz")
        evidence[name]=(directory,report,metrics,data)
        cases={}
        for case,row in metrics["cases"].items():
            lanes=[r["lane"] for r in report["lanes"] if r["name"]==case]
            cases[case]=dict(episodes=row["episodes"],survived=row["survived"],
                recovered=sum(report["lanes"][i]["recovered"] for i in lanes),
                practical_completed=sum(metrics["practical_followup_by_lane"][i] for i in lanes),
                contact_impulse_peak_ns=max(report["lanes"][i]["max_contact_impulse_ns"] for i in lanes),
                termination_causes=[report["lanes"][i]["termination_causes"] for i in lanes])
        results[name]=dict(step=step,cases=cases,closure_m=metrics["max_gap_m"],
            passive_motor_peak_nm=metrics["passive_motor_peak_nm"],trajectory_sha256=report["trajectory_sha256"])
    base=evidence["robust"]
    matching={name:paired(base,item) for name,item in evidence.items()}
    moving=[r["lane"] for r in base[1]["lanes"] if r["name"].startswith("moving_push_")]
    prepush={name:{key:np.array_equal(base[3][key][:1000,moving],item[3][key][:1000,moving])
        for key in ["observations","actions","motor_position","motor_velocity","local_position","alive"]}
        for name,item in evidence.items()}
    assert all(all(row.values()) for row in prepush.values()),prepush
    value=dict(schema="recovery088-nominal-result-v1",status="completed",variants=results,
        initial_pairing=matching,moving_prepush1000frames_exact=prepush,
        scope="One nominal condition, four deterministic copies; no random coverage or candidate qualification",
        new_macro_updates=0,robust_consumed=1384,full_training_admitted=False,candidate_adopted=False)
    write(REPORTS/"iteration088_final.json",value)
    print("088 RESULT",json.dumps(value),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args();value=plan();path=REPORTS/"iteration088_plan.json"
    if args.prepare:
        cpu_probe()
        if path.exists():assert json.loads(path.read_text())==value
        else:write(path,value)
        print(json.dumps(value),flush=True);return
    assert json.loads(path.read_text())==value
    assert json.loads((REPORTS/"iteration087_pipeline.json").read_text())["status"]=="completed"
    subprocess.run(["git","diff","--exit-code","HEAD","--",*value["sources"],
        str(path.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT))],cwd=ROOT,check=True)
    output=RUNS/NAME
    pipeline=dict(status="running",phase="nominal_step_grid",started=time.time(),new_macro_updates=0)
    write(REPORTS/"iteration088_pipeline.json",pipeline)
    try:
        native(NAME,[sys.executable,"scripts/evaluate_robust_getup_batch.py","--checkpoint",str(CHECKPOINT),
            "--output",str(output),"--variants","robust","--replicas","4","--seed","211322",
            "--domain-scale","0","--seconds","12","--isolated-origins","--case-names",*value["cases"],
            "--recovery-leg-steps",*map(str,STEPS)],output)
        analyze(output)
        pipeline.update(status="completed",phase="result_sealed",finished=time.time())
    except BaseException:
        pipeline.update(status="failed_no_retry",error=traceback.format_exc(),finished=time.time());raise
    finally:write(REPORTS/"iteration088_pipeline.json",pipeline)


if __name__=="__main__":main()
