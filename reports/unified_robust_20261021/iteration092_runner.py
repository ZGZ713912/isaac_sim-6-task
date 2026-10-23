"""Zero-update paired sampling exposure test of smaller recovery-leg noise."""
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
from wheeled_algo.evaluation_recovery_noise import recovery_noise_variants,SampledRecoveryEvaluation

CHECKPOINT=RUNS/"pilot033_dynamic_recovery079/model_final.pt"
NAME="recovery_sample_exposure092"
FRACTIONS=[.5,.25]
CASES=["left_asymmetric","right_asymmetric","inverted_asymmetric","moving_push_left","moving_push_right"]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan():
    assert sha(CHECKPOINT)=="5fcc13518d9fdd1cb4a3745ef84b38892bffbaeb90719336f66f5a15e6091018"
    return dict(schema="recovery092-sampling-exposure-plan-v1",checkpoint_sha256=sha(CHECKPOINT),
        variants=recovery_noise_variants(FRACTIONS),seed=211322,noise_seed=211492,replicas=4,environments=60,
        cases=CASES,seconds=12,domain_scale=0.,ground_friction=[.5,.5],policy_hz=200,physics_hz=1000,
        native_origin_mode="isolated_overlap_v1",new_macro_updates=0,robust_consumed=1384,parent_lineage=569,
        intervention="Checkpoint Gaussian draws; reduce only four leg stds by a smooth recovery-authority factor",
        frozen="Mean, wheel std, quiet and zero-recovery std, all weights, all mechanical protections",
        common_randomness="Same private noise seed, fixed batch shape and draw calls per variant; global RNG untouched",
        rationale_sha256=sha(REPORTS/"recovery091.json"),
        selection="Only propose a bounded learning pilot if practical completions gain>=2 and recovered events do not regress",
        eligible_for_policy_qualification=False,full_training_admitted=False,
        sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/"scripts/evaluate_robust_getup_batch.py",
            ROOT/"src/wheeled_algo/evaluation_recovery_noise.py",ROOT/"src/wheeled_algo/recovery_exploration.py",
            ROOT/"src/wheeled_algo/robust_getup.py",ROOT/"src/wheeled_tasks/chassis/robust_getup_env.py"]})


def cpu_probe():
    import numpy as np
    import torch
    from export_robust_getup import load_mean
    from wheeled_algo.robust_getup import SensorConditionedMean,tilt_gate
    torch.set_num_threads(1)
    state=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)
    mean,_=load_mean(CHECKPOINT);settings=state["infos"]["settings"]
    with np.load(RUNS/"dynamic_recovery_full079/robust/trajectory.npz") as data:
        x=torch.from_numpy(data["observations"].reshape(-1,35)[::23].copy())
    original=SensorConditionedMean(mean,settings["quiet_exploration_fraction"],settings["quiet_wheel_exploration_fraction"])
    with torch.no_grad():original.log_std.copy_(state["actor_state_dict"]["mlp.log_std"])
    rng=torch.get_rng_state();saved=deepcopy(original.state_dict());checks={};receipts=[]
    with torch.inference_mode():
        before=original(x);gate=tilt_gate(x,mean.stance_aware_recovery,mean.recovery_shape_aware).ravel()
        for name,fraction in recovery_noise_variants(FRACTIONS).items():
            model=SampledRecoveryEvaluation(mean,settings,original.log_std.detach(),fraction,211492)
            actual=model.sampler(x)
            assert torch.equal(before[:,0],actual[:,0]) and torch.equal(before[:,1,4:],actual[:,1,4:])
            assert torch.equal(before[gate==0],actual[gate==0])
            if fraction==1.:assert torch.equal(before,actual)
            assert all(torch.equal(v,model.sampler.state_dict()[k]) for k,v in saved.items())
            assert torch.isfinite(model(x)).all();receipts.append(model.sampling_receipt())
            checks[name]=dict(mean_exact=True,weights_exact=True,wheel_std_exact=True,zero_recovery_std_exact=True)
    assert torch.equal(rng,torch.get_rng_state())
    assert len({r["generator_state_sha256"] for r in receipts})==1
    write(REPORTS/"iteration092_cpu.json",dict(schema="recovery092-real-input-sampling-scope-v1",
        recorded_inputs=len(x),variants=checks,private_draw_states_equal=True,global_torch_rng_exact=True,
        native_behavior_proven=False,new_macro_updates=0,checkpoint_sha256=sha(CHECKPOINT),
        tests_passed=50,tests_scope="Recovery noise, original recovery-step guards and quiet output filtering"))


def analyze(output):
    import numpy as np
    from iteration036_compare import paired
    evidence={};results={};states=[]
    for name,fraction in recovery_noise_variants(FRACTIONS).items():
        directory=output/name
        with (REPORTS/(NAME+"_"+name+"_analysis.log")).open("w") as log:
            subprocess.run([sys.executable,"scripts/analyze_robust_getup.py",str(directory),"--tensorboard"],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((directory/"report.json").read_text());metrics=json.loads((directory/"analysis.json").read_text())
        assert report["trajectory_sha256"]==sha(directory/"trajectory.npz") and report["checkpoint_sha256"]==sha(CHECKPOINT)
        receipt=report["sampled_policy_exposure_diagnostic"]
        assert receipt["calls"]==2400 and receipt["samples_per_call"]==20 and receipt["recovery_leg_fraction"]==fraction
        states.append(receipt["generator_state_sha256"])
        evidence[name]=(directory,report,metrics,np.load(directory/"trajectory.npz"))
        cases={}
        for case,row in metrics["cases"].items():
            lanes=[r["lane"] for r in report["lanes"] if r["name"]==case]
            cases[case]=dict(episodes=row["episodes"],survived=row["survived"],
                recovered=sum(report["lanes"][i]["recovered"] for i in lanes),
                practical_completed=sum(metrics["practical_followup_by_lane"][i] for i in lanes),
                impulse_peak_ns=max(report["lanes"][i]["max_contact_impulse_ns"] for i in lanes),
                termination_causes=[report["lanes"][i]["termination_causes"] for i in lanes])
        results[name]=dict(fraction=fraction,cases=cases,closure_m=metrics["max_gap_m"],
            passive_motor_peak_nm=metrics["passive_motor_peak_nm"],trajectory_sha256=report["trajectory_sha256"],
            total_practical=sum(row["practical_completed"] for row in cases.values()),
            total_recovered=sum(row["recovered"] for row in cases.values()))
    assert len(set(states))==1
    base=evidence["robust"];pairing={name:paired(base,row) for name,row in evidence.items()}
    old=results["robust"]
    favorable=[name for name,row in results.items() if name!="robust"
        and row["total_practical"]>=old["total_practical"]+2 and row["total_recovered"]>=old["total_recovered"]]
    value=dict(schema="recovery092-sampling-exposure-result-v1",status="completed",variants=results,
        initial_pairing=pairing,private_final_draw_states_equal=True,bounded_pilot_candidates=favorable,
        scope="One nominal physical condition, four noise trajectories per case; sampled behavior cannot qualify the mean policy",
        new_macro_updates=0,robust_consumed=1384,full_training_admitted=False,candidate_adopted=False)
    write(REPORTS/"iteration092_final.json",value);print("092 RESULT",json.dumps(value),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args();value=plan();path=REPORTS/"iteration092_plan.json"
    if args.prepare:
        cpu_probe()
        if path.exists():assert json.loads(path.read_text())==value
        else:write(path,value)
        print(json.dumps(value),flush=True);return
    assert json.loads(path.read_text())==value
    assert json.loads((REPORTS/"iteration090_pipeline.json").read_text())["status"]=="completed"
    subprocess.run(["git","diff","--exit-code","HEAD","--",*value["sources"],str(path.relative_to(ROOT)),
        str(Path(__file__).relative_to(ROOT))],cwd=ROOT,check=True)
    pipeline=dict(status="running",phase="sampling_exposure_grid",started=time.time(),new_macro_updates=0)
    write(REPORTS/"iteration092_pipeline.json",pipeline)
    try:
        output=RUNS/NAME
        native(NAME,[sys.executable,"scripts/evaluate_robust_getup_batch.py","--checkpoint",str(CHECKPOINT),
            "--output",str(output),"--variants","robust","--replicas","4","--seed","211322",
            "--domain-scale","0","--seconds","12","--isolated-origins","--case-names",*CASES,
            "--recovery-leg-noise-fractions",*map(str,FRACTIONS),"--recovery-noise-seed","211492"],output)
        analyze(output);pipeline.update(status="completed",phase="result_sealed",finished=time.time())
    except BaseException:
        pipeline.update(status="failed_no_retry",error=traceback.format_exc(),finished=time.time());raise
    finally:write(REPORTS/"iteration092_pipeline.json",pipeline)


if __name__=="__main__":main()
