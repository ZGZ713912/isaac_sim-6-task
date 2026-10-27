#!/usr/bin/env python3
"""Serial read-only evaluation after one exact fresh-training process exits.

This never launches/restarts training. A stopped/failed/incomplete training
run, changed source or missing native artifact stops this queue. Native tests
retain all outcomes and never authorize long training or hardware motion.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
SOURCE_FILES=("scripts/queue_requested_domain_evaluation.py","scripts/evaluate_scratch_standing.py",
    "scripts/review_standing_acceptance.py","scripts/scratch_domain_evaluation.py",
    "scripts/capture_scratch_feedback.py","scripts/analyze_scratch_feedback_trace.py",
    "scripts/record_scratch_physics.py","scripts/wheel_encoder_response_probe.py",
    "scripts/final_pose_observation_probe.py","src/wheeled_tasks/chassis/randomization_study.py",
    "contracts/wheel_leg_randomization_study_v2.json","contracts/wheel_leg_acceptance_requirements_v2.json")
UNSELECTED_REQUIRED={"enabled","motor_strength","leg_kp_scale","leg_kd_scale","spring_strength",
    "policy_sensor_delay_ticks","policy_bias","policy_noise_sigma","pd_encoder_bias",
    "imu_fixed_rpy_rad","imu_wobble_amplitude_rad","imu_wobble_frequency_hz","imu_wobble_phase_rad",
    "rigid_enabled","mass_scale","independent_inertia_scale","base_com_offset_m",
    "ground_static_friction","ground_dynamic_friction"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_sources(output):
    # Check the entire real entrypoint dependency set before creating snapshots.
    # These helpers include files archived unconditionally by the native reader.
    sources={name:(ROOT/name).read_bytes() for name in SOURCE_FILES}
    for name,data in sources.items():
        destination=output/"source_snapshot"/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(data)
    return {name:hashlib.sha256(data).hexdigest() for name,data in sources.items()}


def write(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def start_ticks(pid):
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[19])
    except FileNotFoundError:
        return None


def terminal_checkpoint(run,identity,expected_updates):
    """A missing old process alone is not successful completion."""
    if start_ticks(identity["pid"])==identity["start_ticks"]:
        raise ValueError("Original training process is still live")
    progress=read(run/"progress.json")
    checkpoint=run/"model_final.pt"
    if (progress["pid"]!=identity["pid"] or progress["start_ticks"]!=identity["start_ticks"]
            or progress["status"]!="completed_unqualified" or progress["completed_updates"]!=expected_updates
            or progress["optimizer_attempts"]!=expected_updates or progress["optimizer_update_in_flight"]
            or progress["checkpoint_reads"]!=[] or not checkpoint.is_file()
            or Path(progress["last_checkpoint"]["path"]).resolve()!=checkpoint.resolve()
            or progress["last_checkpoint"]["sha256"]!=sha(checkpoint)):
        raise ValueError("Training did not finish and seal its complete fresh budget; never restart it here")
    if progress["evaluation"]["completed_updates"]!=expected_updates:
        raise ValueError("Native terminal evaluation is incomplete")
    return checkpoint,progress


def paired_samples(reference,other,profile):
    """Require actual samples, including rigid properties and contact material."""
    if set(reference)!=set(other) or not UNSELECTED_REQUIRED<=set(reference):
        raise ValueError("Missing actual sampled domains; profile names alone are insufficient")
    selected={"without_encoder_bias":{"policy_bias","pd_encoder_bias"},
        "without_policy_delay":{"policy_sensor_delay_ticks"},
        "without_alignment":{"imu_fixed_rpy_rad"}}[profile]
    checks={name:reference[name]==other[name] for name in reference if name not in selected}
    if profile=="without_encoder_bias":
        checks["policy_gyro_bias_exact"]=all(a[:6]==b[:6] and a[10:]==b[10:]
            for a,b in zip(reference["policy_bias"],other["policy_bias"],strict=True))
        checks["policy_encoder_bias_zero"]=all(v==0 for row in other["policy_bias"] for v in row[6:10])
        checks["PD_encoder_bias_zero"]=all(v==0 for row in other["pd_encoder_bias"] for v in row)
    elif profile=="without_policy_delay":
        checks["policy_delay_zero"]=all(v==0 for v in other["policy_sensor_delay_ticks"])
    else:
        checks["IMU_fixed_zero"]=all(v==0 for row in other["imu_fixed_rpy_rad"] for v in row)
    if not all(checks.values()):
        raise ValueError("Unpaired single-group comparison: "+str([k for k,v in checks.items() if not v]))
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run",type=Path,required=True)
    parser.add_argument("--training-pid",type=int,required=True)
    parser.add_argument("--training-start-ticks",type=int,required=True)
    parser.add_argument("--expected-updates",type=int,default=512)
    parser.add_argument("--seed",type=int,default=261201)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run=args.training_run.resolve()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    identity=dict(pid=args.training_pid,start_ticks=args.training_start_ticks)
    receipt=dict(schema="requested-domain-serial-readonly-queue-v1",status="starting",pid=os.getpid(),
        start_ticks=start_ticks(os.getpid()),training_run=str(run),training_identity=identity,
        seed=args.seed,expected_updates=args.expected_updates,profiles=[],active_child=None,
        new_PPO=0,optimizer_constructed=False,training_restart_allowed=False,
        overall_policy_qualified=False,hardware_ready=False,observed_utc=datetime.now(timezone.utc).isoformat())
    child=None
    stop=False
    def request_stop(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    def save():
        receipt["observed_utc"]=datetime.now(timezone.utc).isoformat()
        write(output/"receipt.json",receipt)
    def check_sources():
        if any(sha(ROOT/name)!=value for name,value in receipt["source_sha256"].items()):
            raise ValueError("Frozen queue/evaluator source changed; refuse automatic adoption")
    def execute(command,log,label):
        nonlocal child
        check_sources()
        with log.open("xb") as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            receipt["active_child"]=dict(pid=child.pid,start_ticks=start_ticks(child.pid),
                label=label,argv=command,log=str(log))
            save()
            while child.poll() is None:
                if stop:
                    expected=receipt["active_child"]
                    if start_ticks(child.pid)==expected["start_ticks"]:
                        child.terminate()
                    child.wait()
                    raise InterruptedError("Queue stopped at user/process request")
                time.sleep(1.)
            code=child.returncode
            receipt.setdefault("closed_children",[]).append(dict(receipt["active_child"],exit_code=code))
            receipt["active_child"]=None
            child=None
            save()
            if code:
                raise RuntimeError(f"Read-only child {label} exited{code}; inspect {log}")
    try:
        receipt["source_sha256"]=freeze_sources(output)
        if start_ticks(identity["pid"])!=identity["start_ticks"]:
            # Already-completed runs are allowed only after the same complete
            # terminal audit; never infer completion from a stale state file.
            terminal_checkpoint(run,identity,args.expected_updates)
        else:
            cmd=Path(f"/proc/{identity['pid']}/cmdline").read_bytes()
            if b"scripts/train_scratch_standing.py" not in cmd or str(run).encode() not in cmd:
                raise ValueError("Training identity belongs to another process")
        receipt["status"]="waiting_for_exact_training_process_exit"
        save()
        while start_ticks(identity["pid"])==identity["start_ticks"]:
            if stop:
                raise InterruptedError("Queue stopped without signaling its training dependency")
            p=read(run/"progress.json")
            receipt["observed_training_updates"]=p["completed_updates"]
            receipt["observed_training_status"]=p["status"]
            save()
            time.sleep(5.)
        checkpoint,progress=terminal_checkpoint(run,identity,args.expected_updates)
        gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True)
        if str(identity["pid"]) in gpu.splitlines():
            raise ValueError("Original training still appears in the GPU process list")
        receipt.update(status="serial_native_evaluation",checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint),training_terminal_status=progress["status"],training_original_process_and_GPU_exited=True)
        save()
        physics=None
        actual=None
        reviews={}
        profiles=["nominal","requested"]
        index=0
        while index<len(profiles):
            profile=profiles[index]
            folder=output/("eval_"+profile)
            command=[sys.executable,"-u",str(ROOT/"scripts/evaluate_scratch_standing.py"),
                "--checkpoint",str(checkpoint),"--output",str(folder),"--seed",str(args.seed),
                "--trace-feedback","--randomization-study",str(ROOT/"contracts/wheel_leg_randomization_study_v2.json"),
                "--randomization-profile",profile]
            execute(command,output/(profile+".native.log"),profile+"_native")
            native=read(folder/"receipt.json")
            if (native["status"]!="completed_unqualified" or native["checkpoint_sha256"]!=receipt["checkpoint_sha256"]
                    or native["new_PPO"]!=0 or native["optimizer_constructed"] or native["seed"]!=args.seed):
                raise ValueError("Native receipt changed policy, seed or evaluation scope")
            if physics is None:
                physics=native["physics_runtime"]
            elif native["physics_runtime"]!=physics:
                raise ValueError("Native physics settings differ between profiles")
            if profile=="requested":
                actual=native["evaluation"]["actual_domain_samples"]
            pairing=paired_samples(actual,native["evaluation"]["actual_domain_samples"],profile) if profile.startswith("without_") else None
            review_file=output/(profile+".review.json")
            execute([sys.executable,str(ROOT/"scripts/review_standing_acceptance.py"),str(folder),
                "--requirements",str(ROOT/"contracts/wheel_leg_acceptance_requirements_v2.json"),
                "--output",str(review_file)],output/(profile+".review.log"),profile+"_review")
            review=read(review_file)
            mid=next(g for g in review["groups"] if g["height_m"]==.305)
            reviews[profile]=mid
            receipt["profiles"].append(dict(profile=profile,run=str(folder),review=str(review_file),
                middle_height=mid,all_heights=review["groups"],paired_nonintervened_draw_checks=pairing,
                receipt_sha256=sha(folder/"receipt.json"),review_sha256=sha(review_file)))
            if profile=="requested" and (mid["standing_pass_v2_count"]<7 or mid["max_xy_excursion_m"]>.05):
                profiles.extend(["without_encoder_bias","without_policy_delay","without_alignment"])
            index+=1
            save()
        receipt.update(status="completed_unqualified",profiles_planned=profiles,
            note="Diagnostic standing comparison only. Low/high training coverage, held-out seeds/30s and all permanent capabilities remain required. No long PPO, ONNX export or robot motion is authorized by this queue's success.")
    except InterruptedError as error:
        receipt.update(status="stopped",error=str(error))
    except BaseException:
        receipt.update(status="failed",error=traceback.format_exc())
        raise
    finally:
        save()


if __name__=="__main__":
    main()
