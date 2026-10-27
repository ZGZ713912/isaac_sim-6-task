#!/usr/bin/env python3
"""CPU-only one-frame sensor sensitivity, never a closed-loop qualification.

Replay common authenticated nominal states through several frozen fresh actors.
Only the current sensor argument is perturbed; previous actions remain the
recorded nominal history. No simulator, optimizer or candidate adoption occurs.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def sensor_variants(frame,indices,seed,*,encoder_bias,gyro_bias,noise_sigma,fixed_rpy):
    """Native sensor18 coordinates: gyro*.5, gravity, angles, speeds*.1.

    The caller supplies actual seeded draws from the resolved native domain;
    this function never invents a different noise/rotation interpretation.
    """
    import torch
    from wheeled_tasks.chassis.imu_mount import euler_matrix
    if frame.ndim!=3 or frame.shape[-1]!=35:
        raise ValueError("Need [policy time, lane, actual35D actor input]")
    count=frame.shape[1]
    if (encoder_bias.shape!=(count,4) or gyro_bias.shape!=(count,3)
            or noise_sigma.shape!=(count,18) or fixed_rpy.shape!=(count,3)):
        raise ValueError("Sensor draws must preserve the actual lane indexing")
    baseline=frame[indices].clone()
    bias=baseline.clone()
    bias[:,:,10:14]+=encoder_bias[None]
    imu=baseline.clone()
    inverse=euler_matrix(fixed_rpy).transpose(-1,-2)
    imu[:,:,4:7]=torch.einsum("nij,tnj->tni",inverse,baseline[:,:,4:7])
    imu[:,:,7:10]=torch.einsum("nij,tnj->tni",inverse,baseline[:,:,7:10])
    rng=torch.Generator(device="cpu").manual_seed(seed)
    white=torch.randn(baseline.shape[:2]+(18,),generator=rng)*noise_sigma[None]
    white[:,:,:3]+=gyro_bias[None]
    noisy=baseline.clone()
    noisy[:,:,4:22]+=white
    gravity=noisy[:,:,7:10]
    noisy[:,:,7:10]=gravity/gravity.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    delay=frame[(indices-1).clamp_min(0)].clone()
    # Commands and previous actions remain current, exactly as RobustDomain's
    # sensor-only delay. This is a fixed5ms functional sensitivity, not a new
    # claim that every native lane draws a delayed sensor.
    delayed=baseline.clone()
    delayed[:,:,4:22]=delay[:,:,4:22]
    return dict(baseline=baseline,encoder_bias_only=bias,imu_alignment_only=imu,
        observation_noise_only=noisy,policy_delay_5ms_only=delayed)


def summarize(delta):
    """Report target changes in motor units, not normalized neural outputs."""
    leg=delta[...,:4]*.25
    wheel=delta[...,4:]*10.
    return dict(leg_target_delta_rms_rad=float(np.sqrt(np.mean(leg**2))),
        leg_target_delta_peak_rad=float(np.abs(leg).max()),
        wheel_target_delta_rms_rad_s=float(np.sqrt(np.mean(wheel**2))),
        wheel_target_delta_peak_rad_s=float(np.abs(wheel).max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,action="append",required=True)
    parser.add_argument("--trace-run",type=Path,required=True)
    parser.add_argument("--study",type=Path,default=ROOT/"contracts/wheel_leg_randomization_study_v2.json")
    parser.add_argument("--seed",type=int,default=261401)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    from evaluate_fresh_full_suite import authenticate
    bundles=[authenticate(path.resolve()) for path in args.checkpoint]
    archive=bundles[0]["archive"]
    # Actor source bytes must be identical across runs. Config/state may differ
    # only as authenticated by each checkpoint; no learned tensors are copied.
    expected_names={name for b in bundles for name in b["infos"]["source_hashes"]
        if name.startswith("src/wheeled_algo/") or name.endswith("imu_mount.py") or name.endswith("robust_getup.py")}
    for name in expected_names:
        hashes={sha(b["archive"]/name) for b in bundles}
        if len(hashes)!=1:
            raise ValueError("Mixed actor/domain implementation: "+name)
    sys.path[:0]=[str(archive/"src"),str(archive/"scripts")]
    from rsl_rl.utils import resolve_class
    from rsl_rl.utils.logger import Logger
    from tensordict import TensorDict
    from wheeled_tasks.chassis.robust_getup import RobustDomain
    # The explicit new resolver is not present in older source snapshots.
    import importlib.util
    spec=importlib.util.spec_from_file_location("sensor_response_domain_resolver",ROOT/"src/wheeled_tasks/chassis/randomization_study.py")
    resolver=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resolver)
    resolved=resolver.resolve(read(args.study),"requested",1.)
    trace_root=args.trace_run.resolve()
    receipt=read(trace_root/"receipt.json")
    result=read(trace_root/"evaluation/result.json")
    trace_file=trace_root/"evaluation/feedback_trace.npz"
    physical_file=trace_root/"evaluation/native_trace.npz"
    if (receipt["status"]!="completed_unqualified" or receipt["new_PPO"]!=0 or receipt["optimizer_constructed"]
            or sha(trace_file)!=result["feedback_trace"]["sha256"] or sha(physical_file)!=result["trace_sha256"]):
        raise ValueError("Need authenticated unchanged native replay states")
    lanes=[i for i,c in enumerate(result["cases"]) if c["height_m"]==.305]
    if len(lanes)!=7:
        raise ValueError("Need all seven fixed middle-height cases, never select a successful lane")
    with np.load(trace_file) as z:
        if not z["valid"][:,lanes].all():
            raise ValueError("Replay requires the full recorded middle-height prefix")
        frame=torch.tensor(z["observation_policy"][:,lanes])
    indices=torch.arange(0,len(frame),4)
    # Resolve the exact real domain draw path, then force the diagnostic's seven
    # sensor probes on. This does not change any real environment or checkpoint.
    cfg=deepcopy(resolved["task"])
    cfg["nominal_fraction"]=0.
    domain=RobustDomain(7,"cpu",cfg,args.seed,.001)
    sensors=frame[0,:,4:22]
    domain.reset(torch.arange(7),sensors,tick=0)
    variants=sensor_variants(frame,indices,args.seed+1,encoder_bias=domain.bias[:,6:10],
        gyro_bias=domain.bias[:,:3],noise_sigma=domain.sigma,fixed_rpy=domain.imu_mount.fixed)
    torch.set_num_threads(2)
    rows=[]
    actual_arrays={name:value.numpy() for name,value in variants.items()}
    for file,b in zip(args.checkpoint,bundles,strict=True):
        agent=b["agent"]
        obs=TensorDict(dict(policy=torch.zeros(1,35),critic=torch.zeros(1,83)),batch_size=[1])
        cls,config=resolve_class(agent["actor"])
        if sha(inspect.getsourcefile(cls))!=sha(b["archive"]/b["actor_source"]):
            raise ValueError("Actor source differs from the frozen checkpoint")
        actor=cls(obs,agent["obs_groups"],"actor",6,**config).cpu().eval()
        actor.load_state_dict(b["state"]["actor_state_dict"],strict=True)
        inference=actor.as_onnx().eval()
        outputs={}
        with torch.inference_mode():
            for name,value in variants.items():
                data=value.reshape(-1,35)
                actual=actor(TensorDict(dict(policy=data),batch_size=[len(data)]))
                exported=inference(data)
                if not torch.allclose(actual,exported,rtol=2e-6,atol=2e-7):
                    raise ValueError("Frozen inference wrapper differs from actor")
                outputs[name]=actual.reshape(*value.shape[:2],6).numpy()
        name=file.parent.name
        for key,value in outputs.items():
            actual_arrays[name+"__"+key]=value
        summary={key:summarize(value-outputs["baseline"]) for key,value in outputs.items() if key!="baseline"}
        std=actor.obs_normalizer._std.flatten().detach().numpy()
        eps=actor.obs_normalizer.eps
        normalizer=dict(epsilon=eps,gravity_roll_std=float(std[8]),
            height_command_feature_std=float(std[3]),
            declared_low_height_feature_delta_normalized=float((.23-.305)*5/(std[3]+eps)),
            declared_high_height_feature_delta_normalized=float((.42-.305)*5/(std[3]+eps)))
        rows.append(dict(run=name,checkpoint=str(file.resolve()),checkpoint_sha256=sha(file),
            completed_updates=b["infos"]["completed_updates"],one_frame_sensitivity=summary,normalizer=normalizer))
    np.savez_compressed(args.output/"functional_response.npz",**actual_arrays)
    source=args.output/Path(__file__).name
    source.write_bytes(Path(__file__).read_bytes())
    report=dict(schema="scratch-one-frame-sensor-response-v1",observed_utc=datetime.now(timezone.utc).isoformat(),
        seed=args.seed,rows=rows,replay_trace=str(trace_file),replay_trace_sha256=sha(trace_file),
        physical_trace_sha256=sha(physical_file),trace_cases=lanes,sampled_policy_frames=indices.tolist(),
        functional_trace_sha256=sha(args.output/"functional_response.npz"),source_sha256=sha(source),
        input_study_sha256=sha(args.study),actual_draws=dict(
            policy_encoder_bias_rad=domain.bias[:,6:10].tolist(),gyro_bias_feature=domain.bias[:,:3].tolist(),
            noise_sigma_sensor18=domain.sigma.tolist(),imu_fixed_rpy_rad=domain.imu_mount.fixed.tolist()),
        new_PPO=0,new_native_steps=0,optimizer_constructed=False,closed_loop_evaluated=False,hardware_ready=False,
        scope="Common nominal replay inputs. Sensor-only one-frame functional sensitivity; recorded previous-action history stays unchanged. No PD encoder, mass/COM or friction intervention, and no robustness/causality qualification.")
    # Reuse the official logger for scalar comparison; no TensorBoard server.
    logger=Logger(log_dir=str(args.output),cfg=bundles[0]["agent"],env_cfg=bundles[0]["cfg"],num_envs=7,
        is_distributed=False,gpu_world_size=1,gpu_global_rank=0,device="cpu")
    logger.init_logging_writer()
    try:
        for row in rows:
            for perturbation,metrics in row["one_frame_sensitivity"].items():
                for key,value in metrics.items():
                    logger.writer.add_scalar("SensorFunctionalResponse/"+row["run"]+"/"+perturbation+"/"+key,value,row["completed_updates"])
            for key,value in row["normalizer"].items():
                logger.writer.add_scalar("SensorFunctionalResponse/"+row["run"]+"/normalizer/"+key,value,row["completed_updates"])
        logger.writer.flush()
    finally:
        logger.writer.close()
    report["RSL_events_sha256"]={p.name:sha(p) for p in args.output.glob("events.out.tfevents.*")}
    write(args.output/"result.json",report)
    print(json.dumps(dict(rows=rows,new_native_steps=0),ensure_ascii=False))


if __name__=="__main__":
    main()
