#!/usr/bin/env python3
"""Read-only full capability evidence for an authenticated fresh six-motor actor.

The CPU preflight authenticates source/config/pose provenance. Native evaluation
uses that run's frozen physics/controller code. No PPO, Adam, teacher, action
substitution or scripted recovery is constructed in this process.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
from xml.etree import ElementTree

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
ROOT = Path(__file__).resolve().parents[1]

from wheel_leg_full_suite import build_cases, configure_protocol


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def authenticate(checkpoint):
    import torch
    run=checkpoint.parent
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    infos=state.get("infos",{})
    initialization=infos.get("initialization",{})
    if (infos.get("schema")!="scratch-standing-checkpoint-v1" or infos.get("training_origin")!="scratch"
            or initialization.get("checkpoint_loaded") is not False
            or initialization.get("teacher_loaded") is not False
            or infos.get("continuation_forbidden") is not True):
        raise ValueError("Need an authenticated fresh actor, never a transfer/teacher policy")
    archive=run/"source_snapshot"
    expected=infos["source_hashes"]
    if not expected or "scripts/train_scratch_standing.py" not in expected:
        raise ValueError("Missing frozen task source")
    for name,sha in expected.items():
        file=(archive/name).resolve()
        if not file.is_relative_to(archive.resolve()) or digest(file)!=sha:
            raise ValueError("Frozen source changed: "+name)
    agent_file=run/"agent_config.json"
    if digest(agent_file)!=infos["agent_config_sha256"]:
        raise ValueError("Actor configuration changed")
    agent=json.loads(agent_file.read_text())
    if agent["actor"]["class_name"]!=infos["actor_class"]:
        raise ValueError("Actor class ABI differs from its checkpoint")
    allowed={"MLPModel":None,
        "wheeled_algo.wheel_velocity_increment:WheelVelocityIncrementModel":"src/wheeled_algo/wheel_velocity_increment.py",
        "wheeled_algo.leg_velocity_separated:LegVelocitySeparatedModel":"src/wheeled_algo/leg_velocity_separated.py"}
    if agent["actor"]["class_name"] not in allowed:
        raise ValueError("Undeclared six-motor actor ABI")
    tensors=list(state.get("actor_state_dict",{}).values())
    if not tensors or any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("Actor state is missing or nonfinite")
    cfg_file=run/"config/contract.json"
    if digest(cfg_file)!=infos["mechanism_config_sha256"]:
        raise ValueError("Frozen mechanism configuration changed")
    cfg=json.loads(cfg_file.read_text())
    design_file=Path(cfg["design_preflight"]["file"])
    if digest(design_file)!=cfg["design_preflight"]["sha256"]:
        raise ValueError("Independent design changed")
    bank_file=run/"pose_bank.json"
    if not bank_file.exists():
        bank_file=Path(json.loads((run/"task_contract.json").read_text())["bank"])
    if digest(bank_file)!=infos["bank_sha256"]:
        raise ValueError("Validated reset bank changed")
    return dict(state=state,infos=infos,archive=archive,agent=agent,actor_source=allowed[agent["actor"]["class_name"]],
        cfg=cfg,design=json.loads(design_file.read_text()),bank_file=bank_file,bank=json.loads(bank_file.read_text()))


def prepare(args, bundle):
    cfg,design=deepcopy(bundle["cfg"]),deepcopy(bundle["design"])
    task=deepcopy(bundle["infos"]["settings"])
    spec=json.loads(args.spec.read_text())
    requirements_file=ROOT/spec["user_requirements"]
    requirements=json.loads(requirements_file.read_text())
    if requirements["schema"]!="wheel-leg-permanent-user-requirements-v2":
        raise ValueError("Full suite must preserve the current permanent requirements")
    cases=build_cases(bundle["bank"],spec,args.seed)
    expected={c["id"] for c in requirements["capabilities"]}
    if {c for case in cases for c in case["capabilities"]}!=expected:
        raise ValueError("Full suite dropped or renamed a permanent capability")
    design["scope"]="readonly_fresh_unified_actor_full_capability_evaluation"
    design["review_basis"]=dict(source_mechanism_sha256=digest(args.checkpoint.parent/"config/contract.json"),
        checkpoint_sha256=digest(args.checkpoint),spec_sha256=digest(args.spec),only_runtime_override={})
    domain=None
    if args.randomization_study is not None:
        from scratch_domain_evaluation import configure
        domain,_=configure(cfg,design,task,args.randomization_study,args.randomization_profile,1.)
        write_json(args.output/"randomization_study.resolved.json",domain["resolved"])
        (args.output/"randomization_study.input.json").write_bytes(args.randomization_study.read_bytes())
    overrides=configure_protocol(cfg,design,task,spec,case_count=len(cases),seed=args.seed)
    write_json(args.output/"config/design.json",design)
    cfg["design_preflight"]=dict(file=str(args.output/"config/design.json"),
        sha256=digest(args.output/"config/design.json"),stage="speed")
    write_json(args.output/"config/contract.json",cfg)
    write_json(args.output/"task_contract.json",task)
    write_json(args.output/"cases.json",cases)
    write_json(args.output/"spec.json",spec)
    write_json(args.output/"requirements.json",requirements)
    # The current process has not imported task classes. Import only the
    # archived pure design/control helpers during CPU preflight.
    sys.path[:0]=[str(bundle["archive"]/"src"),str(bundle["archive"]/"scripts")]
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.isolated_origins import validate_isolated_origins
    control=load_chassis_control(ROOT/cfg["control_math_source"])
    control_source=Path(inspect.getsourcefile(load_chassis_control)).resolve()
    if control_source!=bundle["archive"]/"src/wheeled_tasks/chassis/control_contract.py":
        raise ValueError("Preflight did not load the frozen control implementation")
    preflight=design_preflight(cfg,control,ROOT,stage="speed",role="train",seed=args.seed,num_envs=len(cases),block_index=0)
    validate_isolated_origins(cfg,len(cases))
    from wheeled_tasks.chassis.getup_ground import flat_patch
    # Exercise the frozen pure geometry factory in CPU preflight too: a design
    # comparison alone cannot prove a longer requested runway is constructible.
    points,indices=flat_patch(cfg["flat_half_length_m"],cfg["flat_floor_width_m"])
    write_json(args.output/"floor_geometry.json",dict(points=len(points),triangles=len(indices)//3,
        factory_source_sha256=digest(inspect.getsourcefile(flat_patch)),
        half_length_m=cfg["flat_half_length_m"],width_m=cfg["flat_floor_width_m"],
        factory_modified=False,native_simulation=False))
    write_json(args.output/"preflight.json",preflight)
    return cfg,task,spec,requirements,cases,domain,overrides


def native(args, bundle, prepared, receipt):
    import numpy as np
    import torch
    cfg,task,spec,requirements,cases,domain,overrides=prepared
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(dict(headless=True,device=args.device,enable_cameras=False,
        kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true"))
    env=writer=None
    try:
        from rsl_rl.utils import resolve_class
        from rsl_rl.utils.logger import Logger
        from wheeled_tasks.chassis.scratch_standing_env import ScratchStandingEnv
        from wheeled_tasks.chassis.robust_getup_env import RobustGetupEnv
        from wheeled_tasks.chassis.root_kinematics import update_robot_data
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_tasks.chassis.physical_standing_reward import install
        from record_scratch_physics import record as record_physics
        from wheel_leg_full_suite import analyze_case
        # Old nominal comparison runs predate this read-only observer. Load
        # its explicit frozen diagnostic copy, not an accidentally found task
        # module; do not substitute any archived dynamics/controller code.
        module_spec=importlib.util.spec_from_file_location("full_suite_domain_reader",args.output/"diagnostic_sources/randomization_study.py")
        domain_reader=importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(domain_reader)

        class FullSuiteEnv(ScratchStandingEnv):
            def __init__(self,*positional,**keywords):
                self.full_suite_cases=None
                super().__init__(*positional,**keywords)
                self.full_suite_cases=cases
                self.reset(torch.arange(self.num_envs,device=self.device))

            def reset(self,ids):
                if self.full_suite_cases is None:
                    return super().reset(ids)
                # Reuse the exact validated bank, passive fall, six-motor
                # action and native release logic. All writes here are reset
                # initialization; no reset or state write occurs in the loop.
                self.fixed_cases=self.full_suite_cases
                RobustGetupEnv.reset(self,ids)
                angle=torch.tensor([cases[i]["pitch_deg"] for i in ids.tolist()],device=self.device)*torch.pi/180.
                root=self.robot.data.root_link_pose_w.torch[ids].clone()
                x,y,z,w=root[:,3:].unbind(-1)
                s,c=(angle/2.).sin(),(angle/2.).cos()
                root[:,3:]=torch.stack((x*c-z*s,y*c+w*s,z*c+x*s,w*c-y*s),-1)
                self.robot.write_root_link_pose_to_sim_index(root_pose=root,env_ids=ids)
                update_robot_data(self,self.dt)
                self.previous_potential[ids]=self.potential()[ids]

        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        manifest_file=ROOT/cfg["asset_directory"]/"manifest.json"
        manifest=json.loads(manifest_file.read_text())
        if digest(manifest_file)!=cfg["asset_manifest_sha256"]:
            raise ValueError("Native asset manifest differs from the authenticated mechanism")
        asset_file=manifest_file.parent/manifest["urdf"]
        if digest(asset_file)!=manifest["files_sha256"][manifest["urdf"]]:
            raise ValueError("Wheel/hip reference URDF changed")
        robot_xml=ElementTree.parse(asset_file).getroot()
        hip_references=[]
        for pair in (("L_joint1","LL_joint1"),("R_joint1","RR_joint1")):
            joints=[robot_xml.find(f"./joint[@name='{name}']") for name in pair]
            if any(j is None or j.find("parent").attrib["link"]!="base_link" for j in joints):
                raise ValueError("Virtual leg reference must use the actual pair of body-fixed pivots")
            hip_references.append(np.mean([[float(v) for v in j.find("origin").attrib["xyz"].split()] for j in joints],axis=0).tolist())
        env=FullSuiteEnv(cfg,manifest,load_chassis_control(ROOT/cfg["control_math_source"]),ROOT,
            stage_name="speed",num_envs=len(cases),device=args.device,seed=args.seed,level=1.,run_role="train",
            bank=bundle["bank_file"],settings=task)
        receipt.update(physics_runtime=record_physics(env),actual_domain_samples=domain_reader.runtime_samples(env),
            grounded_reset=deepcopy(env.grounded_standing_reset),hip_reference_body_m=hip_references,
            hip_reference_definition="Midpoint of each side's body-fixed closed-chain motor pivots, read from the authenticated URDF")
        physics=receipt["physics_runtime"]
        if (env.dt!=.001 or env.policy_dt!=.005 or env.max_episode_length!=6000
                or physics["articulation_iterations"]!={"physxArticulation:solverPositionIterationCount":64,
                    "physxArticulation:solverVelocityIterationCount":32}):
            raise ValueError("Native controller/clock/solver/horizon differs from the declared protocol")
        if env.getup_cfg["push_start_seconds"]!=spec["push_at_s"]:
            raise ValueError("Native push start differs from the frozen evaluation specification")
        receipt["status"]="native_evaluating"
        write_json(args.output/"receipt.json",receipt)
        install(env,speed_weight=0.,contact_weight=0.,torque_weight=0.)
        actor_class,actor_cfg=resolve_class(bundle["agent"]["actor"])
        expected_source=bundle["actor_source"]
        if expected_source is not None:
            if Path(inspect.getsourcefile(actor_class)).resolve()!=bundle["archive"]/expected_source:
                raise ValueError("Actor class was not imported from its frozen source")
            if expected_source.endswith("leg_velocity_separated.py") and Path(inspect.getsourcefile(actor_class.__bases__[0])).resolve()!=bundle["archive"]/"src/wheeled_algo/wheel_velocity_increment.py":
                raise ValueError("Actor base dependency is not frozen")
        actor=actor_class(env.get_observations(),bundle["agent"]["obs_groups"],"actor",6,**actor_cfg).to(args.device)
        actor.load_state_dict(bundle["state"]["actor_state_dict"],strict=True)
        actor.eval()
        logger=Logger(log_dir=str(args.output),cfg=bundle["agent"],env_cfg=env.cfg,num_envs=env.num_envs,
            is_distributed=False,gpu_world_size=1,gpu_global_rank=0,device=args.device)
        logger.init_logging_writer()
        writer=logger.writer
        step=bundle["infos"]["completed_updates"]
        writer.add_scalar("FullSuiteProtocol/case_count",len(cases),step)
        writer.add_scalar("FullSuiteProtocol/seconds",spec["seconds"],step)
        if domain is not None:
            for tag,value in domain_reader.scalar_metrics(domain["resolved"]).items():
                writer.add_scalar("DomainStudyConfig/"+tag,value,step)
        initial=env.robot.data.root_link_pose_w.torch.clone()
        initial[:,:3]-=env.origins
        observation=env.get_observations()
        cutoff=torch.tensor([case["duration_seconds"] for case in cases],device=args.device)
        names=("valid","time","position","gravity","velocity","omega","requested","applied","actions",
            "motor_q","motor_dq","motor_torque","gap","force","clearance","leg_dq","leg_torque",
            "completion_event","enable_event","fallen","in_motion_fall","recovered_after_push","followup_success",
            "terminated","body_quaternion_world","wheel_center_body","wheel_center_world","wheel_pose_world","stable","pushing")
        traces={name:[] for name in names}
        with torch.inference_mode():
            for frame in range(6000):
                valid=~env.finished.clone() & (frame*.005<cutoff)
                active_before=env.recovery_active.clone()
                action=actor(observation)
                observation,_,_,info=env.step(action)
                d=info["diagnostics"]
                pose=env.robot.data.root_link_pose_w.torch
                relative=env.wheel_centers()-pose[:,None,:3]
                qinv=-pose[:,None,3:6].expand_as(relative)
                cross=torch.linalg.cross(qinv,relative)
                wheel_body=relative+2*(pose[:,None,6:]*cross+torch.linalg.cross(qinv,cross))
                completion=active_before & ~env.recovery_active & ~env.passive & valid
                contacts=env.standing_contact_monitor.last
                physical=env.physical_standing_last
                values=dict(valid=valid,time=d["time"],position=d["local_position"],gravity=d["gravity"],
                    velocity=d["velocity"],omega=d["omega"],requested=d["requested_command"],applied=d["commands"],
                    actions=d["actions"],motor_q=d["motor_position"],motor_dq=d["motor_velocity"],
                    motor_torque=d["motor_effort"],gap=d["gap"],force=contacts["force"],clearance=contacts["clearance"],
                    leg_dq=physical["dq"],leg_torque=physical["torque"],completion_event=completion,
                    enable_event=d["enable_event"],fallen=(d["gravity"][:,2]>-.7)|(d["height"]<.13),
                    in_motion_fall=d["in_motion_fall"],recovered_after_push=d["recovered_after_push"],
                    followup_success=d["followup_success"],terminated=d["terminated"],wheel_center_body=wheel_body,
                    body_quaternion_world=pose[:,3:],
                    wheel_center_world=env.wheel_centers(),
                    wheel_pose_world=env.robot.data.body_link_pose_w.torch[:,env.wheel_ids],stable=d["stable"],pushing=d["pushing"])
                for name,value in values.items():
                    traces[name].append(value.detach().cpu().clone())
                if (frame+1)%200==0:
                    receipt["native_policy_steps"]=frame+1
                    receipt["currently_unfinished_cases"]=int((~env.finished).sum())
                    write_json(args.output/"receipt.json",receipt)
        arrays={key:torch.stack(value).numpy() for key,value in traces.items()}
        # Derivatives of measured wheel-body position reveal actual leg/wheel
        # coordination; no controller or reference trajectory is synthesized.
        delta=arrays["wheel_center_body"]-np.asarray(hip_references)[None,None]
        arrays["virtual_leg_length_m"]=np.linalg.norm(delta[..., [0,2]],axis=-1)
        arrays["virtual_leg_angle_rad"]=np.arctan2(delta[...,0],-delta[...,2])
        arrays["wheel_center_fore_aft_velocity_body_m_s"]=np.gradient(arrays["wheel_center_body"][...,0],.005,axis=0)
        np.savez_compressed(args.output/"native_trace.npz",**arrays)
        rows=[analyze_case(case,arrays,lane,initial[lane].cpu().numpy(),spec,requirements) for lane,case in enumerate(cases)]
        groups=[]
        for capability in requirements["capabilities"]:
            selected=[r for r in rows if capability["id"] in r["case"]["capabilities"]]
            group=dict(capability=capability["id"],case_count=len(selected),
                engineering_pass_count=sum(r["engineering_pass"] for r in selected),
                full_horizon_count=sum(r["full_horizon"] for r in selected),
                native_guard_count=sum(r["native_guard_triggered"] for r in selected),
                robustness_across_domains_confirmed=False)
            groups.append(group)
            for key,value in group.items():
                if isinstance(value,(int,float)):
                    writer.add_scalar("FullSuite/"+capability["id"]+"/"+key,value,step)
            metric_paths=("closure_gap_peak_m","tilt_peak_deg","path/max_xy_excursion_m",
                "path/max_straight_lateral_m","tracking/raw_speed_mae_m_s","tracking/raw_yaw_mae_rad_s",
                "tracking/height_mae_m","smoothness/leg_speed_physical_peak_rad_s",
                "smoothness/leg_torque_slew_rms_nm_per_ms","smoothness/leg_angle_3_95hz_rms_mrad",
                "smoothness/leg_positive_power_mean_w","slip/estimated_contact_slip_rms_m_s",
                "coordination/wheel_fore_aft_range_max_m","coordination/wheel_fore_aft_velocity_rms_max_m_s",
                "path/max_heading_error_deg",
                "spin/actual_body_signed_rpm_mean","recovery/active_recovery_seconds",
                "braking/settling_seconds","braking/post_command_max_excursion_m")
            for path in metric_paths:
                values=[]
                for row in selected:
                    value=row
                    for key in path.split("/"):
                        value=value.get(key) if isinstance(value,dict) else None
                    if isinstance(value,(int,float)):
                        values.append(value)
                tag="FullSuite/"+capability["id"]+"/"+path
                writer.add_scalar(tag+"/measured_count",len(values),step)
                if values:
                    writer.add_scalar(tag+"/mean",sum(values)/len(values),step)
                    writer.add_scalar(tag+"/min",min(values),step)
                    writer.add_scalar(tag+"/max",max(values),step)
        result=dict(schema="fresh-full-suite-native-evaluation-v1",checkpoint_sha256=receipt["checkpoint_sha256"],
            completed_updates=step,cases=rows,capabilities=groups,native_trace_sha256=digest(args.output/"native_trace.npz"),
            overall_policy_qualified=False,hardware_ready=False,new_PPO=0,optimizer_constructed=False,
            note="Frozen engineering comparison only; robust integrated qualification needs the same checkpoint in nominal and requested domains plus independent seeds. Unquantified user limits are not hardware approvals.")
        write_json(args.output/"result.json",result)
        receipt.update(status="completed_unqualified",native_policy_steps=6000,result_sha256=digest(args.output/"result.json"))
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        if env is not None:
            env.close()
        launcher.app.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--spec",type=Path,default=ROOT/"contracts/wheel_leg_full_suite_v1.json")
    parser.add_argument("--seed",type=int,default=261301)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--randomization-study",type=Path)
    parser.add_argument("--randomization-profile")
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    if args.randomization_study is None and args.randomization_profile is not None:
        parser.error("A domain profile requires its explicit study JSON")
    for name in ("checkpoint","output","spec","randomization_study"):
        value=getattr(args,name)
        if value is not None:
            setattr(args,name,value.resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    receipt=dict(schema="fresh-full-suite-readonly-receipt-v1",status="preparing",pid=os.getpid(),
        start_ticks=int(Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19]),
        observed_utc=datetime.now(timezone.utc).isoformat(),checkpoint=str(args.checkpoint),
        checkpoint_sha256=digest(args.checkpoint),new_PPO=0,optimizer_constructed=False,hardware_ready=False,
        seed=args.seed,preflight_only=args.preflight_only,native_policy_steps=0,diagnostic_source_hashes={})
    try:
        diagnostic=args.output/"diagnostic_sources"
        diagnostic.mkdir()
        helpers={name:ROOT/"scripts"/name for name in ("evaluate_fresh_full_suite.py","wheel_leg_full_suite.py",
            "review_standing_acceptance.py","record_scratch_physics.py","scratch_domain_evaluation.py")}
        helpers["randomization_study.py"]=ROOT/"src/wheeled_tasks/chassis/randomization_study.py"
        for name,source in helpers.items():
            (diagnostic/name).write_bytes(source.read_bytes())
            receipt["diagnostic_source_hashes"][name]=digest(source)
        bundle=authenticate(args.checkpoint)
        receipt.update(completed_updates=bundle["infos"]["completed_updates"],
            frozen_source_hashes_verified=len(bundle["infos"]["source_hashes"]),
            training_origin_verified="fresh actor/critic/empty Adam, no teacher or checkpoint initialization")
        prepared=prepare(args,bundle)
        receipt.update(status="CPU_preflight_passed_unqualified",case_count=len(prepared[4]),
            protocol_overrides=prepared[6],domain_intervention=prepared[5],
            cases_sha256=digest(args.output/"cases.json"),spec_sha256=digest(args.output/"spec.json"),
            requirements_sha256=digest(args.output/"requirements.json"),preflight_sha256=digest(args.output/"preflight.json"))
        write_json(args.output/"receipt.json",receipt)
        if not args.preflight_only:
            native(args,bundle,prepared,receipt)
    except BaseException:
        receipt.update(status="failed",error=traceback.format_exc())
        raise
    finally:
        write_json(args.output/"receipt.json",receipt)
    print(json.dumps({k:receipt[k] for k in ("status","case_count","completed_updates","native_policy_steps","hardware_ready")},ensure_ascii=False))


if __name__=="__main__":
    main()
