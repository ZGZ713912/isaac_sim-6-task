#!/usr/bin/env python3
"""One local, explicitly budgeted learner with serial native behavior reviews.

The frozen plan owns the update budget. Evaluation never runs alongside its
learner. Checkpoint selection uses measured behavior rather than reward/loss;
selection does not certify a policy for physical deployment.
"""
from __future__ import annotations

import argparse
import fcntl
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
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from train_robust_getup import digest,write_json


def read(path):
    return json.loads(Path(path).read_text())


def stages(total=2048,block=256):
    if not (256<=total<=4096 and 32<=block<=256 and total%block==0):
        raise ValueError("Require256..4096 updates in complete32..256-update blocks")
    return [dict(index=i+1,updates=block,offset=i*block,
        severity=.6 if i*block<512 else .8 if i*block<1024 else 1.,
        friction_min=.3 if i*block<1024 else .2) for i in range(total//block)]


def adaptation(history,scheduled_severity):
    """Reduce LR and hold the domain ramp after two measured regressions."""
    lr=1e-4
    regressions=0
    best=history[0]["score"]
    for row in history[1:]:
        if row["score"] < best-.05:
            regressions+=1
        else:
            regressions=0
        best=max(best,row["score"])
        if regressions>=2:
            lr=max(1e-5,lr*.5)
            regressions=0
    severity=scheduled_severity
    if len(history)>=3 and all(row["score"]<best-.05 for row in history[-2:]):
        severity=min(severity,history[-1].get("training_severity",.6))
    return dict(actor_lr=lr,severity=severity)


def behavior_metrics(evaluation,standing):
    """Fixed denominators and explicit failures; survival alone earns no score."""
    report=read(Path(evaluation)/"evaluation/report.json")
    summary=read(Path(evaluation)/"evaluation/gap_summary.json")
    quality=read(standing)
    parts={name:[r for r in summary["rows"] if r["behavior_case"].startswith(prefix)]
        for name,prefix in dict(entry="real_start",standing="stand",motion=("forward","backward","arc"),
            fallen=("left_symmetric","right_symmetric","inverted_symmetric")).items()}
    if ({k:len(v) for k,v in parts.items()}!={"entry":12,"standing":21,"motion":12,"fallen":6}
            or len({r["name"] for r in summary["rows"]})!=51 or quality["episodes"]!=21
            or not 0<=quality["pose_and_position_pass"]<=21):
        raise ValueError("Incomplete or duplicated diagnostic populations cannot rank checkpoints")
    result={}
    for name,rows in parts.items():
        lane_names={r["name"] for r in rows}
        result[name]=dict(episodes=len(rows),survived=sum(r["survived"] for r in rows),
            recovered=sum(r["recovered"] for r in rows),
            followup_success=sum(r["followup_success"] for r in report["lanes"] if r["name"] in lane_names),
            stable90=sum(r.get("tail_stable_fraction",0.)>=.9 for r in rows),
            max_leg_speed_rad_s=max((r.get("max_leg_speed_rad_s",0.) for r in rows),default=0.),
            max_excursion_m=max((r.get("max_position_excursion_m",0.) for r in rows),default=0.),
            vx_mae_m_s=sum(r.get("vx_mae_m_s",0.) for r in rows)/max(1,len(rows)),
            yaw_mae_rad_s=sum(r.get("yaw_mae_rad_s",0.) for r in rows)/max(1,len(rows)))
    result["standing"]["pose_and_position_pass"]=quality["pose_and_position_pass"]
    def ratio(name,key):
        return result[name][key]/max(1,result[name]["episodes"])
    result["score"]=(.35*ratio("entry","followup_success")+.2*ratio("fallen","followup_success")
        +.3*ratio("standing","pose_and_position_pass")+.1*ratio("motion","stable90")
        +.05*max(0.,1.-result["standing"]["max_leg_speed_rad_s"]/30.))
    result["formal_acceptance"]=False
    return result


def prepare(output,initial,baseline,total):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    initial,baseline=Path(initial).resolve(),Path(baseline).resolve()
    import torch
    from export_robust_getup import load_mean
    _,info=load_mean(initial)
    if info["settings"].get("gap_upright_target_limit_version")!="error-v1":
        raise ValueError("Long run must learn with the final target transform active")
    if info["settings"].get("gap_actuator_timing_profile")!="native":
        raise ValueError("Do not mix unidentified extra timing into this campaign")
    bank=ROOT/"reports/real_robot_robustness_20261024/pose_bank_gap002.json"
    if info["bank_sha256"]!=digest(bank):
        raise ValueError("Bank differs from the migrated checkpoint")
    sources=set((ROOT/"src").rglob("*.py"))
    sources.update(ROOT/"scripts"/name for name in ("run_gap_long_training.py","train_robust_getup.py",
        "export_robust_getup.py","evaluate_gap_robustness.py","analyze_gap_standing.py","compare_gap_robustness.py"))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(sources)}
    for relative in hashes:
        target=output/"source"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
    plan=dict(schema="wheel-leg-gap-long-v1",created_unix=time.time(),root=str(ROOT),output=str(output),
        python=sys.executable,initial_checkpoint=str(initial),initial_checkpoint_sha256=digest(initial),
        initial_learning_clock=info["completed_updates"],baseline=str(baseline),bank=str(bank),bank_sha256=digest(bank),
        prior_experiment_consumed=1448,prior_offline_steps=128,total_new_updates=total,stages=stages(total),
        num_envs=256,rollout_steps=128,curriculum_updates=512,curriculum_origin=info["completed_updates"],
        evaluation_seed=241201,evaluation_seconds=8.,holdout_seed=241627,source_sha256=hashes,
        authorization="User explicitly requested a long training run, behavioral review and optimization on2026-10-24.",
        long_training_authorized=True,behavior_qualified=False,hardware_ready=False,
        policy_hz=200,physics_and_pd_hz=1000,actor_dim=35,action_dim=6,
        mass_range_at_full_severity=[.9,1.1],timing_profile="native",
        training_scope="Train recovery, quiet, motion residual and exploration; retain frozen locomotion parent.",
        adaptation="After two behavior regressions reduce actor LR and hold severity ramp; never refund updates or revert optimizer invisibly.",
        stop_scope="Nonfinite/native failure or requested stop ends scheduling and preserves charged work.",
        hardware_started=False,remote_changed=False,local_tensorboard_server=False)
    write_json(output/"plan.json",plan)
    return plan


def verify_sources(plan):
    changed=[key for key,value in plan["source_sha256"].items() if digest(ROOT/key)!=value]
    if changed:
        raise ValueError("Frozen campaign source changed: "+", ".join(changed))
    if digest(plan["bank"])!=plan["bank_sha256"]:
        raise ValueError("Frozen pose bank changed")


def verify_restore(parent,initial,output):
    import torch
    a,b=[torch.load(path,map_location="cpu",weights_only=False) for path in (parent,initial)]
    def equal(left,right):
        if isinstance(left,torch.Tensor):
            return isinstance(right,torch.Tensor) and torch.equal(left,right)
        if isinstance(left,dict):
            return left.keys()==right.keys() and all(equal(v,right[k]) for k,v in left.items())
        if isinstance(left,(tuple,list)):
            return len(left)==len(right) and all(equal(x,y) for x,y in zip(left,right,strict=True))
        return left==right
    checks={k:equal(a[k],b[k]) for k in ("actor_state_dict","critic_state_dict")}
    checks["Adam_moments_and_steps"]=equal(a["optimizer_state_dict"]["state"],b["optimizer_state_dict"]["state"])
    for i,(left,right) in enumerate(zip(a["optimizer_state_dict"]["param_groups"],b["optimizer_state_dict"]["param_groups"],strict=True)):
        checks[f"optimizer_group_{i}_except_lr"]=equal({k:v for k,v in left.items() if k!="lr"},
            {k:v for k,v in right.items() if k!="lr"})
    for key in ("completed_updates","environment_rng","domain_rng","torch_rng","cuda_rng"):
        checks[key]=equal(a["infos"][key],b["infos"][key])
    out=dict(checks=checks,parent_sha256=digest(parent),initial_sha256=digest(initial),
        intentional_change="Actor LR and native reset/domain/command distribution; no tensor migration.")
    write_json(output,out)
    if not all(checks.values()):
        raise ValueError("Initial actor/critic/Adam/RNG restore differs")


class Campaign:
    def __init__(self,plan):
        self.plan=plan
        self.output=Path(plan["output"])
        self.path=self.output/"progress.json"
        if self.path.exists():
            raise ValueError("Campaign already started; inspect its receipt instead of starting a duplicate")
        self.stop=False
        self.child=None
        self.state=dict(schema="wheel-leg-gap-long-progress-v1",status="starting",pid=os.getpid(),
            started_unix=time.time(),new_optimizer_updates=0,new_optimizer_attempts=0,
            actual_experiment_consumed=plan["prior_experiment_consumed"],
            current_learning_clock=plan["initial_learning_clock"],current_checkpoint=plan["initial_checkpoint"],
            active_job=None,completed_stages=[],reviews=[],best=None,
            behavior_qualified=False,hardware_ready=False)

    def save(self):
        self.state["updated_unix"]=time.time()
        write_json(self.path,self.state)

    def signal(self,number,frame):
        self.stop=True
        self.state["stop_requested_signal"]=number

    def job(self,name,command,progress=None,parent=None):
        verify_sources(self.plan)
        if self.stop:
            raise InterruptedError("Campaign stop requested")
        log=self.output/"logs"/(name+".log")
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("x") as stream:
            self.child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            self.state["active_job"]=dict(name=name,pid=self.child.pid,command=command,log=str(log))
            self.save()
            initial_checked=False
            restore_error=None
            stop_sent=False
            consumed_before=self.state["new_optimizer_attempts"]
            completed_before=self.state["new_optimizer_updates"]
            while self.child.poll() is None:
                if self.stop and not stop_sent:
                    self.child.send_signal(signal.SIGTERM)
                    stop_sent=True
                if progress is not None and Path(progress).exists():
                    row=read(progress)
                    if parent is not None:
                        completed=row.get("new_optimizer_updates",0)
                        attempted=row.get("new_optimizer_attempts",completed)
                        self.state.update(new_optimizer_updates=completed_before+completed,
                            new_optimizer_attempts=consumed_before+attempted,
                            actual_experiment_consumed=self.plan["prior_experiment_consumed"]+consumed_before+attempted,
                            current_learning_clock=row.get("completed_updates",self.state["current_learning_clock"]))
                        initial=Path(progress).parent/"model_initial.pt"
                        if initial.exists() and not initial_checked:
                            try:
                                verify_restore(parent,initial,Path(progress).parent/"initial_restore.json")
                                initial_checked=True
                            except BaseException as error:
                                self.child.send_signal(signal.SIGTERM)
                                stop_sent=True
                                restore_error=error
                                initial_checked=True
                        self.state["active_job"]["completed_updates"]=completed
                    self.state["active_job"]["native_status"]=row.get("status")
                    self.save()
                time.sleep(3.)
            code=self.child.returncode
        self.child=None
        if progress is not None:
            row=read(progress)
            if parent is not None:
                completed=row.get("new_optimizer_updates",0)
                attempted=row.get("new_optimizer_attempts",completed)
                self.state.update(new_optimizer_updates=completed_before+completed,
                    new_optimizer_attempts=consumed_before+attempted,
                    actual_experiment_consumed=self.plan["prior_experiment_consumed"]+consumed_before+attempted,
                    current_learning_clock=row.get("completed_updates",self.state["current_learning_clock"]))
            self.save()
            if restore_error is not None:
                raise restore_error
            if self.stop:
                raise InterruptedError("Requested stop; last complete checkpoint and all consumption retained")
            if code or row.get("status")!="completed":
                raise RuntimeError(f"Native job{name} incomplete, returncode={code}, status={row.get('status')}")
            if parent is not None and not initial_checked:
                verify_restore(parent,Path(progress).parent/"model_initial.pt",Path(progress).parent/"initial_restore.json")
        elif code:
            raise RuntimeError(f"Job{name} failed with returncode={code}")
        self.state["active_job"]=None
        self.save()

    def review(self,checkpoint,index,training_severity):
        if index==0:
            evaluation=Path(self.plan["baseline"])
            if read(evaluation/"progress.json")["status"]!="completed":
                raise ValueError("Baseline must be complete before the long learner starts")
        else:
            evaluation=self.output/f"eval_{index:03d}"
            self.job(f"eval_{index:03d}",[self.plan["python"],"scripts/evaluate_gap_robustness.py",
                "--checkpoint",str(checkpoint),"--output",str(evaluation),"--actuator-timing","native",
                "--seconds",str(self.plan["evaluation_seconds"]),"--seed",str(self.plan["evaluation_seed"])],
                evaluation/"progress.json")
        quality=self.output/f"standing_{index:03d}.json"
        self.job(f"standing_{index:03d}",[self.plan["python"],"scripts/analyze_gap_standing.py",
            "--run",str(evaluation),"--output",str(quality)])
        metrics=behavior_metrics(evaluation,quality)
        metrics.update(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
            evaluation=str(evaluation),training_severity=training_severity,
            learning_clock=self.state["current_learning_clock"],new_updates=self.state["new_optimizer_updates"])
        self.state["reviews"].append(metrics)
        if self.state["best"] is None or metrics["score"]>self.state["best"]["score"]:
            self.state["best"]=metrics
            write_json(self.output/"best_candidate.json",dict(metrics,hardware_ready=False,formal_acceptance=False))
        if index:
            self.job(f"paired_{index:03d}",[self.plan["python"],"scripts/compare_gap_robustness.py",
                "--before",self.plan["baseline"],"--after",str(evaluation),
                "--output",str(self.output/f"paired_{index:03d}.json")])
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(self.output/"metrics"))
        for group,content in metrics.items():
            fields=content.items() if isinstance(content,dict) else [(None,content)]
            for field,number in fields:
                if isinstance(number,(int,float)):
                    writer.add_scalar("LongGap/"+group+("/"+field if field else ""),number,self.state["current_learning_clock"])
        writer.add_scalar("LongGap/experiment_ppo_consumed",self.state["actual_experiment_consumed"],self.state["current_learning_clock"])
        writer.close()
        write_json(self.output/f"review_{index:03d}.json",metrics)
        self.save()

    def run(self):
        self.save()
        try:
            checkpoint=Path(self.plan["initial_checkpoint"])
            if digest(checkpoint)!=self.plan["initial_checkpoint_sha256"]:
                raise ValueError("Initial checkpoint changed")
            self.review(checkpoint,0,.6)
            for stage in self.plan["stages"]:
                if self.state["new_optimizer_attempts"]+stage["updates"]>self.plan["total_new_updates"]:
                    raise ValueError("Next block exceeds the frozen update budget")
                tuning=adaptation(self.state["reviews"],stage["severity"])
                folder=self.output/f"train_{stage['index']:03d}"
                command=[self.plan["python"],"scripts/train_robust_getup.py","--output",str(folder),
                    "--bank",self.plan["bank"],"--resume",str(checkpoint),"--updates",str(stage["updates"]),
                    "--num-envs",str(self.plan["num_envs"]),"--rollout-steps",str(self.plan["rollout_steps"]),
                    "--seed",str(241300+stage["index"]),"--isolated-origins","--gap-robustness",
                    "--gap-severity",str(tuning["severity"]),"--gap-actuator-timing","native",
                    "--actor-lr",str(tuning["actor_lr"]),"--friction-min",str(stage["friction_min"]),
                    "--domain-scale",".35","--eval-seconds","2",
                    "--curriculum-updates",str(self.plan["curriculum_updates"]),
                    "--curriculum-origin",str(self.plan["curriculum_origin"])]
                self.state["status"]="training"
                self.state["current_stage"]=dict(stage,**tuning)
                self.job(f"train_{stage['index']:03d}",command,folder/"progress.json",parent=checkpoint)
                progress=read(folder/"progress.json")
                if progress["new_optimizer_updates"]!=stage["updates"]:
                    raise ValueError("Native completed count differs from its reserved stage")
                checkpoint=folder/"model_final.pt"
                self.state["current_checkpoint"]=str(checkpoint)
                self.state["completed_stages"].append(dict(stage,**tuning,checkpoint=str(checkpoint),sha256=digest(checkpoint)))
                self.state["status"]="reviewing"
                self.save()
                self.review(checkpoint,stage["index"],tuning["severity"])
            best=self.state["best"]["checkpoint"]
            for name,seconds,extra in (("holdout",12,[]),("standing30s",30,["--case-group","standing"]),
                ("low_mu020",8,["--case-group","standing","--friction-min",".2","--friction-max",".2"]),
                ("low_mu030",8,["--case-group","standing","--friction-min",".3","--friction-max",".3"])):
                folder=self.output/name
                self.job(name,[self.plan["python"],"scripts/evaluate_gap_robustness.py","--checkpoint",best,
                    "--output",str(folder),"--actuator-timing","native","--seconds",str(seconds),
                    "--seed",str(self.plan["holdout_seed"])]+extra,folder/"progress.json")
                self.job(name+"_standing",[self.plan["python"],"scripts/analyze_gap_standing.py",
                    "--run",str(folder),"--output",str(self.output/(name+"_standing.json"))])
            self.state.update(status="completed_unqualified",finished_unix=time.time(),active_job=None,
                note="Long training and behavior reviews complete; inspect trajectories and30s acceptance before export.")
        except InterruptedError as error:
            self.state.update(status="stopped",error=str(error),finished_unix=time.time())
        except BaseException:
            self.state.update(status="failed",error=traceback.format_exc(),finished_unix=time.time())
            raise
        finally:
            self.save()
            write_json(self.output/"result.json",self.state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--initial",type=Path)
    parser.add_argument("--baseline",type=Path)
    parser.add_argument("--total-updates",type=int,default=2048)
    args=parser.parse_args()
    if args.prepare:
        if args.initial is None or args.baseline is None:
            parser.error("Preparation requires the migrated initial checkpoint and complete paired baseline path")
        prepare(args.output,args.initial,args.baseline,args.total_updates)
        return
    plan=read(args.output/"plan.json")
    if plan["root"]!=str(ROOT) or plan["output"]!=str(args.output.resolve()):
        raise ValueError("Campaign plan belongs to another checkout or output path")
    with (args.output/"supervisor.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign=Campaign(plan)
        signal.signal(signal.SIGTERM,campaign.signal)
        signal.signal(signal.SIGINT,campaign.signal)
        campaign.run()


if __name__=="__main__":
    main()
