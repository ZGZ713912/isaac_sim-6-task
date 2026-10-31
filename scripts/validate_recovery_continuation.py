#!/usr/bin/env python3
"""Isolated continuation experiment; its weights cannot enter final v4 training."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts")]
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--updates",type=int,default=0,help="New validation updates, 0=only cold prerequisite assessment")
    p.add_argument("--num-envs",type=int,default=32)
    p.add_argument("--device",default="cpu")
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--allow-friction-migration",action="store_true",
        help="Explicitly allow only the authenticated pre-friction v202 parent; re-assess mastery in the new plant")
    p.add_argument("--allow-belly-pretraining",action="store_true",
        help="If prerequisite mastery fails, explicitly permit further belly learning instead of complex recovery")
    return p


def load_candidate(path, plan, *, allow_friction_migration=False):
    import torch
    from train_scratch_standing import digest
    from wheeled_tasks.chassis.prone_motion import validate
    path = path.expanduser().resolve(strict=True)
    state = torch.load(path,map_location="cpu",weights_only=False)
    info = state["infos"]
    if info.get("schema") != "prone-motion-checkpoint-v1" or info.get("training_origin") != "scratch":
        raise ValueError("Validation needs an authenticated original full-vehicle scratch checkpoint")
    source = validate(info["resolved_plan"])
    keys=("full_vehicle","mechanism","leg_motor_friction","wheel_model","gas_spring_model",
          "gas_spring_feedforward","standing_odometry","timing")
    changed=[key for key in keys if source.get(key) != plan.get(key)]
    migration=None
    if changed:
        if not allow_friction_migration:
            raise ValueError("Continuation candidate differs in measured mechanics or observation: "+changed[0])
        receipt=json.loads((ROOT/"model/wheel_leg_v2.0.2/friction_v1/migration.json").read_text())
        binding=receipt["previous_plan"]
        if digest(ROOT/binding["file"]) != binding["sha256"]:
            raise ValueError("The authenticated friction parent plan changed")
        parent=json.loads((ROOT/binding["file"]).read_text())
        current=json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4_standard_20261031.json").read_text())
        if any(source.get(k) != parent.get(k) or plan.get(k) != current.get(k) for k in keys):
            raise ValueError("Only the recorded pre-friction parent to identified-friction derivative is allowed")
        for k in ("geometry_table_sha256","geometry_model_spec_sha256"):
            if source["standing_odometry"][k] != plan["standing_odometry"][k]:
                raise ValueError("Friction continuation cannot change robot/encoder geometry")
        migration=dict(kind="recorded_prefriction_parent_to_identified_leg_friction",changed_contract_keys=changed,
            parent_plan=binding,mastery_must_be_remeasured=True)
    for name in ("actor_state_dict","critic_state_dict","optimizer_state_dict"):
        if not state.get(name): raise ValueError("Continuation candidate lacks "+name)
    for name in ("actor_state_dict","critic_state_dict"):
        if not all(torch.isfinite(v).all() for v in state[name].values()):
            raise ValueError("Nonfinite checkpoint model state")
    if not state["optimizer_state_dict"].get("state"):
        raise ValueError("A continuation candidate must have the original populated Adam state")
    return path,state,dict(checkpoint=str(path),sha256=digest(path),parent_completed_updates=state["iter"],
        source_continuation_forbidden=info.get("continuation_forbidden"),plant_migration=migration,
        authorization_scope="User allows continuation for isolated recovery validation only; final10000 remains scratch")


def assert_restored(actual, expected):
    """Assert every inherited Adam moment/step and parameter group, not its count."""
    import torch
    if isinstance(expected,torch.Tensor):
        torch.testing.assert_close(actual.detach().cpu(),expected.detach().cpu(),rtol=0.,atol=0.)
    elif isinstance(expected,dict):
        if set(actual) != set(expected): raise ValueError("Restored checkpoint keys differ")
        for key in expected: assert_restored(actual[key],expected[key])
    elif isinstance(expected,(list,tuple)):
        if len(actual) != len(expected): raise ValueError("Restored checkpoint lengths differ")
        for a,b in zip(actual,expected): assert_restored(a,b)
    elif actual != expected:
        raise ValueError("Restored checkpoint value differs")


def main():
    args = parser().parse_args()
    if not 0 <= args.updates < 10000 or not 20 <= args.num_envs <= 256:
        raise ValueError("Use a bounded validation budget below10000 and20..256 environments")
    from train_prone_motion import configuration,learner_config,source_hashes
    from train_scratch_standing import write_json,archive_committed_sources,digest
    from wheeled_tasks.chassis.prone_motion_cli import resolve_plan
    original = json.loads((ROOT/"contracts/wheel_leg_prone_motion_v4.json").read_text())
    plan = resolve_plan(original,num_envs=args.num_envs,domain_randomization="off")
    path,state,origin = load_candidate(args.checkpoint,plan,allow_friction_migration=args.allow_friction_migration)
    args.output.mkdir(parents=True,exist_ok=False)
    cfg,settings,bank = configuration(args.output,plan,num_envs=args.num_envs,device=args.device,performance_backend="eager")
    report = dict(schema="recovery-continuation-validation-v1",scope="isolated_validation_only",
        origin=origin,requested_validation_updates=args.updates,completed_validation_updates=0,
        final_training_eligible=False,final_training_checkpoint_reads_allowed=False,hardware_ready=False,
        status="preflight_completed" if args.preflight_only else "starting")
    write_json(args.output/"receipt.json",report)
    contract=json.loads((args.output/"task_contract.json").read_text())
    contract.update(scope="continuation_validation_only",training_origin="explicit_validation_continuation",
        validation_updates=args.updates,formal_training_budget_updates=10000,final_training_eligible=False)
    write_json(args.output/"task_contract.json",contract)
    if args.preflight_only: return
    sources=source_hashes(ROOT/"contracts/wheel_leg_prone_motion_v4.json",plan)
    sources[str(Path(__file__).relative_to(ROOT))]=digest(Path(__file__))
    report["source_revision"]=archive_committed_sources(sources,args.output)
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(dict(headless=True,device=args.device,enable_cameras=False))
    env=runner=hook=None
    try:
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from wheeled_tasks.chassis.prone_motion_env import ProneMotionEnv
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_tasks.chassis.recovery_mastery import RecoveryMastery
        from wheeled_algo.recovery_rollout import RecoveryRolloutMask
        from wheeled_algo.recovery_mastery import RecoveryMasteryHook
        from evaluate_recovery_mastery import evaluate
        torch.set_num_threads(4)
        mastery=RecoveryMastery(plan["multi_pose_recovery"]["mastery_gate"])
        env=ProneMotionEnv(cfg,json.loads((ROOT/cfg["asset_directory"]/"manifest.json").read_text()),
            load_chassis_control(ROOT/cfg["control_math_source"]),ROOT,stage_name="speed",num_envs=args.num_envs,
            device=args.device,seed=plan["seed"],level=1.,run_role="train",bank=bank,settings=settings,recovery_mastery=mastery)
        env.actor_updates=state["iter"]
        agent=learner_config(plan,args.device);agent["max_iterations"]=args.updates
        runner=OnPolicyRunner(env,agent,log_dir=str(args.output),device=args.device)
        runner.alg.load(state,load_cfg=None,strict=True)
        for name,model in (("actor_state_dict",runner.alg.actor),("critic_state_dict",runner.alg.critic)):
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(),state[name][key],rtol=0.,atol=0.)
        restored=runner.alg.optimizer.state_dict()
        assert_restored(restored,state["optimizer_state_dict"])
        report["restored"]=dict(actor=True,critic=True,adam_state_entries=len(restored["state"]),
            learning_rate=runner.alg.optimizer.param_groups[0]["lr"])
        if env.global_curriculum is not None:
            from wheeled_algo.global_curriculum import GlobalCurriculumHook
            hook=GlobalCurriculumHook(runner,env.global_curriculum,completed=lambda:env.actor_updates,output=args.output)
            hook.review(state["iter"],force=True)
            report["global_curriculum"]=env.global_curriculum.state_dict()
        baseline=evaluate(env,runner.alg.actor,mastery.config)
        baseline["decision"]=mastery.review(baseline["cases"],state["iter"])
        write_json(args.output/"baseline_mastery.json",baseline)
        report["baseline_mastery"]=baseline["decision"]
        standing_passed=env.global_curriculum is None or env.global_curriculum.levels["standing"] > 0
        if args.updates and (mastery.unlocked and standing_passed or args.allow_belly_pretraining):
            RecoveryRolloutMask(runner.alg,num_envs=args.num_envs)
            native_update=runner.alg.update
            def update():
                result=native_update()
                report["completed_validation_updates"]+=1
                env.actor_updates=state["iter"]+report["completed_validation_updates"]
                report["recovery_mastery"]=mastery.state_dict()
                write_json(args.output/"receipt.json",report)
                return result
            runner.alg.update=update
            if hook is None:
                hook=RecoveryMasteryHook(runner,mastery,completed=lambda:env.actor_updates,output=args.output)
            # The research artifact has an incompatible origin marker. The
            # final training CLI has no checkpoint option or loading path.
            def save(filename,infos=None):
                saved=runner.alg.save()
                saved.update(iter=report["completed_validation_updates"],infos=dict(
                    schema="recovery-validation-checkpoint-v1",training_origin="continuation_validation_only",
                    final_training_eligible=False,parent=origin,recovery_mastery=mastery.state_dict(),resolved_plan=plan))
                torch.save(saved,filename)
            runner.save=save
            env.reset(torch.arange(env.num_envs,device=env.device))
            runner.learn(args.updates,init_at_random_ep_len=False)
            save(args.output/"validation_final.pt")
            report["status"]="validation_completed_unqualified"
        else:
            report["status"]="assessment_completed" if not args.updates else "blocked_by_belly_prerequisite"
    except BaseException:
        import traceback
        report.update(status="failed",error=traceback.format_exc())
        raise
    finally:
        write_json(args.output/"receipt.json",report)
        if hook is not None: hook.close()
        if runner is not None and runner.logger.writer is not None:
            runner.logger.writer.flush();runner.logger.writer.close()
        if env is not None: env.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
