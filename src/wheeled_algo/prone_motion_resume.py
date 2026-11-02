"""Explicit, same-course recovery of an interrupted prone-motion learner.

Old checkpoints predate resume authorization. Only the explicit --resume input
opens this path; ordinary launches still construct a fresh learner.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


# These files implement resume and do not change the plant, reward or course.
RESUME_SOURCE_CHANGES = frozenset({
    "scripts/train_prone_motion.py", "scripts/train_prone_motion.sh",
    "src/wheeled_tasks/chassis/prone_motion_cli.py",
    "src/wheeled_algo/prone_motion_resume.py", "tests/test_prone_motion_resume.py",
    "docs/PRONE_MOTION_RESUME.md",
    "src/wheeled_algo/training_control.py", "tests/test_training_control.py", "docs/TRAINCTL.md",
})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(checkpoint):
    """Read JSON provenance only, so native probes never deserialize a policy."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix != ".pt":
        raise ValueError("--resume requires an existing training .pt checkpoint")
    folder = checkpoint.parent
    plan = json.loads((folder/"curriculum_plan.resolved.json").read_text())
    from wheeled_tasks.chassis.prone_motion_cli import validate_resolved
    validate_resolved(plan)
    options = json.loads((folder/"run_options.json").read_text())
    sources = json.loads((folder/"source_hashes.json").read_text())
    checkpoint_hash = digest(checkpoint)
    progress_path = folder/"progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    last = progress.get("last_checkpoint", {})
    if Path(last.get("path", "")).name == checkpoint.name and last.get("sha256") != checkpoint_hash:
        raise ValueError("Resume checkpoint differs from the saved SHA-256 receipt")
    return dict(path=checkpoint, sha256=checkpoint_hash, plan=plan, options=options,
                sources=sources, bank_sha256=digest(folder/"pose_bank.json"), progress=progress)


def authenticate_sources(bundle, current, root):
    """No model, observation, reward, curriculum or dynamics migration is allowed."""
    root = Path(root).resolve()
    previous = bundle["sources"]
    if not previous:
        raise ValueError("Resume requires the original source manifest")
    changed = []
    for name, expected in previous.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or invalid original source: "+name)
        actual = digest(path)
        if current.get(name) != actual:
            raise ValueError("Original source is absent from the new freeze: "+name)
        if actual != expected:
            changed.append(name)
    added = sorted(set(current)-set(previous))
    unexpected = (set(changed)|set(added))-RESUME_SOURCE_CHANGES
    if unexpected:
        raise ValueError("Same-course resume refuses changed training sources: "+str(sorted(unexpected)))
    return dict(original_source_count=len(previous), unchanged_source_count=len(previous)-len(changed),
                resume_support_changes=sorted(changed), added_resume_sources=added,
                plant_reward_observation_and_course_unchanged=True)


def validate_checkpoint(state, bundle):
    import torch
    required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos"}
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError("Resume needs actor, critic, normalizers, Adam and iteration state")
    info = state["infos"]
    if (info.get("schema") != "prone-motion-checkpoint-v1" or info.get("training_origin") != "scratch"
            or info.get("resolved_plan") != bundle["plan"] or info.get("source_hashes") != bundle["sources"]
            or info.get("bank_sha256") != bundle["bank_sha256"]):
        raise ValueError("Checkpoint and original run provenance disagree")
    completed = state["iter"]
    if (type(completed) is not int or completed != info.get("completed_updates")
            or not 0 < completed < bundle["plan"]["max_updates"]):
        raise ValueError("Resume needs a complete update before the original course endpoint")
    optimizer = state["optimizer_state_dict"]
    if not optimizer.get("state") or not optimizer.get("param_groups"):
        raise ValueError("Cannot resume with an empty Adam")
    for group in ("actor_state_dict", "critic_state_dict"):
        values = state[group]
        if not all(f"obs_normalizer.{key}" in values for key in ("_mean", "_var", "_std", "count")):
            raise ValueError("Checkpoint is missing observation normalization")
        if not all(torch.isfinite(value).all() for value in values.values()):
            raise ValueError("Checkpoint has nonfinite model values")
    for values in optimizer["state"].values():
        if not {"step", "exp_avg", "exp_avg_sq"} <= values.keys() or float(values["step"]) <= 0:
            raise ValueError("Checkpoint is missing accumulated Adam state")
        if not all(torch.isfinite(value).all() for value in values.values() if isinstance(value, torch.Tensor)):
            raise ValueError("Checkpoint has nonfinite Adam values")
    if not all(key in info for key in ("torch_rng", "cuda_rng", "env_rng", "domain_rng")):
        raise ValueError("Checkpoint is missing saved RNG state")
    restore_curriculum(state, bundle["plan"])
    if bundle["plan"].get("multi_pose_recovery") is not None:
        from wheeled_tasks.chassis.multi_pose_recovery import ResetSampler
        ResetSampler.validate_state(info.get("recovery_sampler"))
        if type(info.get("prone_reset_generation")) is not int or info["prone_reset_generation"] < 0:
            raise ValueError("Checkpoint is missing the curriculum reset clock")
    return completed


def restore_curriculum(state, plan):
    """Restore admission before constructing any reset distribution or domain."""
    info = state["infos"]
    mastery = course = None
    gate = plan.get("multi_pose_recovery", {}).get("mastery_gate")
    if gate is not None:
        from wheeled_tasks.chassis.recovery_mastery import RecoveryMastery
        mastery = RecoveryMastery(gate)
        mastery.load_state_dict(info.get("recovery_mastery"))
        if any(clock is not None and clock > state["iter"]
               for clock in (mastery.unlocked_at, mastery.last_evaluated_update)):
            raise ValueError("Saved recovery mastery is newer than the checkpoint")
    if plan.get("global_curriculum") is not None:
        from wheeled_tasks.chassis.global_curriculum import GlobalCurriculum
        course = GlobalCurriculum(plan, mastery)
        course.load_state_dict(info.get("global_curriculum"))
        if course.recovery.state_dict() != info["recovery_mastery"]:
            raise ValueError("Saved global and recovery mastery disagree")
        completed = state["iter"]
        clocks = [course.domain_started_at, course.last_review, mastery.last_evaluated_update,
                  mastery.unlocked_at, *course.skill_last_review.values()]
        clocks.extend(row["completed_updates"] for row in course.reviews)
        if any(clock is not None and clock > completed for clock in clocks):
            raise ValueError("Saved curriculum is newer than the checkpoint")
    return mastery, course


def equal_state(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_state(v, right[k]) for k,v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_state(a,b) for a,b in zip(left,right))
    return left == right


def restore(runner, state, bundle):
    """Restore the native algorithm, then prove exact model and Adam equality."""
    from wheeled_algo.scratch_standing import tensor_hash
    completed = validate_checkpoint(state, bundle)
    if not runner.alg.load(state, None, strict=True):
        raise RuntimeError("Native PPO refused full iteration restoration")
    restored = runner.alg.save()
    checks = {key+"_exact": equal_state(state[key], restored[key])
              for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict")}
    checks["learning_rate_exact"] = runner.alg.learning_rate == state["optimizer_state_dict"]["param_groups"][0]["lr"]
    if not all(checks.values()):
        raise RuntimeError("Resume state mismatch: "+str(checks))
    runner.current_learning_iteration = completed
    runner.env.actor_updates = completed
    runner.logger.tot_timesteps = completed*bundle["plan"]["num_envs"]*bundle["plan"]["rollout_steps"]
    adam_steps = [float(value["step"]) for value in runner.alg.optimizer.state.values()]
    return dict(schema="prone-motion-resume-v1", checkpoint_loaded=True, teacher_loaded=False,
                checkpoint=str(bundle["path"]), checkpoint_sha256=bundle["sha256"], completed_updates=completed,
                remaining_updates=bundle["plan"]["max_updates"]-completed,
                optimizer_states=len(runner.alg.optimizer.state), optimizer_step_range=[min(adam_steps), max(adam_steps)],
                learning_rate=runner.alg.learning_rate,
                actor_tensor_sha256=tensor_hash(runner.alg.actor.state_dict()),
                critic_tensor_sha256=tensor_hash(runner.alg.critic.state_dict()), checks=checks,
                environment_state_restored=False, episodes_reinitialized=True,
                curriculum_restarted=False, original_initialization=deepcopy(state["infos"]["initialization"]))


def restore_rng(state, env):
    import random
    import numpy as np
    import torch
    info = state["infos"]
    torch.set_rng_state(info["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in info["cuda_rng"]])
    env.generator.set_state(info["env_rng"].cpu())
    env.domain.rng.set_state(info["domain_rng"].cpu())
    if "python_rng" in info:
        random.setstate(info["python_rng"])
    if "numpy_rng" in info:
        np.random.set_state(info["numpy_rng"])
    sampler = getattr(env, "recovery_sampler", None)
    if sampler is not None:
        sampler.load_state_dict(info["recovery_sampler"])
        env.prone_reset_generation = info["prone_reset_generation"]
    return dict(torch=True, cuda=True, environment=True, domain=True,
                python="python_rng" in info, numpy="numpy_rng" in info,
                recovery_sampler=sampler is not None,
                exact_physics_trajectory_replay=False)
