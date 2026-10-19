"""Candidate entry functions; copied into the trainer, not imported at runtime."""


def restored_training_frames(info, old_plan):
    """Successful inherited rollout frames; discarded trials remain separately charged."""
    value = info.get("training_transitions_total")
    if value is None:
        if info.get("population_transition"):
            raise ValueError("A changed-population parent requires its saved cumulative frame count")
        value = info["successful_updates_total"] * info["num_envs"] * old_plan["num_steps_per_env"]
    if type(value) is not int or value < 0:
        raise ValueError("Inherited successful rollout frames must be a nonnegative integer")
    return value


def validate_population_transition(info, old_plan, new_plan, new_count, revision, new_sources):
    """Authenticate a population-only migration without changing its objective or clocks."""
    changed = {key for key in set(old_plan) | set(new_plan) if old_plan.get(key) != new_plan.get(key)}
    if changed != {"allowed_num_envs", "default_num_envs"}:
        raise ValueError("Population migration can change only the two population plan fields")
    old_count = info["num_envs"]
    if (type(new_count) is not int or new_count <= 0 or new_count == old_count
            or new_plan["allowed_num_envs"] != [new_count] or new_plan["default_num_envs"] != new_count
            or old_count not in old_plan["allowed_num_envs"]):
        raise ValueError("Population migration must name the exact old and new populations")
    entry = "scripts/train_flat_base.py"
    source_changes = {name for name in set(info["source_sha256"]) | set(new_sources)
                      if info["source_sha256"].get(name) != new_sources.get(name)}
    if source_changes != {entry}:
        raise ValueError("Population migration permits only the reviewed trainer entry change")
    expected = dict(schema="flat-training-population-transition-v1", old_num_envs=old_count,
        new_num_envs=new_count, num_steps_per_env=old_plan["num_steps_per_env"],
        inherited_successful_updates=info["successful_updates_total"],
        inherited_successful_training_frames=restored_training_frames(info, old_plan),
        additional_critic_warmup_updates=0)
    if revision.get("population_transition") != expected:
        raise ValueError("Population receipt must retain the exact inherited exposure and omit new warmup")
    return expected


def probe_population_capacity(runner, env, plan, report, torch):
    """One full native rollout and all learner backward passes, with Adam writes disabled."""
    from copy import deepcopy
    import time
    def exact(left, right):
        if torch.is_tensor(left):
            return torch.equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)):
            return type(left) is type(right) and len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
        return left == right
    before = deepcopy(runner.alg.save())
    torch.cuda.reset_peak_memory_stats()
    obs = env.get_observations().to(runner.device)
    runner.alg.train_mode()
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        for index in range(plan["num_steps_per_env"]):
            actions = runner.alg.act(obs)
            obs, rewards, dones, extras = env.step(actions.to(env.device))
            obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
            if (not all(torch.isfinite(value).all() for value in obs.values())
                    or not torch.isfinite(rewards).all()):
                raise RuntimeError("Capacity rollout contains a nonfinite observation or reward")
            runner.alg.process_env_step(obs, rewards, dones, extras)
            if index % 24 == 0:
                print("FLAT_CAPACITY", index, report["num_envs"], flush=True)
        runner.alg.compute_returns(obs)
    torch.cuda.synchronize()
    collection_s = time.monotonic() - started
    blocked = 0
    original_step = runner.alg.optimizer.step
    def no_optimizer_write(*args, **kwargs):
        nonlocal blocked
        blocked += 1
    runner.alg.optimizer.step = no_optimizer_write
    learning_started = time.monotonic()
    try:
        losses = runner.alg.update()
    finally:
        runner.alg.optimizer.step = original_step
    torch.cuda.synchronize()
    learning_s = time.monotonic() - learning_started
    after = runner.alg.save()
    checks = {name + "_exact": exact(before[name], after[name]) for name in
              ("actor_state_dict", "critic_state_dict", "optimizer_state_dict")}
    checks["all_native_minibatch_backward_passes"] = blocked == plan["num_mini_batches"] * plan["num_learning_epochs"]
    checks["finite_native_losses"] = all(__import__("math").isfinite(value) for value in losses.values())
    report["capacity_probe"] = dict(passed=all(checks.values()), checks=checks,
        num_envs=report["num_envs"], rollout_steps=plan["num_steps_per_env"],
        native_policy_frames=report["num_envs"] * plan["num_steps_per_env"],
        optimizer_writes=0, blocked_optimizer_writes=blocked, collection_s=collection_s,
        backward_s=learning_s, policy_frames_per_collection_s=report["num_envs"] * plan["num_steps_per_env"] / collection_s,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        losses=losses, scope="Capacity only; unchanged parent weights and Adam; isolated RNG; no learning checkpoint or behavior qualification")
    if not all(checks.values()):
        raise RuntimeError("Capacity probe changed a model or Adam, or failed its native backward checks")
