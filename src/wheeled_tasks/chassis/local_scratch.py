"""An independent local experiment using the frozen native V6 speed recipe.

The native worker and independent design are authenticated separately. The local
200 Hz policy plan explicitly converts rollout, discount and action differences
in physical time; the latest speed objectives and physical safeguards remain.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PLAN_FILE = "contracts/v6_local_scratch_200hz_v2.json"
POLICY200_PLAN_FILE = "contracts/v6_local_scratch_200hz_v1.json"
LEGACY_PLAN_FILE = "contracts/v6_local_scratch_v1.json"
SCHEMA = "v6-local-scratch-worker-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_plan(root, plan_file=PLAN_FILE):
    root = Path(root)
    if plan_file not in (PLAN_FILE, POLICY200_PLAN_FILE, LEGACY_PLAN_FILE):
        raise ValueError("Unsupported local scratch plan file")
    plan = json.loads((root / plan_file).read_text())
    if plan.get("schema") != "v6-local-scratch-plan-v1":
        raise ValueError("Unsupported local scratch plan")
    for name in ("source_worker", "source_design", "reward_reference"):
        if digest(root / plan[name + "_file"]) != plan[name + "_sha256"]:
            raise ValueError(f"Local scratch {name} source changed")
    if plan_file != LEGACY_PLAN_FILE and digest(root / plan["control_file"]) != plan["control_sha256"]:
        raise ValueError("Local scratch 200 Hz control source changed")
    if plan.get("prior_experiment_receipt_file"):
        path = root / plan["prior_experiment_receipt_file"]
        receipt = json.loads(path.read_text())
        if (digest(path) != plan["prior_experiment_receipt_sha256"]
                or receipt.get("status") != "behavior_rejected"
                or receipt.get("consumption") != plan["prior_experiment_consumption"]
                or plan["update_ceiling"] + plan["prior_experiment_consumption"] != 20000):
            raise ValueError("Prior local experiment consumption binding changed")
    return plan


def bundle_paths(root, directory):
    root = Path(root).resolve()
    directory = Path(directory)
    directory = directory if directory.is_absolute() else root / directory
    directory = directory.resolve()
    if not directory.is_relative_to(root / "runs"):
        raise ValueError("Generated local configuration must be inside this repository's runs directory")
    relative = directory.relative_to(root)
    return {name: str(relative / (name + ".json")) for name in ("worker", "design", "budget")}


def _encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def build_bundle(root, directory, num_envs, *, plan_file=PLAN_FILE):
    root = Path(root)
    plan = read_plan(root, plan_file)
    if type(num_envs) is not int or num_envs not in plan["allowed_num_envs"]:
        raise ValueError("Local scratch environment count must be one of the frozen populations")
    paths = bundle_paths(root, directory)
    worker = json.loads((root / plan["source_worker_file"]).read_text())
    latest = json.loads((root / plan["reward_reference_file"]).read_text())
    # The native v18 worker is also v26/v27's named source. Its complete
    # payload differs from the current static worker only in these bindings.
    bindings = {"design_preflight", "task_campaign", "integrated_production"}
    if {k: v for k, v in worker.items() if k not in bindings} != {
            k: v for k, v in latest.items() if k not in bindings}:
        raise ValueError("Latest speed recipe no longer matches the authenticated native reward source")
    worker.pop("task_campaign")
    timing_changes = ()
    if plan_file != LEGACY_PLAN_FILE:
        temporal = plan["ppo_timing"]
        worker.update(physics_dt=plan["timing"]["physics_dt"], policy_dt=plan["timing"]["policy_dt"],
            control_math_source=plan["control_file"], control_math_sha256=plan["control_sha256"],
            num_steps_per_env=temporal["num_steps_per_env"], gamma=temporal["gamma"], lam=temporal["lam"],
            action_difference_reference_dt=plan["action_difference_reference_dt"])
        timing_changes = ("physics_dt", "policy_dt", "control_math_source", "control_math_sha256",
            "num_steps_per_env", "gamma", "lam", "action_difference_reference_dt")
    if "initial_noise_std" in plan:
        worker["initial_noise_std"] = plan["initial_noise_std"]
        timing_changes += ("initial_noise_std",)
    worker.update(target_num_envs=num_envs,
        ppo_minibatch_samples=num_envs * worker["num_steps_per_env"] // worker["num_mini_batches"],
        checkpoint_first_update=plan["checkpoint_first_update"])
    for field, origin in plan["curriculum_origins"].items():
        worker[field]["origin_actor_update"] = origin
    marker = dict(schema=SCHEMA, plan_file=plan_file, plan_sha256=digest(root / plan_file),
        initialization=plan["initialization"], num_envs=num_envs, seed=plan["seed"])
    worker["local_scratch"] = marker
    budget = dict(schema="v6-local-scratch-budget-v1", initialization=plan["initialization"],
        inherited_consumption=plan.get("prior_experiment_consumption", 0),
        update_ceiling=plan["update_ceiling"], num_envs=num_envs,
        num_steps_per_env=worker["num_steps_per_env"],
        transition_ceiling=plan["update_ceiling"] * num_envs * worker["num_steps_per_env"])
    if plan_file != LEGACY_PLAN_FILE:
        budget.update(policy_dt=worker["policy_dt"],
            rollout_seconds=worker["num_steps_per_env"] * worker["policy_dt"],
            simulated_seconds_per_env_ceiling=plan["update_ceiling"] * worker["num_steps_per_env"] * worker["policy_dt"])
    worker.update(budget_spec_file=paths["budget"], budget_spec_sha256=hashlib.sha256(_encode(budget)).hexdigest())
    source_design = json.loads((root / plan["source_design_file"]).read_text())
    row = deepcopy(source_design["stages"][plan["stage"]]["roles"]["train"])
    row["contract"].pop("task_campaign")
    changes = ("target_num_envs", "ppo_minibatch_samples", "checkpoint_first_update",
        "budget_spec_file", "budget_spec_sha256", *plan["curriculum_origins"], *timing_changes)
    for key in changes:
        row["contract"][key] = deepcopy(worker[key])
    row["contract"]["local_scratch"] = deepcopy(marker)
    row["invocation"]["num_envs"] = num_envs
    for field, origin in plan["curriculum_origins"].items():
        expected = row["runtime"][field]
        if "origin_actor_update" in expected:
            expected["origin_actor_update"] = origin
        expected["settings"]["origin_actor_update"] = origin
    if plan_file != LEGACY_PLAN_FILE:
        control = json.loads((root / plan["control_file"]).read_text())
        row["control"]["timing"] = deepcopy(plan["timing"])
        row["control"]["simulation_timing_approximation"] = deepcopy(control["simulation_timing_approximation"])
        row["runtime"].update(plan["timing"])
        row["runtime"]["action_difference_reference_dt"] = plan["action_difference_reference_dt"]
        for field in ("finite_passive_landing", "start_stop_timing", "nonstandard_start", "release_stop"):
            row["runtime"][field]["policy_dt"] = plan["timing"]["policy_dt"]
    design = dict(schema=source_design["schema"], status="frozen", scope=source_design["scope"],
        training_release=False, hardware_deployment_ready=False,
        review_basis=dict(source_design_file=plan["source_design_file"],
            source_design_sha256=plan["source_design_sha256"], plan_file=plan_file,
            plan_sha256=marker["plan_sha256"], changed_contract_paths=list(changes) + ["task_campaign", "local_scratch"]),
        stages={plan["stage"]: {"roles": {"train": row}}})
    worker["design_preflight"] = dict(file=paths["design"],
        sha256=hashlib.sha256(_encode(design)).hexdigest(), stage=plan["stage"])
    return paths, {"worker": worker, "design": design, "budget": budget}


def write_bundle(root, directory, num_envs, *, plan_file=PLAN_FILE):
    paths, bundle = build_bundle(root, directory, num_envs, plan_file=plan_file)
    directory = Path(root) / Path(paths["worker"]).parent
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in bundle.items():
        (Path(root) / paths[name]).write_bytes(_encode(value))
    return Path(root) / paths["worker"]


def validate_config(config, root):
    marker = config.get("local_scratch", {})
    if marker.get("schema") != SCHEMA:
        raise ValueError("Unsupported local scratch worker")
    directory = Path(config["design_preflight"]["file"]).parent
    _, bundle = build_bundle(root, directory, marker.get("num_envs"), plan_file=marker.get("plan_file"))
    if config != bundle["worker"]:
        raise ValueError("Local scratch worker differs from its frozen native derivation")
    for name in ("design", "budget"):
        path = Path(root) / directory / (name + ".json")
        if path.read_bytes() != _encode(bundle[name]):
            raise ValueError(f"Local scratch independent {name} changed")
    return bundle["budget"]


def validate_invocation(config, args, root):
    validate_config(config, root)
    plan = read_plan(root, config["local_scratch"]["plan_file"])
    # A continuation keeps this exact local contract and sealed consumption.
    # Foreign state and caller-supplied clocks remain forbidden.
    if (args.transfer or args.consumed_updates is not None
            or args.learning_rate_scale is not None or args.task_actor_update is not None
            or args.stage_actor_update is not None or args.actor_updates_consumed != 0
            or args.global_actor_update_offset != 0 or args.block_index != 0
            or args.confirmed_jump_bins_json != "[]"
            or args.seed != plan["seed"] or args.num_envs != config["target_num_envs"]
            or args.stage != plan["stage"] or not 1 <= args.updates <= plan["update_ceiling"]):
        raise ValueError("Local scratch requires fresh learning state, zero clocks and its frozen invocation")
    if args.resume:
        try:
            parent = validate_parent(args.resume, config)
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError("Local scratch requires fresh learning state or its own finalized checkpoint") from error
        if parent + args.updates > plan["update_ceiling"]:
            raise ValueError("Local continuation exceeds the cumulative local update ceiling")
        return parent
    return 0


def validate_parent(checkpoint, config):
    """Accept only a finalized same-experiment block with no discarded work."""
    from .full_curriculum import checkpoint_contract_path, checkpoint_update_count
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.name != "model_final.pt":
        raise ValueError("Local continuation requires the finalized block checkpoint")
    source = checkpoint_contract_path(checkpoint)
    if json.loads(source.read_text()) != config:
        raise ValueError("Local continuation changes the exact experiment contract")
    seal = json.loads(checkpoint.with_suffix(".checkpoint.json").read_text())
    count = checkpoint_update_count(checkpoint)
    completion = json.loads((checkpoint.parent / "completion.json").read_text())
    if (completion["status"] not in ("completed", "stopped")
            or count != completion.get("parent_updates", 0) + completion["successful_updates"]
            or seal.get("learning_lineage_updates") != count
            or seal.get("stage_actor_update") != count
            or seal.get("actor_updates_consumed") != count
            or seal.get("global_actor_update_offset") != 0
            or seal.get("role") != "learning_state"):
        raise ValueError("Local parent has incomplete or inconsistent learning/consumption clocks")
    return count


def restore_bundle(root, checkpoint):
    """Copy the authenticated generated bindings without changing their paths/hash."""
    from .full_curriculum import checkpoint_contract_path
    root, checkpoint = Path(root), Path(checkpoint).resolve()
    config = json.loads(checkpoint_contract_path(checkpoint).read_text())
    validate_parent(checkpoint, config)
    directory = Path(config["design_preflight"]["file"]).parent
    # Rebuild from the independent plan before accepting checkpoint-controlled paths.
    paths, bundle = build_bundle(root, directory, config["local_scratch"]["num_envs"],
        plan_file=config["local_scratch"]["plan_file"])
    if bundle["worker"] != config:
        raise ValueError("Parent is not the frozen local experiment derivation")
    target = root / directory
    target.mkdir(parents=True, exist_ok=False)
    for name in ("design", "budget"):
        original = checkpoint.parent / "source" / paths[name]
        if original.read_bytes() != _encode(bundle[name]):
            raise ValueError("Parent's independent local source binding changed")
        (root / paths[name]).write_bytes(original.read_bytes())
    (root / paths["worker"]).write_bytes(checkpoint_contract_path(checkpoint).read_bytes())
    validate_config(config, root)
    return root / paths["worker"]


def source_files(root, plan_file=PLAN_FILE):
    plan = read_plan(root, plan_file)
    return [plan_file, "src/wheeled_tasks/chassis/local_scratch.py", "scripts/train_local_chassis.py",
        plan["source_worker_file"], plan["source_design_file"], plan["reward_reference_file"],
        "src/wheeled_tasks/chassis/nonstandard_start.py", "src/wheeled_tasks/chassis/nonstandard_start_revision.py",
        "src/wheeled_tasks/chassis/release_stop.py", "src/wheeled_tasks/chassis/ordinary_posture_curriculum.py",
        "src/wheeled_tasks/chassis/locomotion_difficulty.py", "src/wheeled_tasks/chassis/graded_locomotion_revision.py",
        *([plan["prior_experiment_receipt_file"]] if plan.get("prior_experiment_receipt_file") else [])]
