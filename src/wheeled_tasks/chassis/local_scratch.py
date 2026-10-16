"""An independent local experiment using the frozen native V6 speed recipe.

The native worker and independent design are authenticated separately. Only
initialization, population, curriculum origins and the independent local budget
change; the latest speed reward, mechanics and physical safeguards are retained.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PLAN_FILE = "contracts/v6_local_scratch_v1.json"
SCHEMA = "v6-local-scratch-worker-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_plan(root):
    root = Path(root)
    plan = json.loads((root / PLAN_FILE).read_text())
    if plan.get("schema") != "v6-local-scratch-plan-v1":
        raise ValueError("Unsupported local scratch plan")
    for name in ("source_worker", "source_design", "reward_reference"):
        if digest(root / plan[name + "_file"]) != plan[name + "_sha256"]:
            raise ValueError(f"Local scratch {name} source changed")
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


def build_bundle(root, directory, num_envs):
    root = Path(root)
    plan = read_plan(root)
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
    worker.update(target_num_envs=num_envs,
        ppo_minibatch_samples=num_envs * worker["num_steps_per_env"] // worker["num_mini_batches"],
        checkpoint_first_update=plan["checkpoint_first_update"])
    for field, origin in plan["curriculum_origins"].items():
        worker[field]["origin_actor_update"] = origin
    marker = dict(schema=SCHEMA, plan_file=PLAN_FILE, plan_sha256=digest(root / PLAN_FILE),
        initialization=plan["initialization"], num_envs=num_envs, seed=plan["seed"])
    worker["local_scratch"] = marker
    budget = dict(schema="v6-local-scratch-budget-v1", initialization=plan["initialization"],
        inherited_consumption=0, update_ceiling=plan["update_ceiling"], num_envs=num_envs,
        num_steps_per_env=worker["num_steps_per_env"],
        transition_ceiling=plan["update_ceiling"] * num_envs * worker["num_steps_per_env"])
    worker.update(budget_spec_file=paths["budget"], budget_spec_sha256=hashlib.sha256(_encode(budget)).hexdigest())
    source_design = json.loads((root / plan["source_design_file"]).read_text())
    row = deepcopy(source_design["stages"][plan["stage"]]["roles"]["train"])
    row["contract"].pop("task_campaign")
    changes = ("target_num_envs", "ppo_minibatch_samples", "checkpoint_first_update",
        "budget_spec_file", "budget_spec_sha256", *plan["curriculum_origins"])
    for key in changes:
        row["contract"][key] = deepcopy(worker[key])
    row["contract"]["local_scratch"] = deepcopy(marker)
    row["invocation"]["num_envs"] = num_envs
    for field, origin in plan["curriculum_origins"].items():
        expected = row["runtime"][field]
        if "origin_actor_update" in expected:
            expected["origin_actor_update"] = origin
        expected["settings"]["origin_actor_update"] = origin
    design = dict(schema=source_design["schema"], status="frozen", scope=source_design["scope"],
        training_release=False, hardware_deployment_ready=False,
        review_basis=dict(source_design_file=plan["source_design_file"],
            source_design_sha256=plan["source_design_sha256"], plan_file=PLAN_FILE,
            plan_sha256=marker["plan_sha256"], changed_contract_paths=list(changes) + ["task_campaign", "local_scratch"]),
        stages={plan["stage"]: {"roles": {"train": row}}})
    worker["design_preflight"] = dict(file=paths["design"],
        sha256=hashlib.sha256(_encode(design)).hexdigest(), stage=plan["stage"])
    return paths, {"worker": worker, "design": design, "budget": budget}


def write_bundle(root, directory, num_envs):
    paths, bundle = build_bundle(root, directory, num_envs)
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
    _, bundle = build_bundle(root, directory, marker.get("num_envs"))
    if config != bundle["worker"]:
        raise ValueError("Local scratch worker differs from its frozen native derivation")
    for name in ("design", "budget"):
        path = Path(root) / directory / (name + ".json")
        if path.read_bytes() != _encode(bundle[name]):
            raise ValueError(f"Local scratch independent {name} changed")
    return bundle["budget"]


def validate_invocation(config, args, root):
    validate_config(config, root)
    plan = read_plan(root)
    # This entry cannot impersonate a continuation or import someone else's
    # optimizer, consumption or curriculum clock.
    if (args.resume or args.transfer or args.consumed_updates is not None
            or args.learning_rate_scale is not None or args.task_actor_update is not None
            or args.stage_actor_update is not None or args.actor_updates_consumed != 0
            or args.global_actor_update_offset != 0 or args.block_index != 0
            or args.confirmed_jump_bins_json != "[]"
            or args.seed != plan["seed"] or args.num_envs != config["target_num_envs"]
            or args.stage != plan["stage"] or not 1 <= args.updates <= plan["update_ceiling"]):
        raise ValueError("Local scratch requires fresh learning state, zero clocks and its frozen invocation")


def source_files(root):
    plan = read_plan(root)
    return [PLAN_FILE, "src/wheeled_tasks/chassis/local_scratch.py", "scripts/train_local_chassis.py",
        plan["source_worker_file"], plan["source_design_file"], plan["reward_reference_file"],
        "src/wheeled_tasks/chassis/nonstandard_start.py", "src/wheeled_tasks/chassis/nonstandard_start_revision.py",
        "src/wheeled_tasks/chassis/release_stop.py", "src/wheeled_tasks/chassis/ordinary_posture_curriculum.py",
        "src/wheeled_tasks/chassis/locomotion_difficulty.py", "src/wheeled_tasks/chassis/graded_locomotion_revision.py"]
