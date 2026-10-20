"""A bounded local pilot exam, derived from authenticated native case/design tables.

This eight-case diagnostic is not the campaign's full qualification exam.
Case definitions, thresholds, replica counts and canonical placements are pinned.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .local_scratch import PLAN_FILE, _encode, bundle_paths, digest, read_plan

CASES = ("stand_230mm", "stand_305mm", "stand_430mm", "forward_05", "backward_05",
         "rotate_1", "rotate_1_reverse", "start_stop_05")


def build_bundle(root, directory):
    from .evaluation import fixed_suite_contract, select_evaluation_cases
    root = Path(root)
    plan = read_plan(root)
    paths = bundle_paths(root, directory)
    native = json.loads((root / plan["source_worker_file"]).read_text())
    source_design = json.loads((root / plan["source_design_file"]).read_text())
    # Frozen roles define the unassisted simulator; the native train row pins
    # the latest catalogue including the 230/430 mm standing cases.
    row = deepcopy(source_design["stages"]["speed"]["roles"]["quick"])
    config = fixed_suite_contract(native)
    settings = select_evaluation_cases(
        source_design["stages"]["speed"]["roles"]["train"]["contract"]["evaluation"], CASES)
    if settings["episodes_per_case"] != 4:
        raise ValueError("Local pilot replica contract changed")
    config["evaluation"] = settings
    config["scene_groups"] = [{"name": c["name"], "fraction": 1 / len(CASES),
        "terrain": [c.get("terrain", "flat")]} for c in settings["cases"]]
    config["episode_seconds"] = settings["episode_seconds"]
    config.pop("task_campaign")
    marker = dict(schema="v6-local-pilot-evaluation-v1", plan_file=PLAN_FILE,
        plan_sha256=digest(root / PLAN_FILE), scope="eight_case_learning_diagnostic_not_qualification")
    config["local_evaluation"] = marker
    temporal = plan["ppo_timing"]
    overrides = dict(physics_dt=plan["timing"]["physics_dt"], policy_dt=plan["timing"]["policy_dt"],
        control_math_source=plan["control_file"], control_math_sha256=plan["control_sha256"],
        num_steps_per_env=temporal["num_steps_per_env"], gamma=temporal["gamma"], lam=temporal["lam"],
        action_difference_reference_dt=plan["action_difference_reference_dt"])
    config.update(overrides)
    expected = row["contract"]
    expected.pop("task_campaign")
    for key in ("evaluation", "scene_groups", "episode_seconds", "local_evaluation", *overrides):
        expected[key] = deepcopy(config[key])
    expected["evaluation.seed"] = settings["seed"]
    expected["evaluation.confirmation_seed"] = settings["confirmation_seed"]
    row["invocation"]["num_envs"] = len(CASES) * settings["episodes_per_case"]
    row["invocation"]["seed"] = dict(rule="fixed", value=settings["seed"])
    control = json.loads((root / plan["control_file"]).read_text())
    row["control"]["timing"] = deepcopy(plan["timing"])
    row["control"]["simulation_timing_approximation"] = deepcopy(control["simulation_timing_approximation"])
    row["runtime"].update(plan["timing"])
    row["runtime"]["action_difference_reference_dt"] = plan["action_difference_reference_dt"]
    for field in ("finite_passive_landing", "start_stop_timing", "nonstandard_start", "release_stop"):
        row["runtime"][field]["policy_dt"] = plan["timing"]["policy_dt"]
    design = dict(schema=source_design["schema"], status="frozen", scope=marker["scope"],
        training_release=False, hardware_deployment_ready=False,
        review_basis=dict(source_design_file=plan["source_design_file"],
            source_design_sha256=plan["source_design_sha256"], case_names=list(CASES),
            scope=marker["scope"], plan_file=PLAN_FILE, plan_sha256=marker["plan_sha256"]),
        stages={"speed": {"roles": {"quick": row}}})
    config["design_preflight"] = dict(file=paths["design"],
        sha256=hashlib.sha256(_encode(design)).hexdigest(), stage="speed")
    return paths, dict(worker=config, design=design)


def write_bundle(root, directory):
    paths, bundle = build_bundle(root, directory)
    (Path(root) / Path(paths["worker"]).parent).mkdir(parents=True, exist_ok=False)
    for name, value in bundle.items():
        (Path(root) / paths[name]).write_bytes(_encode(value))
    return Path(root) / paths["worker"]


def validate_config(config, root):
    directory = Path(config["design_preflight"]["file"]).parent
    paths, bundle = build_bundle(root, directory)
    if config != bundle["worker"] or (Path(root) / paths["design"]).read_bytes() != _encode(bundle["design"]):
        raise ValueError("Local pilot exam differs from its independent frozen derivation")
    return dict(scope=bundle["worker"]["local_evaluation"]["scope"], training_allowed=False)


def log_result(result, checkpoint):
    """Attach diagnostics to the candidate's existing RSL-RL event directory."""
    from torch.utils.tensorboard import SummaryWriter
    checkpoint = Path(checkpoint)
    directory = checkpoint.parent.parent if checkpoint.parent.name == "initial_state" else checkpoint.parent
    step = result["checkpoint_updates"]
    metrics = ("survival_rate", "height_mae_m", "vx_mae_m_s", "yaw_mae_rad_s",
        "planar_speed_mean_m_s", "closure_gap_max_m", "terminal_episode_time_mean_s")
    with SummaryWriter(str(directory)) as writer:
        writer.add_scalar("LocalEvaluation/passed_cases", sum(c["passed"] for c in result["cases"].values()), step)
        for name, case in result["cases"].items():
            for metric in metrics:
                if metric in case:
                    writer.add_scalar(f"LocalEvaluation/{name}/{metric}", case[metric], step)
