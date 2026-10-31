"""Explicit run overrides; the measured mechanism and acceptance stay frozen."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

TASKS = {f"WheelLeg-ProneMotion-v{v}": f"contracts/wheel_leg_prone_motion_v{v}.json" for v in (1, 2, 3, 4, 5)}
DEFAULT_TASK = "WheelLeg-ProneMotion-v4"
OPTION_KEYS = ("num_envs", "max_iterations", "seed", "save_interval", "learning_rate",
               "domain_randomization", "dr_strength", "num_steps_per_env", "recovery_curriculum",
               "recovery_impact_window_ms", "recovery_force_limits_n", "recovery_impulse_limits_ns",
               "recovery_gate_success_rate", "recovery_gate_every", "recovery_gate_min_trials",
               "curriculum_eval_every", "curriculum_success_rate", "curriculum_required_passes",
               "position_iterations", "velocity_iterations")
LEGACY_OPTION_KEYS = OPTION_KEYS[:8]
LEGACY_RECOVERY_OPTION_KEYS = OPTION_KEYS[:12]
LEGACY_MASTERY_OPTION_KEYS = OPTION_KEYS[:15]
LEGACY_GLOBAL_OPTION_KEYS = OPTION_KEYS[:18]


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _configured(base, options):
    if ((set(options) == set(LEGACY_OPTION_KEYS) and "multi_pose_recovery" not in base)
            or set(options) in (set(LEGACY_RECOVERY_OPTION_KEYS),set(LEGACY_MASTERY_OPTION_KEYS),
                                set(LEGACY_GLOBAL_OPTION_KEYS))):
        options = {**dict.fromkeys(OPTION_KEYS),**options}
    if set(options) != set(OPTION_KEYS):
        raise ValueError("Unknown or incomplete runtime options")
    result = deepcopy(base)
    for option, field in (("num_envs", "num_envs"), ("max_iterations", "max_updates"),
                          ("seed", "seed"), ("save_interval", "snapshot_interval"),
                          ("learning_rate", "learning_rate"), ("num_steps_per_env", "rollout_steps")):
        if options[option] is not None:
            result[field] = options[option]
    for key, minimum in (("num_envs", 20), ("max_updates", 20), ("seed", 0),
                         ("snapshot_interval", 1), ("rollout_steps", 1)):
        if type(result[key]) is not int or result[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if result["seed"] > 2**32-1:
        raise ValueError("seed must fit NumPy's uint32 seed range")
    if result["num_envs"] > 512:
        raise ValueError("This isolated-world task currently supports --num_envs 20..512")
    if any(options[key] is not None for key in ("position_iterations", "velocity_iterations")):
        solver = dict(result.get("solver_iterations", dict(position_iterations=64, velocity_iterations=32)))
        for key, minimum in (("position_iterations", 1), ("velocity_iterations", 0)):
            if options[key] is not None:
                solver[key] = options[key]
            if type(solver[key]) is not int or not minimum <= solver[key] <= 255:
                raise ValueError(f"{key} must be an integer in [{minimum},255]")
        result["solver_iterations"] = solver
    if not math.isfinite(result["learning_rate"]) or result["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    mode, strength = options["domain_randomization"], options["dr_strength"]
    if mode not in ("curriculum", "on", "off") or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("DR mode must be curriculum/on/off; strength must be in [0,1]")
    scale = result["max_updates"]/base["max_updates"]
    for key in ("foundation_until_update", "speed_full_update", "spin_full_update", "height_full_update"):
        result[key] = round(base[key]*scale)
    recovery_options = OPTION_KEYS[8:15]
    if result.get("multi_pose_recovery") is None:
        if any(options[k] is not None for k in recovery_options):
            raise ValueError("Recovery options require an explicit multi-pose recovery plan, included in the current v4 task")
    else:
        from .multi_pose_recovery import validate as validate_recovery
        from .recovery_impact import validate as validate_impact
        for stage in result["multi_pose_recovery"]["curriculum"]:
            stage["start_update"] = round(stage["start_update"]*scale)
        if options["recovery_curriculum"] is not None:
            result["multi_pose_recovery"]["curriculum_mode"] = options["recovery_curriculum"]
        if options["recovery_impact_window_ms"] is not None:
            result["recovery_impact"]["window_seconds"] = options["recovery_impact_window_ms"]/1000.
        for option,field in (("recovery_force_limits_n","force_limits_n"),("recovery_impulse_limits_ns","impulse_limits_ns")):
            if options[option] is not None:
                result["recovery_impact"][field] = deepcopy(options[option])
        for option,field in (("recovery_gate_success_rate","minimum_success_rate"),
                             ("recovery_gate_every","evaluation_interval_updates"),
                             ("recovery_gate_min_trials","minimum_trials_per_leg_class")):
            if options[option] is not None:
                if result["multi_pose_recovery"].get("mastery_gate") is None:
                    raise ValueError("Mastery options require a gated recovery plan")
                result["multi_pose_recovery"]["mastery_gate"][field] = options[option]
        validate_recovery(result["multi_pose_recovery"])
        validate_impact(result["recovery_impact"])
    global_options = (("curriculum_eval_every","evaluation_interval_updates"),
                      ("curriculum_success_rate","minimum_success_rate"),
                      ("curriculum_required_passes","required_consecutive_passes"))
    for option,field in global_options:
        if options[option] is not None:
            if result.get("global_curriculum") is None:raise ValueError("Global gate options need the new v4 curriculum")
            result["global_curriculum"][field] = options[option]
    if result.get("global_curriculum") is not None:
        from .global_curriculum import validate as validate_global
        validate_global(result["global_curriculum"])
    if mode == "off" or strength == 0:
        result["domain_sequence"] = [dict(start_update=0, profile="nominal", strength=0.)]
    elif mode == "on":
        result["domain_sequence"] = [dict(start_update=0, profile="requested", strength=strength)]
    else:
        result["domain_sequence"] = [dict(row, start_update=round(row["start_update"]*scale),
            strength=row["strength"]*strength) for row in base["domain_sequence"]]
        if len(set(row["start_update"] for row in result["domain_sequence"])) != len(result["domain_sequence"]):
            raise ValueError("Budget too short for distinct randomization stages")
    result["training_progression"] = (
        f"One continuous learn({result['max_updates']}) call; snapshots every "
        f"{result['snapshot_interval']} complete updates. Manual checkpoint selection; no resume. "
        "Curriculum update thresholds scale with this run's budget; acceptance limits do not change.")
    if result.get("global_curriculum") is not None:
        result["training_progression"] = (f"One fresh learn({result['max_updates']}); URDF nominal standing first; "
            "cold per-skill/per-level admission, persistent replay and measured domain admission; no elapsed-update unlock or resume.")
    return result


def resolve_plan(base, **overrides):
    from .prone_motion import validate
    base = validate(base)
    if "runtime_cli" in base:
        raise ValueError("Resolve from an original task contract, not a previously resolved run")
    options = dict.fromkeys(OPTION_KEYS)
    options.update(domain_randomization="curriculum", dr_strength=1.)
    if set(overrides)-set(options):
        raise ValueError("Unsupported run overrides: "+str(set(overrides)-set(options)))
    options.update(overrides)
    result = _configured(base, options)
    result["runtime_cli"] = dict(schema="prone-motion-cli-v1", base_plan=base,
        base_plan_sha256=_hash(base), options=options)
    return result


def validate_resolved(plan):
    from .prone_motion import validate
    record = plan["runtime_cli"]
    base = record["base_plan"]
    if "runtime_cli" in base or record["schema"] != "prone-motion-cli-v1" or record["base_plan_sha256"] != _hash(base):
        raise ValueError("Runtime base task identity changed")
    validate(base)
    expected = _configured(base, record["options"])
    expected["runtime_cli"] = deepcopy(record)
    if plan != expected:
        raise ValueError("Resolved run changed outside its declared CLI overrides")
    return deepcopy(plan)


def add_common(parser, *, training):
    parser.add_argument("--python", help="Python executable or launcher selected by the train/play shell entrypoint")
    parser.add_argument("--task", choices=TASKS, default=DEFAULT_TASK if training else None,
                        help="Training defaults to v4 with integrated multi-pose recovery; play uses its checkpoint's original plan")
    parser.add_argument("--plan", type=Path, help="Original task JSON; defaults to --task's contract")
    parser.add_argument("--output", type=Path, help="New output directory (must not already exist)")
    parser.add_argument("--run_name", "--run-name", help="Directory name under runs/prone_motion or runs/prone_motion_play")
    parser.add_argument("--num_envs", "--num-envs", type=int, default=None if training else 1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=training,
                        help="Disable the viewer; --no-headless opens the viewer")
    parser.add_argument("--domain_randomization", "--domain-randomization", choices=("curriculum", "on", "off"),
                        default="curriculum" if training else "off")
    parser.add_argument("--dr_strength", "--dr-strength", type=float, default=1.)
    parser.add_argument("--performance_backend", "--performance-backend", choices=("cuda_graph", "eager"), default="cuda_graph")
    parser.add_argument("--preflight-only", "--preflight_only", action="store_true", help="Validate configuration without Isaac or PPO")
    parser.add_argument("--print-output", action="store_true", help=argparse.SUPPRESS)


def train_parser():
    parser = argparse.ArgumentParser(description="Fresh wheel-leg RL: prone rise, quiet stand, +/-2m/s, spin and turning.")
    add_common(parser, training=True)
    parser.add_argument("--max_iterations", "--max-iterations", "--updates", type=int)
    parser.add_argument("--save_interval", "--save-interval", type=int)
    parser.add_argument("--num_steps_per_env", "--num-steps-per-env", "--rollout-steps", type=int)
    parser.add_argument("--learning_rate", "--learning-rate", type=float)
    parser.add_argument("--position-iterations", "--position_iterations", type=int,
                        help="PhysX position solver iterations per physics step; inherited default64")
    parser.add_argument("--velocity-iterations", "--velocity_iterations", type=int,
                        help="PhysX velocity solver iterations per physics step; inherited default32")
    parser.add_argument("--recovery-curriculum", choices=("staged","all"), default=None)
    parser.add_argument("--recovery-gate-success-rate",type=float,default=None)
    parser.add_argument("--recovery-gate-every",type=int,default=None,help="PPO updates between four-class cold checks")
    parser.add_argument("--recovery-gate-min-trials",type=int,default=None)
    parser.add_argument("--curriculum-eval-every",type=int,default=None,help="Updates between global cold skill reviews")
    parser.add_argument("--curriculum-success-rate",type=float,default=None)
    parser.add_argument("--curriculum-required-passes",type=int,default=None)
    parser.add_argument("--recovery-impact-window-ms", type=float, default=None)
    parser.add_argument("--recovery-force-limits-n", type=json.loads, default=None,
                        help='JSON object with chassis/gimbal/legs/wheels limits in N; null means unmeasured')
    parser.add_argument("--recovery-impulse-limits-ns", type=json.loads, default=None,
                        help='JSON object with chassis/gimbal/legs/wheels fixed-window impulse limits in N.s')
    parser.add_argument("--native-probe", "--native_probe", action="store_true", help="Two-domain native wiring check; zero PPO")
    parser.add_argument("--stop-existing-009", action="store_true", help="Retire only the authenticated old009; never resume it")
    return parser


def output_path(args, root, *, training):
    if args.run_name and (args.run_name in (".", "..") or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name)):
        raise ValueError("run_name must be a single directory name using letters, digits, dot, dash or underscore")
    if args.output is not None:
        return args.output.expanduser().resolve()
    name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    return (root/"runs"/("prone_motion" if training else "prone_motion_play")/name).resolve()


def validate_device(device):
    if not re.fullmatch(r"cuda:\d+", device):
        raise ValueError("This native GPU PhysX task requires --device cuda:N")


def training_options(args, root):
    validate_device(args.device)
    if args.preflight_only and args.native_probe:
        raise ValueError("Choose one probe type")
    path = (args.plan or root/TASKS[args.task]).expanduser().resolve()
    if not path.is_relative_to(root):
        raise ValueError("--plan must be inside the source repository for source freezing")
    base = json.loads(path.read_text())
    if base["schema"] != "wheel-leg-prone-motion-v"+args.task.rsplit("v", 1)[1]:
        raise ValueError("--task and --plan select different task versions")
    plan = resolve_plan(base, **{key: getattr(args, key) for key in OPTION_KEYS})
    return path, plan, output_path(args, root, training=True)
