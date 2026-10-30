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

TASKS = {f"WheelLeg-ProneMotion-v{v}": f"contracts/wheel_leg_prone_motion_v{v}.json" for v in (1, 2, 3, 4)}
DEFAULT_TASK = "WheelLeg-ProneMotion-v4"
OPTION_KEYS = ("num_envs", "max_iterations", "seed", "save_interval", "learning_rate",
               "domain_randomization", "dr_strength", "num_steps_per_env")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _configured(base, options):
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
    if result["num_envs"] > 256:
        raise ValueError("This isolated-world task currently supports --num_envs 20..256")
    if not math.isfinite(result["learning_rate"]) or result["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    mode, strength = options["domain_randomization"], options["dr_strength"]
    if mode not in ("curriculum", "on", "off") or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("DR mode must be curriculum/on/off; strength must be in [0,1]")
    scale = result["max_updates"]/base["max_updates"]
    for key in ("foundation_until_update", "speed_full_update", "spin_full_update", "height_full_update"):
        result[key] = round(base[key]*scale)
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
                        help="Training defaults to v4; play infers the task from its checkpoint unless explicitly selected")
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
