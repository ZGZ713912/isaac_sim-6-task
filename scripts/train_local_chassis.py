#!/usr/bin/env python3
"""Prepare a fresh local V6 experiment and enter the stock chassis trainer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.local_scratch import PLAN_FILE, read_plan, restore_bundle, validate_parent, write_bundle


def freeze_source(root, directory):
    """Deploy exactly the committed repository, independently of later edits."""
    directory.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    archive = directory / "source.tar.gz"
    with archive.open("wb") as stream:
        subprocess.run(["git", "archive", "--format=tar.gz", commit], cwd=root, stdout=stream, check=True)
    source = directory / "source"
    source.mkdir()
    with tarfile.open(archive, "r:gz") as stream:
        stream.extractall(source, filter="data")
    with archive.open("rb") as stream:
        archive_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = dict(commit=commit, source_archive=str(archive),
        source_archive_sha256=archive_sha,
        frozen_source=str(source))
    (directory / "launch_source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return source


def main():
    plan = read_plan(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, choices=plan["allowed_num_envs"], default=plan["default_num_envs"])
    parser.add_argument("--updates", type=int, default=plan["update_ceiling"])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Finalized checkpoint of this same local experiment; --updates is additional")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--validate-scene", type=int, default=0)
    parser.add_argument("--max-runtime-seconds", type=float, default=172800.)
    args = parser.parse_args()
    if not 1 <= args.updates <= plan["update_ceiling"]:
        parser.error("updates exceed the independent local budget")
    if args.resume:
        from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path
        parent_config = json.loads(checkpoint_contract_path(args.resume).read_text())
        parent = validate_parent(args.resume, parent_config)
        if (parent_config.get("local_scratch", {}).get("plan_file") != PLAN_FILE
                or parent_config["target_num_envs"] != args.num_envs
                or parent + args.updates > plan["update_ceiling"]):
            parser.error("resume must preserve the population/200 Hz plan and cumulative 20000-update ceiling")
    # A launched PPO must come from a committed training version. The scene
    # and static probes can validate a candidate before it is committed.
    if not args.preflight_only and not args.validate_scene:
        pending = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
        if pending:
            parser.error("Commit verified training changes before starting PPO")
        for file in ("scripts/train_local_chassis.py", "src/wheeled_tasks/chassis/local_scratch.py",
                     PLAN_FILE, plan["control_file"]):
            if subprocess.run(["git", "ls-files", "--error-unmatch", file], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
                parser.error("The local training entry and plan must be committed before PPO")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    source = ROOT
    if not args.preflight_only and not args.validate_scene:
        source = freeze_source(ROOT, ROOT / "runs/local_scratch_sources" / stamp)
    config = (restore_bundle(source, args.resume) if args.resume else
        write_bundle(source, source / "runs/local_scratch_configs" / stamp, args.num_envs))
    run_dir = args.run_dir or ROOT / "runs/v6_local_scratch" / stamp
    command = [sys.executable, str(source / "scripts/train_chassis.py"), "--contract", str(config),
        "--stage", plan["stage"], "--num-envs", str(args.num_envs), "--updates", str(args.updates),
        "--seed", str(plan["seed"]), "--device", "cuda:0", "--research", "--run-dir", str(run_dir),
        "--max-runtime-seconds", str(args.max_runtime_seconds)]
    if args.preflight_only:
        command.append("--preflight-only")
    if args.resume:
        command.extend(["--resume", str(args.resume.resolve())])
    if args.validate_scene:
        command.extend(["--validate-scene", str(args.validate_scene)])
    print("Local V6 scratch contract:", config, flush=True)
    print("Timing: 200 Hz policy / 200 Hz physics / decimation 1; 96 steps = 0.48 s per rollout", flush=True)
    print("Training output:", run_dir, flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
