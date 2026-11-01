#!/usr/bin/env python3
"""Admit one fresh full Wheelbipe run, export ONNX, and retain final diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from wheeled_tasks.wheelbipe_port.training_contract import constraint_limit_m,check_constraint_gap


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-receipt",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--num-envs",type=int,default=512)
    parser.add_argument("--seed",type=int,default=109)
    parser.add_argument("--evaluation-steps",type=int,default=500)
    parser.add_argument("--physics-hz",type=int,choices=(200,500,1000,2000),default=200)
    parser.add_argument("--max-constraint-gap-mm",type=float,default=5.)
    args = parser.parse_args()
    try:
        gap_limit = constraint_limit_m(args.max_constraint_gap_mm)
    except ValueError as error:
        parser.error(str(error))
    if args.num_envs<1 or args.evaluation_steps<1:
        parser.error("Environment and evaluation budgets must be positive")
    # One experiment owner only; this never signals another task's process.
    lock = open(args.output.parent/"experiment.lock","a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pilot = json.loads(args.pilot_receipt.read_text())
    if pilot["status"]!="training_complete" or pilot["completed_updates"]<30:
        raise ValueError("The genuine fresh GPU pilot has not completed")
    if pilot["arguments"]["device"]!="cuda:0" or pilot["arguments"]["task"]!="flat":
        raise ValueError("The admission must come from this flat GPU port")
    settings = pilot["arguments"]
    if (settings["randomization_profile"]!="requested" or settings["randomization_strength"]!=1.
            or settings["solver_position_iterations"]!=64 or settings["solver_velocity_iterations"]!=32
            or settings["external_forces_every_iteration"]):
        raise ValueError("The admitted physical/randomization settings differ from the full run")
    frequency = pilot["arguments"]["physics_hz"]
    if frequency!=args.physics_hz or pilot["native"]["policy_dt_s"]!=.02:
        raise ValueError("Do not silently migrate pilot/checkpoint integration clocks")
    if (pilot["arguments"]["num_envs"]!=args.num_envs
            or pilot.get("constraint_gap_limit_m",.003)!=gap_limit):
        raise ValueError("The pilot population or declared loop limit differs from this full run")
    check_constraint_gap(pilot["max_constraint_gap_m"],gap_limit,"Pilot")
    if (not pilot["policy_tensors_finite"]
            or pilot["fresh_initial_optimizer_state_count"]!=0
            or pilot["final_optimizer_state_count"]!=17):
        raise ValueError("Pilot physics, fresh state or genuine PPO checks failed")
    for relative,expected in pilot["implementation_files_sha256"].items():
        if digest(ROOT/relative)!=expected:
            raise ValueError("Full source differs from the admitted pilot: "+relative)
    if digest(ROOT/"model/wheelbipe_port_v202/collision_support.json")!=pilot["collision_support_sha256"]:
        raise ValueError("Full reset collision geometry differs from the admitted pilot")
    if digest(ROOT/"model/wheelbipe_port_v202/ground_plane.usda")!=pilot["ground_plane_sha256"]:
        raise ValueError("Full ground geometry differs from the admitted pilot")
    if digest(pilot["latest_checkpoint"])!=pilot["checkpoint_sha256"]:
        raise ValueError("Pilot checkpoint differs from its completed receipt")
    pilot_onnx = Path(pilot["latest_checkpoint"]).parent/"onnx/contract.json"
    parity = json.loads(pilot_onnx.read_text())
    if parity["checkpoint_sha256"]!=pilot["checkpoint_sha256"] or parity["max_absolute_error"]>1.e-5:
        raise ValueError("Pilot ONNX export/parity has not passed")
    if digest(pilot_onnx.parent/"policy.onnx")!=parity["onnx_sha256"]:
        raise ValueError("The pilot ONNX file differs from its parity receipt")
    args.output.mkdir(parents=True,exist_ok=False)
    status = dict(schema="wheelbipe-v202-full-campaign-v1",status="admitted",
        source=str(ROOT),started_at=datetime.now(timezone.utc).isoformat(),
        pilot_receipt=str(args.pilot_receipt.resolve()),pilot_receipt_sha256=digest(args.pilot_receipt),
        owner_pid=os.getpid(),full_updates=20000,num_envs=args.num_envs,seed=args.seed,
        policy_hz=50,physics_hz=frequency,constraint_gap_limit_m=gap_limit,source_unchanged_from_pilot=True,
        fresh_full_policy=True,hardware_io=False,behavior_qualified=False)
    path = args.output/"campaign.json"
    write(path,status)
    process = None
    def terminate(signum,frame):
        raise KeyboardInterrupt("Campaign stop requested")
    signal.signal(signal.SIGTERM,terminate)

    def run(stage,command,receipt=None):
        nonlocal process
        log = args.output/(stage+".log")
        with log.open("w") as output:
            process = subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
            status.update(status=stage,worker_pid=process.pid,command=command,log=str(log))
            write(path,status)
            terminal_at = None
            while process.poll() is None:
                time.sleep(5.)
                if receipt is None or not receipt.is_file():
                    continue
                value = json.loads(receipt.read_text())
                if value["status"] in ("failed","preflight_complete","training_complete","evaluation_complete"):
                    terminal_at = time.monotonic() if terminal_at is None else terminal_at
                    if time.monotonic()-terminal_at>30.:
                        # All durable writers have completed; bound the known
                        # WSL graphics-only shutdown issue on this owned child.
                        process.terminate()
                        try:
                            process.wait(timeout=5.)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        status[stage+"_bounded_shutdown"] = True
            if receipt is not None:
                result = json.loads(receipt.read_text())
                if result["status"] not in ("preflight_complete","training_complete","evaluation_complete"):
                    raise RuntimeError(f"{stage} did not complete: {result.get('error')}")
            elif process.returncode:
                raise RuntimeError(f"{stage} exited {process.returncode}; see {log}")
        status[stage+"_exit_code"] = process.returncode
        write(path,status)
        process = None

    common = [sys.executable,"-u",str(ROOT/"scripts/train_wheelbipe_port.py"),
        "--task","flat","--device","cuda:0","--physics-hz",str(frequency),
        "--solver-position-iterations","64","--solver-velocity-iterations","32",
        "--no-external-forces-every-iteration","--randomization-profile","requested",
        "--max-constraint-gap-mm",str(args.max_constraint_gap_mm)]
    try:
        rough = args.output/"rough_preflight"
        rough_command = common.copy()
        rough_command[rough_command.index("flat")] = "rough"
        run("rough_preflight",rough_command+["--mode","preflight","--num-envs","8",
            "--steps","200","--seed","111","--output",str(rough)],rough/"receipt.json")
        # Exercise the full clone population and memory footprint before PPO.
        preflight = args.output/"population_preflight"
        run("population_preflight",common+["--mode","preflight","--num-envs",str(args.num_envs),
            "--steps","100","--seed",str(args.seed),"--output",str(preflight)],preflight/"receipt.json")
        full = args.output/"train"
        run("training",common+["--mode","train","--num-envs",str(args.num_envs),
            "--iterations","20000","--save-interval","500","--seed",str(args.seed),
            "--output",str(full)],full/"receipt.json")
        trained = json.loads((full/"receipt.json").read_text())
        if trained["completed_updates"]!=20000:
            raise ValueError("The original full update budget was not completed")
        if trained["initial_checkpoint_sha256"]==pilot["initial_checkpoint_sha256"]:
            raise ValueError("The full experiment unexpectedly reused the pilot initialization")
        checkpoint = Path(trained["latest_checkpoint"])
        onnx = args.output/"onnx"
        run("export",[sys.executable,str(ROOT/"scripts/export_wheelbipe_port_onnx.py"),
            "--checkpoint",str(checkpoint),"--output",str(onnx)])
        contract = json.loads((onnx/"contract.json").read_text())
        if contract["completed_updates"]!=20000 or contract["max_absolute_error"]>1.e-5:
            raise ValueError("Final ONNX full-budget/parity guard failed")
        shutil.copytree(full/"params",onnx/"params")
        status.update(status="exported",onnx=str(onnx/"policy.onnx"),
            onnx_sha256=contract["onnx_sha256"],checkpoint=str(checkpoint))
        write(path,status)
        evaluation = args.output/"cold_evaluation"
        evaluation_command = common.copy()
        evaluation_command[evaluation_command.index("requested")] = "nominal"
        run("cold_evaluation",evaluation_command+["--mode","evaluate","--checkpoint",str(checkpoint),
            "--num-envs","32","--steps",str(args.evaluation_steps),"--seed","110",
            "--output",str(evaluation)],evaluation/"receipt.json")
        requirements = json.loads((ROOT/"contracts/wheel_leg_acceptance_requirements_v2.json").read_text())
        capabilities = [dict(id=row["id"],requirement=row["requirement"],status="not_qualified",
            evidence="Only cold command diagnostics; complete scope/measurement requirements have not been evaluated")
            for row in requirements["capabilities"]]
        write(args.output/"capability_status.json",dict(checkpoint_sha256=trained["checkpoint_sha256"],
            requirements_sha256=digest(ROOT/"contracts/wheel_leg_acceptance_requirements_v2.json"),
            diagnostic_evaluation=str(evaluation/"receipt.json"),capabilities=capabilities))
        status.update(status="complete",finished_at=datetime.now(timezone.utc).isoformat(),
            evaluation=str(evaluation/"receipt.json"),
            acceptance="Diagnostic cold cases only; permanent capability acceptance remains separate")
        write(path,status)
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        status.update(status="failed",error=repr(error),traceback=traceback.format_exc())
        write(path,status)
        raise


if __name__=="__main__":
    main()
