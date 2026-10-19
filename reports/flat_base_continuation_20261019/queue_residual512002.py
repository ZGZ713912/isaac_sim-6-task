"""At most one more 64-update pilot, conditional on the queued RMCS replay."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/flat_base_20261019"
STATE = RUN / "residual512_queue002.json"
sys.path.insert(0, str(ROOT / "scripts"))
from compare_flat_base import compare

if STATE.exists():
    raise ValueError("Existing queue receipt must be preserved; do not launch twice")
paths = [ROOT / "scripts/iterate_flat_base.py", ROOT / "scripts/train_flat_base.py",
         ROOT / "scripts/evaluate_flat_base.py", ROOT / "scripts/flat_rmcs_command_reference.py",
         ROOT / "contracts/flat_base_finetune_v6.json",
         *(ROOT / "src/wheeled_tasks/chassis").glob("*.py"),
         *(ROOT / "src/wheeled_algo").glob("*.py")]
hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
state = dict(schema="flat-conditional-residual-512-queue-v1", pid=os.getpid(),
    status="waiting_RMCS_command_reference_pair", maximum_additional_optimizer_updates=64,
    source_sha256=hashes, hardware_motion_started=False, long_training_authorized=False)


def save():
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    temp.replace(STATE)


try:
    save()
    deadline = time.monotonic() + 3600
    while True:
        parent = json.loads((RUN / "rmcs_reference_queue001.json").read_text())
        if parent["status"] == "complete":
            break
        if parent["status"] == "failed":
            raise ValueError("The separate RMCS-reference replay did not complete")
        if time.monotonic() >= deadline:
            raise TimeoutError("RMCS-reference replay did not finish within one hour")
        time.sleep(5)
    regressions, comparisons, evidence = [], {}, {}
    for suffix in ("basic", "envelope"):
        pair = [RUN / f"{name}_{suffix}001/evaluation.json" for name in ("rmcs_source", "rmcs448")]
        old, new = [json.loads(p.read_text()) for p in pair]
        comparisons[suffix] = compare(old, new)
        evidence.update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in pair})
        if not comparisons[suffix]["no_lost_baseline_checks"]:
            regressions.append(dict(cohort=suffix, lost_checks=comparisons[suffix]["lost_baseline_checks"]))
        for name, case in new["result"]["cases"].items():
            vx, yaw, _ = case["command"]
            if abs(vx) < .75 and abs(yaw) < 1.5:
                continue
            for metric, limit in (("vx_mae_m_s", .02), ("yaw_mae_rad_s", .05)):
                before, after = old["result"]["cases"][name][metric], case[metric]
                if not math.isfinite(after) or after > before + limit:
                    regressions.append(dict(cohort=suffix, case=name, metric=metric, before=before, after=after))
    state.update(command_reference_comparisons=comparisons, regressions=regressions,
                 evaluation_sha256=evidence)
    if regressions:
        state.update(status="regression_stopped", new_optimizer_updates=0)
        save()
    else:
        for relative, expected in hashes.items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen pilot source changed: " + relative)
        directory = RUN / "residual_iterations448_to512"
        command = ["/home/noir/miniconda3/envs/isaacsim6/bin/python", "scripts/iterate_flat_base.py",
            "--plan", "contracts/flat_base_finetune_v6.json", "--ceiling", "512",
            "--checkpoint", str(RUN / "pilot010/model_000448.pt"),
            "--basic", str(RUN / "pilot010_eval/evaluation.json"),
            "--envelope", str(RUN / "envelope448_001/evaluation.json"),
            "--deployment", str(RUN / "deployment448_001/evaluation.json"), "--output", str(directory)]
        with (RUN / "residual_iterations448_to512.log").open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
                                     stdout=log, stderr=subprocess.STDOUT)
            state.update(status="running_bounded_pilot", worker_pid=child.pid)
            save()
            code = child.wait()
        state["exit_code"] = code
        if code:
            raise RuntimeError("The bounded pilot failed; no retry")
        campaign = json.loads((directory / "campaign.json").read_text())
        state.update(status=campaign["status"], campaign=str(directory / "campaign.json"))
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()
except Exception:
    state.update(status="failed", error=traceback.format_exc())
    save()
    raise
