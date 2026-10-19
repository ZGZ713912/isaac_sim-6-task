"""Serial, read-only native RMCS-command replay after the existing pilot queue."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/flat_base_20261019"
STATE = RUN / "rmcs_reference_queue001.json"
PYTHON = "/home/noir/miniconda3/envs/isaacsim6/bin/python"
source_paths = [ROOT / "scripts/evaluate_flat_base.py", ROOT / "scripts/flat_rmcs_command_reference.py",
                ROOT / "src/wheeled_algo/onnx_actor_import.py", ROOT / "src/wheeled_algo/command_residual.py",
                *(ROOT / "src/wheeled_tasks/chassis").glob("*.py")]
hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
state = dict(schema="flat-rmcs-command-reference-paired-queue-v1", pid=os.getpid(),
    status="waiting_existing_pilot_queue", new_optimizer_updates=0, policy_hz=50, physics_hz=1000,
    source_sha256=hashes, history=[], hardware_deployment_ready=False)


def save():
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(STATE)


try:
    if STATE.exists():
        raise ValueError("This queue already has a receipt; do not launch it twice")
    save()
    deadline = time.monotonic() + 3600
    while True:
        parent = json.loads((RUN / "pilot010_queue.json").read_text())
        if parent["status"] == "complete":
            break
        if parent["status"] in ("failed", "regression_stopped"):
            raise ValueError("Previous pilot queue did not complete")
        if time.monotonic() >= deadline:
            raise TimeoutError("Existing serial queue did not finish within one hour")
        time.sleep(5)
    checkpoint = RUN / "pilot010/model_000448.pt"
    state["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for name, candidate, extended in [("rmcs_source_basic001", False, False), ("rmcs448_basic001", True, False),
                                      ("rmcs_source_envelope001", False, True), ("rmcs448_envelope001", True, True)]:
        for relative, expected in hashes.items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen replay source changed: " + relative)
        command = [PYTHON, "scripts/evaluate_flat_base.py", "--output", str(RUN / name),
                   "--deployment-pd", "--rmcs-command-reference"]
        if candidate:
            command += ["--checkpoint", str(checkpoint)]
        if extended:
            command += ["--envelope"]
        state.update(status="running", phase=name)
        with (RUN / (name + ".log")).open("w") as log:
            worker = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
                                      stdout=log, stderr=subprocess.STDOUT)
            state["worker_pid"] = worker.pid
            save()
            code = worker.wait()
        state["history"].append(dict(phase=name, exit_code=code))
        save()
        if code:
            raise RuntimeError("Native replay failed: " + name)
    state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
    save()
except Exception:
    state.update(status="failed", error=traceback.format_exc())
    save()
    raise
