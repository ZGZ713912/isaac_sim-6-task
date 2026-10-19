"""Single read-only 200 Hz counterpart to the completed 704 RMCS replay."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/flat_base_20261019"
STATE = RUN / "clock704_queue001.json"
if STATE.exists():
    raise SystemExit("This diagnostic already has an owner receipt")
parent = json.loads((RUN / "feedback704_queue001.json").read_text())
if parent["status"] != "complete" or any(Path("/proc/" + str(parent[k])).exists() for k in ("pid", "worker_pid")):
    raise SystemExit("Parent replay must finish and exit before clock isolation")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
hashes = {name: sha(ROOT / name) for name in parent["source_sha256"]}
checkpoint = RUN / "pilot014/model_000704.pt"
if sha(checkpoint) != parent["checkpoint_sha256"]:
    raise SystemExit("Parent checkpoint differs")
state = dict(schema="flat-clock704-diagnostic-queue-v1", pid=os.getpid(), status="starting",
    source_sha256=hashes, checkpoint_sha256=sha(checkpoint), new_optimizer_updates=0,
    started_at=datetime.now(timezone.utc).isoformat(), qualification_eligible=False)


def save():
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(STATE)


try:
    save()
    with (RUN / "rmcs704_pd200_diagnostic001.log").open("w") as log:
        worker = subprocess.Popen(["/home/noir/miniconda3/envs/isaacsim6/bin/python",
            "scripts/evaluate_flat_base.py", "--checkpoint", str(checkpoint),
            "--output", str(RUN / "rmcs704_pd200_diagnostic001"), "--envelope", "--rmcs-command-reference",
            "--diagnostic-ablation", "full", "--diagnostic-seconds", "10", "--diagnostic-training-pd"],
            cwd=ROOT, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}, stdout=log, stderr=subprocess.STDOUT)
        state.update(status="running", worker_pid=worker.pid)
        save()
        code = worker.wait()
    if code:
        raise RuntimeError(f"Diagnostic exited {code}")
    if any(sha(ROOT / name) != value for name, value in hashes.items()):
        raise ValueError("Frozen diagnostic source changed")
    result = RUN / "rmcs704_pd200_diagnostic001/evaluation.json"
    data = json.loads(result.read_text())
    if (data["status"] != "evaluated" or data["checkpoint_sha256"] != sha(checkpoint)
            or data["physics_hz"] != 200 or data["diagnostic_counterfactual"]["qualification_eligible"]):
        raise ValueError("Diagnostic result is incomplete or misidentified")
    state.update(status="complete", evaluation_sha256=sha(result), finished_at=datetime.now(timezone.utc).isoformat())
    save()
except Exception:
    state.update(status="failed", error=traceback.format_exc())
    save()
    raise
