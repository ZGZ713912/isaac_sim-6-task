"""Serial zero-optimizer branch isolation, with exact native replay sources."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/flat_base_20261019"
STATE = RUN / "ablation512_queue001.json"
if STATE.exists():
    raise SystemExit("Ablation already has a receipt")
lock = (RUN / ".ablation512_queue001.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
paths = list((ROOT / "src/wheeled_tasks/chassis").glob("*.py"))
paths += list((ROOT / "src/wheeled_algo").glob("*.py"))
paths += [ROOT / "scripts/evaluate_flat_base.py", ROOT / "scripts/flat_rmcs_command_reference.py"]
hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
state = dict(schema="flat-feedback-native-ablation-queue-v1", pid=os.getpid(), status="starting",
    started_at=datetime.now(timezone.utc).isoformat(), new_optimizer_updates=0,
    qualification_eligible=False, subsequent_training_queued=False, source_sha256=hashes, history=[])


def save():
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATE)


try:
    for name in ("feedback512_queue001.json", "feedback512_birth_anchor_audit001/audit.json"):
        previous = json.loads((RUN / name).read_text())
        if previous["status"] not in ("complete", "audited"):
            raise ValueError("Previous native work is unfinished")
        if any(Path("/proc/" + str(previous[k])).exists() for k in ("pid", "worker_pid") if k in previous):
            raise ValueError("Previous native work has a live process")
    checkpoint = RUN / "pilot011/model_000512.pt"
    parent = RUN / "pilot010/model_000448.pt"
    if (sha(checkpoint) != "f86aa34c16b0de0ee20c7d8ff5b7fb65978672b999c4adea8f71fa29a21382b5"
            or sha(parent) != "334bbee9a258a55d03be273f0c86d356b58ddcca6a8eddd4c1ebb6ad1c1ceaa2"):
        raise ValueError("Authenticated source checkpoint changed")
    save()
    for variant in ("full", "feedback_off", "parent_command"):
        if any(sha(ROOT / p) != value for p, value in hashes.items()):
            raise ValueError("Ablation execution source changed")
        phase = "ablation512_" + variant + "001"
        command = ["/home/noir/miniconda3/envs/isaacsim6/bin/python", "scripts/evaluate_flat_base.py",
            "--checkpoint", str(checkpoint), "--output", str(RUN / phase),
            "--envelope", "--deployment-pd", "--rmcs-command-reference",
            "--diagnostic-ablation", variant]
        if variant == "parent_command":
            command += ["--diagnostic-parent", str(parent)]
        state.update(status="running", phase=phase)
        with (RUN / (phase + ".log")).open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
                stdout=log, stderr=subprocess.STDOUT)
            state["worker_pid"] = child.pid
            save()
            code = child.wait()
        state["history"].append(dict(phase=phase, exit_code=code))
        save()
        if code:
            raise RuntimeError("Native diagnostic failed: " + phase)
    state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
    save()
except Exception:
    state.update(status="failed", error=traceback.format_exc())
    save()
    raise
