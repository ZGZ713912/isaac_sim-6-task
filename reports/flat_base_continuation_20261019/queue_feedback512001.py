"""One authenticated 64-update pilot, followed by serial frozen evaluations."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/flat_base_20261019"
STATE = RUN / "feedback512_queue001.json"
PYTHON = "/home/noir/miniconda3/envs/isaacsim6/bin/python"
if STATE.exists() or (RUN / "pilot011").exists():
    raise SystemExit("Pilot already has a receipt or output directory")
lock = (RUN / ".feedback512_queue001.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
revision_path = ROOT / "reports/flat_base_continuation_20261019/training_plan_revision009.json"
revision = json.loads(revision_path.read_text())
hashes = dict(revision["new_source_sha256"])
for path in ("scripts/evaluate_flat_base.py", "scripts/compare_flat_base.py",
             "contracts/flat_base_finetune_v7.json"):
    hashes[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
state = dict(schema="flat-feedback-512-pilot-serial-queue-v1", pid=os.getpid(), status="starting",
    started_at=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
    parent_checkpoint_sha256=revision["parent_checkpoint_sha256"],
    plan_revision_sha256=hashlib.sha256(revision_path.read_bytes()).hexdigest(),
    maximum_additional_optimizer_updates=64, new_optimizer_updates=0,
    subsequent_training_queued=False, hardware_motion_started=False, history=[])


def save():
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(STATE)


def verify_sources():
    for path, sha in hashes.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha:
            raise ValueError("Frozen pilot source changed: " + path)


def run(phase, command):
    verify_sources()
    state.update(status="running", phase=phase)
    with (RUN / (phase + ".log")).open("w") as log:
        worker = subprocess.Popen([PYTHON, *command], cwd=ROOT,
            env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}, stdout=log, stderr=subprocess.STDOUT)
        state["worker_pid"] = worker.pid
        save()
        code = worker.wait()
    state["history"].append(dict(phase=phase, exit_code=code))
    if (RUN / "pilot011/progress.json").exists():
        state["new_optimizer_updates"] = json.loads((RUN / "pilot011/progress.json").read_text())["successful_updates_total"] - 448
    save()
    if code:
        raise RuntimeError("Pilot child failed: " + phase)


try:
    save()
    for previous in ("pilot010_queue.json", "rmcs_reference_queue001.json"):
        if json.loads((RUN / previous).read_text())["status"] != "complete":
            raise ValueError("Previous native queue must finish before this single-GPU pilot")
    parent = RUN / "pilot010/model_000448.pt"
    if hashlib.sha256(parent.read_bytes()).hexdigest() != state["parent_checkpoint_sha256"]:
        raise ValueError("Authenticated parent checkpoint changed")
    run("pilot011", ["scripts/train_flat_base.py", "--plan", "contracts/flat_base_finetune_v7.json",
        "--output", str(RUN / "pilot011"), "--resume", str(parent),
        "--resume-plan-revision", str(revision_path), "--num-envs", "512", "--updates", "64"])
    checkpoint = RUN / "pilot011/model_000512.pt"
    state["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    phases = [("feedback512_basic001", [], "baseline_native002", "pilot010_eval"),
              ("feedback512_envelope001", ["--envelope"], "envelope_source002", "envelope448_001"),
              ("rmcs512_basic001", ["--deployment-pd", "--rmcs-command-reference"], "rmcs_source_basic001", "rmcs448_basic001"),
              ("rmcs512_envelope001", ["--deployment-pd", "--rmcs-command-reference", "--envelope"], "rmcs_source_envelope001", "rmcs448_envelope001")]
    for phase, flags, original, previous in phases:
        run(phase, ["scripts/evaluate_flat_base.py", "--checkpoint", str(checkpoint),
                    "--output", str(RUN / phase), *flags])
        sys.path.insert(0, str(ROOT / "scripts"))
        from compare_flat_base import compare
        candidate = json.loads((RUN / phase / "evaluation.json").read_text())
        for label, baseline in (("source", original), ("parent448", previous)):
            path = RUN / baseline / "evaluation.json"
            result = compare(json.loads(path.read_text()), candidate)
            result.update(baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                candidate_sha256=hashlib.sha256((RUN / phase / "evaluation.json").read_bytes()).hexdigest())
            (ROOT / "reports/flat_base_continuation_20261019" / ("comparison_"+phase+"_"+label+".json")).write_text(
                json.dumps(result, indent=2, allow_nan=False)+"\n")
    state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
    save()
except Exception:
    state.update(status="failed", error=traceback.format_exc())
    save()
    raise
