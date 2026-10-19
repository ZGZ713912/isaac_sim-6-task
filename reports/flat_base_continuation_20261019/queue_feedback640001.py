"""One 64-update gate-revision pilot, then four serial matched native cohorts."""
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
REPORTS = ROOT / "reports/flat_base_continuation_20261019"
STATE = RUN / "feedback640_queue001.json"
PYTHON = "/home/noir/miniconda3/envs/isaacsim6/bin/python"
if STATE.exists() or (RUN / "pilot013").exists():
    raise SystemExit("Pilot already has a receipt or output directory")
lock = (RUN / ".feedback640_queue001.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
revision_path = REPORTS / "training_plan_revision011.json"
revision = json.loads(revision_path.read_text())
hashes = dict(revision["new_source_sha256"])
for name in ("scripts/evaluate_flat_base.py", "scripts/compare_flat_base.py", "contracts/flat_base_finetune_v9.json",
             "reports/flat_base_continuation_20261019/verify_feedback640_native.py"):
    hashes[name] = sha(ROOT / name)
state = dict(schema="flat-feedback640-pilot-serial-queue-v1", pid=os.getpid(), status="starting",
    started_at=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
    parent_checkpoint_sha256=revision["parent_checkpoint_sha256"], plan_revision_sha256=sha(revision_path),
    maximum_additional_optimizer_updates=64, new_optimizer_updates=0, subsequent_training_queued=False,
    hardware_motion_started=False, history=[])


def save():
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(STATE)


def run(phase, command):
    if any(sha(ROOT / path) != value for path, value in hashes.items()):
        raise ValueError("Frozen pilot source changed")
    state.update(status="running", phase=phase)
    with (RUN / (phase + ".log")).open("w") as log:
        worker = subprocess.Popen([PYTHON, *command], cwd=ROOT,
            env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}, stdout=log, stderr=subprocess.STDOUT)
        state["worker_pid"] = worker.pid
        save()
        code = worker.wait()
    state["history"].append(dict(phase=phase, exit_code=code))
    progress = RUN / "pilot013/progress.json"
    if progress.exists():
        state["new_optimizer_updates"] = json.loads(progress.read_text())["successful_updates_total"] - 576
    save()
    if code:
        raise RuntimeError("Pilot child failed: " + phase)


try:
    save()
    previous = json.loads((RUN / "feedback576_queue001.json").read_text())
    if previous["status"] != "complete" or any(Path("/proc/" + str(previous[k])).exists()
            for k in ("pid", "worker_pid") if k in previous):
        raise ValueError("Previous native work has not finished and exited")
    verification_path = REPORTS / "tracking_gate576_operator_fixture002/verification.json"
    verification = json.loads(verification_path.read_text())
    if not verification["passed"] or verification["parent_checkpoint_sha256"] != state["parent_checkpoint_sha256"]:
        raise ValueError("Actual parent gate/operator proof failed")
    state["operator_verification_sha256"] = sha(verification_path)
    diagnostic = json.loads((RUN / "tracking_gate576_native001/evaluation.json").read_text())
    if diagnostic["status"] != "evaluated" or diagnostic["checkpoint_sha256"] != state["parent_checkpoint_sha256"]:
        raise ValueError("Native gate isolation did not complete with the exact parent")
    parent = RUN / "pilot012/model_000576.pt"
    if sha(parent) != state["parent_checkpoint_sha256"]:
        raise ValueError("Authenticated parent checkpoint changed")
    run("pilot013", ["scripts/train_flat_base.py", "--plan", "contracts/flat_base_finetune_v9.json",
        "--output", str(RUN / "pilot013"), "--resume", str(parent),
        "--resume-plan-revision", str(revision_path), "--num-envs", "512", "--updates", "64"])
    run("feedback640_native_verification001", [str(REPORTS / "verify_feedback640_native.py")])
    verification = json.loads((REPORTS / "feedback640_native_milestones.json").read_text())
    if not verification["passed"] or not verification["all_required_milestones_present"]:
        raise ValueError("Actual native full-state/learning milestones failed")
    checkpoint = RUN / "pilot013/model_000640.pt"
    state["checkpoint_sha256"] = sha(checkpoint)
    phases = [("feedback640_basic001", [], "baseline_native002", "feedback576_basic001"),
              ("feedback640_envelope001", ["--envelope"], "envelope_source002", "feedback576_envelope001"),
              ("rmcs640_basic001", ["--deployment-pd", "--rmcs-command-reference"], "rmcs_source_basic001", "rmcs576_basic001"),
              ("rmcs640_envelope001", ["--deployment-pd", "--rmcs-command-reference", "--envelope"], "rmcs_source_envelope001", "rmcs576_envelope001")]
    sys.path.insert(0, str(ROOT / "scripts"))
    from compare_flat_base import compare
    for phase, flags, original, previous in phases:
        run(phase, ["scripts/evaluate_flat_base.py", "--checkpoint", str(checkpoint),
                    "--output", str(RUN / phase), *flags])
        candidate_path = RUN / phase / "evaluation.json"
        candidate = json.loads(candidate_path.read_text())
        for label, baseline in (("source", original), ("parent576", previous)):
            path = RUN / baseline / "evaluation.json"
            result = compare(json.loads(path.read_text()), candidate)
            result.update(baseline_sha256=sha(path), candidate_sha256=sha(candidate_path))
            (REPORTS / ("comparison_"+phase+"_"+label+".json")).write_text(
                json.dumps(result, indent=2, allow_nan=False)+"\n")
    state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
    save()
except Exception:
    state.update(status="failed", error=traceback.format_exc())
    save()
    raise
