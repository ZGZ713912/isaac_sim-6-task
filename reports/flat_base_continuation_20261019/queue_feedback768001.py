"""One 64-actor-update rate pilot, followed by four serial matched cohorts."""
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
STATE = RUN / "feedback768_queue001.json"
PYTHON = "/home/noir/miniconda3/envs/isaacsim6/bin/python"
if STATE.exists() or (RUN / "pilot015").exists():
    raise SystemExit("Pilot already has a receipt or output directory")
lock = (RUN / ".feedback768_queue001.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
revision_path = REPORTS / "training_plan_revision013.json"
revision = json.loads(revision_path.read_text())
hashes = dict(revision["new_source_sha256"])
for name in ("scripts/evaluate_flat_base.py", "scripts/compare_flat_base.py", "contracts/flat_base_finetune_v11.json",
             "reports/flat_base_continuation_20261019/verify_feedback768_native.py", "scripts/flat_default_height_profile.py"):
    hashes[name] = sha(ROOT / name)
state = dict(schema="flat-feedback768-pilot-serial-queue-v1", pid=os.getpid(), status="starting",
    started_at=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
    parent_checkpoint_sha256=revision["parent_checkpoint_sha256"], plan_revision_sha256=sha(revision_path),
    maximum_additional_optimizer_updates=64, new_optimizer_updates=0, subsequent_training_queued=False,
    hardware_motion_started=False, history=[])


def save():
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(STATE)


def run(phase, command):
    if any(sha(ROOT / path) != value for path,value in hashes.items()):
        raise ValueError("Frozen pilot source changed")
    state.update(status="running", phase=phase)
    with (RUN / (phase + ".log")).open("w") as log:
        worker = subprocess.Popen([PYTHON, *command], cwd=ROOT,
            env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}, stdout=log, stderr=subprocess.STDOUT)
        state["worker_pid"] = worker.pid
        save()
        code = worker.wait()
    state["history"].append(dict(phase=phase, exit_code=code))
    progress = RUN / "pilot015/progress.json"
    if progress.exists():
        state["new_optimizer_updates"] = json.loads(progress.read_text())["successful_updates_total"] - 704
    save()
    if code:
        raise RuntimeError("Pilot child failed: " + phase)


try:
    save()
    for name in ("feedback704_queue001.json", "clock704_queue001.json"):
        previous = json.loads((RUN / name).read_text())
        if previous["status"] != "complete" or any(Path("/proc/"+str(previous[k])).exists()
                for k in ("pid","worker_pid")):
            raise ValueError("Previous native owner has not finished and exited")
    for name,value in revision["prerequisite_sha256"].items():
        if sha(REPORTS / name) != value:
            raise ValueError("Completed prerequisite review changed: " + name)
    parent = RUN / "pilot014/model_000704.pt"
    if sha(parent) != state["parent_checkpoint_sha256"]:
        raise ValueError("Authenticated parent checkpoint changed")
    run("pilot015", ["scripts/train_flat_base.py", "--plan", "contracts/flat_base_finetune_v11.json",
        "--output", str(RUN / "pilot015"), "--resume", str(parent),
        "--resume-plan-revision", str(revision_path), "--num-envs", "512", "--updates", "64"])
    run("feedback768_native_verification001", [str(REPORTS / "verify_feedback768_native.py")])
    verification = json.loads((REPORTS / "feedback768_native_milestones.json").read_text())
    if not verification["passed"] or not verification["all_required_milestones_present"]:
        raise ValueError("Actual native rate/restoration/learning milestones failed")
    checkpoint = RUN / "pilot015/model_000768.pt"
    state["checkpoint_sha256"] = sha(checkpoint)
    phases = [("feedback768_basic001", [], "baseline_native002", "feedback704_basic001"),
              ("feedback768_envelope001", ["--envelope"], "envelope_source002", "feedback704_envelope001"),
              ("rmcs768_envelope001", ["--deployment-pd","--rmcs-command-reference","--envelope"], "rmcs_source_envelope001", "rmcs704_envelope001"),
              ("rmcs_default768_envelope001", ["--deployment-pd","--rmcs-command-reference","--packaged-default-height","--envelope"], "rmcs_default_source_envelope001", "rmcs_default704_envelope001")]
    sys.path.insert(0,str(ROOT / "scripts"))
    from compare_flat_base import compare
    for phase,flags,original,previous in phases:
        run(phase,["scripts/evaluate_flat_base.py","--checkpoint",str(checkpoint),"--output",str(RUN/phase),*flags])
        candidate_path=RUN/phase/"evaluation.json"
        candidate=json.loads(candidate_path.read_text())
        if candidate.get("status") != "evaluated" or candidate.get("checkpoint_sha256") != state["checkpoint_sha256"]:
            raise ValueError("Evaluation did not complete with the final768 checkpoint")
        for label,baseline in (("source",original),("parent704",previous)):
            path=RUN/baseline/"evaluation.json"
            result=compare(json.loads(path.read_text()),candidate)
            result.update(baseline_sha256=sha(path),candidate_sha256=sha(candidate_path))
            (REPORTS/("comparison_"+phase+"_"+label+".json")).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    state.update(status="complete",finished_at=datetime.now(timezone.utc).isoformat())
    save()
except Exception:
    state.update(status="failed",error=traceback.format_exc())
    save()
    raise
