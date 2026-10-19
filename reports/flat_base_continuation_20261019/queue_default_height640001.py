"""Paired 0-PPO source/candidate replay at the authenticated RMCS default height."""
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
REPORT = ROOT / "reports/flat_base_continuation_20261019"
STATE = RUN / "default_height640_queue001.json"
PYTHON = "/home/noir/miniconda3/envs/isaacsim6/bin/python"
if STATE.exists():
    raise SystemExit("Default-height replay already has a receipt")
lock = (RUN / ".default_height640_queue001.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
static_path = REPORT / "default_height_preflight001/evaluation.json"
static = json.loads(static_path.read_text())
if static["status"] != "static_preflight_passed" or static["packaged_default_height"]["height_m"] != .23:
    raise SystemExit("Default-height preflight is incomplete")
hashes = dict(static["source_sha256"])
for name in ("contracts/flat_base_finetune_v1.json", "models/wheel_leg_flat_base_20261018/source.onnx",
             "runs/flat_base_20261019/pilot013/model_000640.pt", "scripts/compare_flat_base.py"):
    hashes[name] = sha(ROOT / name)
state = dict(schema="flat-default-height640-paired-replay-queue-v1", pid=os.getpid(), status="starting",
    started_at=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
    new_optimizer_updates=0, subsequent_training_queued=False, hardware_motion_started=False,
    packaged_default_height=static["packaged_default_height"], static_preflight_sha256=sha(static_path), history=[])


def save():
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2)+'\n'); temp.replace(STATE)


try:
    save()
    previous = json.loads((RUN / "feedback640_queue001.json").read_text())
    if previous["status"] != "complete" or any(Path("/proc/"+str(previous[k])).exists()
            for k in ("pid", "worker_pid")):
        raise RuntimeError("Previous native replay has not finished and exited")
    for phase, extra in (("rmcs_default_source_envelope001", []),
                         ("rmcs_default640_envelope001", ["--checkpoint", str(RUN / "pilot013/model_000640.pt")])):
        if any(sha(ROOT / name) != expected for name,expected in hashes.items()):
            raise ValueError("Frozen default-height replay source or checkpoint changed")
        state.update(status="running", phase=phase)
        with (RUN / (phase+'.log')).open('w') as log:
            worker = subprocess.Popen([PYTHON, "scripts/evaluate_flat_base.py", "--output", str(RUN / phase),
                "--envelope", "--deployment-pd", "--rmcs-command-reference", "--packaged-default-height", *extra],
                cwd=ROOT, env={**os.environ, "OMNI_KIT_ACCEPT_EULA":"YES"}, stdout=log, stderr=subprocess.STDOUT)
            state['worker_pid'] = worker.pid; save()
            code = worker.wait()
        state['history'].append(dict(phase=phase, exit_code=code)); save()
        if code: raise RuntimeError("Default-height native child failed: "+phase)
    sys.path.insert(0, str(ROOT / "scripts"))
    from compare_flat_base import compare
    paths = [RUN / phase / "evaluation.json" for phase in
             ("rmcs_default_source_envelope001", "rmcs_default640_envelope001")]
    a,b = [json.loads(p.read_text()) for p in paths]
    result = compare(a,b)
    result.update(source_evaluation_sha256=sha(paths[0]), candidate_evaluation_sha256=sha(paths[1]),
        packaged_default_height=state['packaged_default_height'], new_optimizer_updates=0,
        behavior_qualified=False, heldout_confirmation_completed=False)
    (REPORT / "comparison_rmcs_default640_envelope001_source.json").write_text(json.dumps(result,indent=2)+'\n')
    state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat()); save()
except Exception:
    state.update(status="failed", error=traceback.format_exc()); save()
    raise
