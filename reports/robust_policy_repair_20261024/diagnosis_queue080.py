"""Serial native diagnosis only; never launches PPO or adopts a model."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
plan = json.loads((BASE / "diagnosis_plan080.json").read_text())
receipt = BASE / "diagnosis_queue080.json"
if receipt.exists():
    raise ValueError("A native diagnosis queue is single use")
state = dict(schema="serial-native-diagnosis-v2", pid=os.getpid(), status="starting",
             jobs=[], new_PPO=0, hardware_started=False, qualification=None)
child = None
stopping = False


def save():
    state["updated_unix"] = time.time()
    tmp = receipt.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(receipt)


def stop(number, frame):
    global stopping
    stopping = True
    state["stop_requested_signal"] = number
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
save()
try:
    for job in plan["jobs"]:
        if stopping:
            raise InterruptedError("Native queue stopped")
        for name, expected in plan["source_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Native queue input/source changed: " + name)
        if (ROOT / job["output"]).exists():
            raise ValueError("Never reuse a native output directory")
        row = dict(job, status="running", started_unix=time.time())
        state["jobs"].append(row)
        state["status"] = "running"
        with (ROOT / job["log"]).open("x") as stream:
            child = subprocess.Popen(job["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", OPENBLAS_NUM_THREADS="1"))
            row["pid"] = child.pid
            save()
            code = child.wait()
            child = None
        if code == 0 and not stopping:
            completion_file = ROOT / job["output"] / "progress.json"
            completion = json.loads(completion_file.read_text()) if completion_file.exists() else {}
            if completion.get("status") != job.get("completion_status", "completed") or (ROOT / job["output"] / "failure.json").exists():
                row.update(status="failed", returncode=code, finished_unix=time.time(),
                           error="Native process returned zero without a completed artifact")
                save()
                raise RuntimeError(row["error"] + ": " + job["name"])
        row.update(status="stopped" if stopping else "completed" if code == 0 else "failed",
                   returncode=code, finished_unix=time.time())
        save()
        if stopping:
            raise InterruptedError("Native queue stopped")
        if code:
            raise RuntimeError("Native job failed: " + job["name"])
        print(job["name"] + " completed", flush=True)
    state["status"] = "completed"
except BaseException as error:
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)
        child.wait()
    state.update(status="stopped" if isinstance(error, InterruptedError) else "failed", error=str(error))
    raise
finally:
    state["finished_unix"] = time.time()
    save()
