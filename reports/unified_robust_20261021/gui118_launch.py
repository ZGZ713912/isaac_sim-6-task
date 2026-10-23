"""Wait for the current native worker, show GUI, then release its serial runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent


def identity(pid):
    path = Path(f"/proc/{pid}/stat")
    if not path.exists():
        return None
    row = path.read_text().split()
    return dict(start_ticks=row[21], state=row[2])


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--attempt-id", default="001", choices=("001", "002", "003"))
    args = parser.parse_args()
    boundary = json.loads((REPORTS / "gui118_boundary.json").read_text())
    receipt = dict(schema="gui118-launch-v1", status="waiting_existing_native", wrapper_pid=os.getpid(),
        started=time.time(), source_commit=args.source_commit, archive=str(args.source_archive.resolve()),
        archive_sha256=hashlib.sha256(args.source_archive.read_bytes()).hexdigest(),
        viewer_sha256=hashlib.sha256((REPORTS / "gui118_getup.py").read_bytes()).hexdigest(),
        runner_held=boundary["runner_pid"], optimizer_updates=0)
    path = REPORTS / ("gui118_launch.json" if args.attempt_id == "001" else f"gui118_launch{args.attempt_id}.json")
    if path.exists():
        raise ValueError("Do not overwrite an existing GUI launch receipt")
    write(path, receipt)
    while True:
        native = identity(boundary["holdout_native_pid"])
        if native is None or native["state"] == "Z":
            break
        if native["start_ticks"] != boundary["holdout_native_start_ticks"]:
            raise RuntimeError("Native PID has been reused")
        time.sleep(2.)
    progress = ROOT / "runs/unified_robust_20261022/composed_static_holdout115/progress.json"
    if json.loads(progress.read_text())["status"] != "completed":
        raise RuntimeError("Existing native worker did not complete; no automatic GUI retry")
    command = [sys.executable, str(REPORTS / "gui118_getup.py"), "--output", str(args.output.resolve())]
    with (REPORTS / f"gui118_native{args.attempt_id}.log").open("x") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        receipt.update(status="viewer_started", viewer_pid=child.pid, command=command, native_finished_before_gui=True)
        write(path, receipt)
        code = child.wait()
    result = args.output / "report.json"
    status = json.loads(result.read_text()).get("status") if result.exists() else None
    receipt.update(viewer_returncode=code, viewer_status=status, finished=time.time())
    # A failed GUI remains held for diagnosis. A normal close or explicit
    # interruption releases the original queue without changing its source.
    if status in ("closed", "interrupted"):
        runner = identity(boundary["runner_pid"])
        if runner and runner["start_ticks"] == boundary["runner_start_ticks"] and runner["state"] in ("T", "t"):
            os.kill(boundary["runner_pid"], signal.SIGCONT)
            receipt.update(status="viewer_closed_original_queue_released", runner_released=True)
        else:
            receipt.update(status="viewer_closed_runner_identity_rejected", runner_released=False)
    else:
        receipt.update(status="viewer_failed_queue_held", runner_released=False)
    write(path, receipt)


if __name__ == "__main__":
    main()
