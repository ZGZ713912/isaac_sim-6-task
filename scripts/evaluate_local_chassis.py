#!/usr/bin/env python3
"""Compare local 200 Hz checkpoints on the pinned eight-case pilot exam."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from wheeled_tasks.chassis.local_evaluation import write_bundle
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traces", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    config = write_bundle(ROOT, ROOT / "runs/local_pilot_exams" / stamp)
    command = [sys.executable, str(ROOT / "scripts/evaluate_chassis.py"),
        "--contract", str(config), "--output", str(args.output), "--role", "quick"]
    for checkpoint in args.checkpoint:
        command.extend(["--checkpoint", str(checkpoint.resolve())])
    if not args.traces:
        command.append("--no-traces")
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
