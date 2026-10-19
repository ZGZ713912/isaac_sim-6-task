#!/usr/bin/env python3
"""Extract a trainable actor mean without inventing a full PPO checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-hz", type=float, required=True,
                        help="User-provided policy rate; it cannot be recovered from this graph")
    args = parser.parse_args()
    if args.policy_hz <= 0 or not math.isfinite(args.policy_hz):
        parser.error("policy-hz must be finite and positive")
    if args.output.exists():
        parser.error("Use a new output directory to preserve existing artifacts")
    import torch
    from wheeled_algo.onnx_actor_import import FORMAT, recover_actor, verify_outputs

    torch.set_num_threads(1)
    source = args.source.resolve(strict=True)
    actor, report = recover_actor(source)
    report["verification"] = verify_outputs(actor, source)
    report.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "source_path": str(source), "policy_hz": args.policy_hz,
                   "policy_hz_evidence": "user_provided", "policy_dt": 1.0 / args.policy_hz,
                   "full_checkpoint": False, "behavior_verified_in_simulator": False,
                   "training_ready": False,
                   "remaining": ["confirm observation/action semantics",
                                 "match the robot model and control contract",
                                 "verify unchanged baseline behavior before PPO"]})
    args.output.mkdir(parents=True)
    shutil.copy2(source, args.output / "source.onnx")
    if hashlib.sha256((args.output / "source.onnx").read_bytes()).hexdigest() != report["source_sha256"]:
        raise ValueError("Source changed during recovery")
    torch.save({"format": FORMAT, "source_sha256": report["source_sha256"],
                "actor_mean_state_dict": actor.state_dict(), "policy_hz": args.policy_hz},
               args.output / "actor_mean.pt")
    torch.jit.script(actor).save(str(args.output / "actor_mean.torchscript.pt"))
    report["artifacts"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted(args.output.iterdir())}
    (args.output / "inspection.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "parameters": report["parameter_count"],
                      "verification": report["verification"], "full_checkpoint": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
