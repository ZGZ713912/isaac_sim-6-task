#!/usr/bin/env python3
"""Build an independent full v2.0.1 closed-chain plant from the frozen CAD."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.full_robot_asset import export_bundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_bundle(ROOT, args.output)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files_sha256"}, ensure_ascii=False))
