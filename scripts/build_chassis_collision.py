#!/usr/bin/env python3
"""Export the canonical source-support-v3 selection without superseded hulls."""
import argparse
import json
from pathlib import Path
from wheeled_tasks.chassis.source_chassis_collision import load

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "model/wheel_leg_v2.0.2/chassis_collision"


def build(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Never overwrite an existing generated asset")
    geometry = load(ROOT)
    output.mkdir(parents=True)
    import os
    parent = ROOT / geometry["parent_usd"]
    (output / "robot.usda").write_text('#usda 1.0\n(\n    defaultPrim = "Robot"\n    subLayers = [@' + os.path.relpath(parent, output) + '@]\n)\n')
    (output / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    build(parser.parse_args().output)
