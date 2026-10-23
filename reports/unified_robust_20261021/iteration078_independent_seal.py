"""Seal descriptive078 data while preserving the failed strict matching receipt."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(REPORTS)]
from iteration036_compare import standing
from iteration045_runner import write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    pipeline = json.loads((REPORTS / "iteration078_pipeline.json").read_text())
    assert pipeline["status"] == "failed" and "Unmatched native pair" in pipeline["error"]
    output = RUNS / "position_release_static078"
    assert json.loads((output / "progress.json").read_text())["status"] == "completed"
    variants = {}
    for directory in sorted(output.iterdir()):
        if not (directory / "analysis.json").exists():
            continue
        report = json.loads((directory / "report.json").read_text())
        analysis = json.loads((directory / "analysis.json").read_text())
        assert sha(directory / "trajectory.npz") == report["trajectory_sha256"]
        rows = standing((directory, report, analysis, np.load(directory / "trajectory.npz")))
        failures = [case for case, row in rows.items() if row["survived"] != row["episodes"]
            or row["full_window_max_excursion_m"] > .05 or row["wheel_clearance_after_02s_peak_m"] > .003]
        variants[directory.name] = dict(standing=rows, absolute_basic_failures=failures,
            trajectory_sha256=report["trajectory_sha256"], analysis_sha256=sha(directory / "analysis.json"))
    assert len(variants) == 3 and all(row["absolute_basic_failures"] for row in variants.values())
    result = dict(schema="quiet078-independent-descriptive-v1", native_completed=True,
        failed_pipeline_sha256=sha(REPORTS / "iteration078_pipeline.json"),
        admissible_pair=False, reason="Different variant blocks draw different enabled domains, reset parameters and materials; strict audit rejected them",
        variants=variants, causal_benefit_proven=False, candidates_rejected=True,
        rejection="Every variant independently violates absolute5cm; no cross-variant improvement claim",
        new_macro_updates=0, robust_consumed=1192, selected_lineage=505, full_training_admitted=False,
        correction="New randomized position grids rejected by CLI; any future random causal test uses equal-index serial native jobs")
    write(REPORTS / "iteration078_independent_result.json", result)
    print(json.dumps({name: row["absolute_basic_failures"] for name, row in variants.items()}), flush=True)


if __name__ == "__main__":
    main()
