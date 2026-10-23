"""Zero-PPO native equivalence before the position-observation learning pilot."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT/"runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts"),str(REPORTS)]
from iteration045_runner import native, write
from iteration036_compare import read, paired, standing


def main():
    checkpoint = RUNS/"quiet_position056_sealed/model_migrated.pt"
    plan = json.loads((REPORTS/"iteration056_plan.json").read_text())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == plan["checkpoint_sha256"]
    output = RUNS/"position_zero_static056"
    pipeline = dict(schema="position056-native-zero-pipeline-v1",status="running",phase="native_zero_parity",
        started=time.time(), new_macro_updates=0, robust_consumed=1089, long_training_admitted=False)
    write(REPORTS/"iteration056_pipeline.json",pipeline)
    try:
        native("position_zero_static056",[sys.executable,"scripts/evaluate_robust_getup_batch.py",
            "--checkpoint",str(checkpoint),"--output",str(output),"--variants","robust","--replicas","16",
            "--seed","211322","--domain-scale",".35","--friction-min",".3","--seconds","12",
            "--isolated-origins","--case-names","stand_low","stand","stand_high"],output)
        with (REPORTS/"position_zero_static056_analysis.log").open("w") as log:
            subprocess.run([sys.executable,"scripts/analyze_robust_getup.py",str(output/"robust"),"--tensorboard"],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        before, after = read("quiet_velocity_baseline036"),read("position_zero_static056")
        pairing = paired(before,after)
        keys = ("local_position","motor_position","motor_velocity","actions")
        errors = {k:float(np.abs(before[3][k]-after[3][k]).max()) for k in keys}
        keep = list(range(14))+list(range(16,35))
        legacy_error = float(np.abs(before[3]["observations"][:,:,keep]-after[3]["observations"][:,:,keep]).max())
        result = dict(schema="position056-native-zero-equivalence-v1",pairing=pairing,
            maximum_array_errors=errors, maximum_legacy_observation_error=legacy_error,
            old=standing(before),new=standing(after),
            new_position_observation_peak=float(np.abs(after[3]["observations"][:,:,14:16]).max()),
            exact_alive_and_passive=all(np.array_equal(before[3][k],after[3][k]) for k in ("alive","passive")),
            checkpoint_sha256=after[1]["checkpoint_sha256"],trajectory_sha256=after[1]["trajectory_sha256"],
            new_macro_updates=0,robust_consumed=1089,long_training_admitted=False,installed_in_rmcs=False)
        result["native_zero_parity"] = (max(errors.values()) < 2e-4 and legacy_error < 2e-4
            and errors["local_position"] < 1e-5 and result["exact_alive_and_passive"])
        write(REPORTS/"iteration056_final.json",result)
        assert result["native_zero_parity"], result
        pipeline.update(status="completed",phase="zero_equivalence_passed",finished=time.time())
        write(REPORTS/"iteration056_pipeline.json",pipeline)
        print(json.dumps(result),flush=True)
    except BaseException:
        pipeline.update(status="failed",error=traceback.format_exc(),finished=time.time())
        write(REPORTS/"iteration056_pipeline.json",pipeline)
        raise


if __name__ == "__main__":
    main()
