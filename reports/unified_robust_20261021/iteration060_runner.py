"""Matched native test; only high position weight differs from059."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
REPORTS=Path(__file__).parent
RUNS=ROOT/"runs/unified_robust_20261022"
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts"),str(REPORTS)]
from iteration045_runner import native,write
from iteration036_compare import read,paired,standing
from iteration048_runner import static_failures


def main():
    checkpoint=RUNS/"quiet_high_transfer060/model_migrated.pt"
    migration=json.loads((REPORTS/"iteration060_migration.json").read_text())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==migration["checkpoint_sha256"]
    output=RUNS/"high_transfer_static060"
    progress=native("high_transfer_static060",[sys.executable,"scripts/evaluate_robust_getup_batch.py",
        "--checkpoint",str(checkpoint),"--output",str(output),"--variants","robust","--replicas","16",
        "--seed","211322","--domain-scale",".35","--friction-min",".3","--seconds","12",
        "--isolated-origins","--case-names","stand_low","stand","stand_high"],output)
    assert progress["status"]=="completed"
    with (REPORTS/"high_transfer_static060_analysis.log").open("w") as log:
        subprocess.run([sys.executable,"scripts/analyze_robust_getup.py",str(output/"robust"),"--tensorboard"],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    baseline,parent,after=read("quiet_velocity_baseline036"),read("low_transfer_static059"),read("high_transfer_static060")
    match=paired(parent,after)
    untouched=[r["lane"] for r in after[1]["lanes"] if r["name"]!="stand_high"]
    errors={k:float(np.abs(parent[3][k][:,untouched]-after[3][k][:,untouched]).max())
            for k in ("observations","local_position","motor_position","motor_velocity","actions")}
    assert max(errors.values())==0.,errors
    rows=standing(after)
    failures=static_failures(rows,standing(baseline))
    result=dict(schema="position060-high-transfer-native-v1",pairing_to059=match,pairing_to036=paired(baseline,after),
        before=standing(parent),after=rows,baseline=standing(baseline),untouched_low_mid_array_errors=errors,
        high_target_passed=not any(x.startswith("stand_high:") for x in failures),static_failures=failures,
        checkpoint_sha256=after[1]["checkpoint_sha256"],trajectory_sha256=after[1]["trajectory_sha256"],
        new_macro_updates=0,robust_consumed=1128,lineage_clock=473,long_training_admitted=False,
        scope="Single high-head transfer; low/mid remain059 exactly, including unresolved mid-height vibration")
    write(REPORTS/"iteration060_final.json",result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
