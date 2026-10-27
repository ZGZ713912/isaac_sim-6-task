from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
ROOT=Path("/home/noir/Documents/workspace/isaac_wheeled_rl_train")
REPORT=ROOT/"reports/robust_policy_repair_20261024"
BASE=ROOT/"runs/scratch_standing_20261026"
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
profiles={"nominal":"eval009_v2_nominal_seed261201", "requested":"eval009_v2_requested_seed261201", "without_alignment_paired":"eval009_v2_without_alignment_paired_seed261201"}
rows=[]; raw={}; receipts={}; evidence={}
for profile,name in profiles.items():
    run=BASE/name
    receipt=read(run/"receipt.json"); result=read(run/"evaluation/result.json"); review=read(REPORT/(name+"_review001.json"))
    assert receipt["status"]=="completed_unqualified" and receipt["new_PPO"]==0 and not receipt["optimizer_constructed"]
    assert sha(run/"evaluation/native_trace.npz")==result["trace_sha256"]==review["native_trace_sha256"]
    assert sha(run/"evaluation/result.json")==review["original_result_sha256"]
    assert review["RSL_events"]["all_values_read_back"]
    mid=next(g for g in review["groups"] if g["height_m"]==.305)
    selected=[r for r in result["cases"] if r["height_m"]==.305]
    rows.append(dict(profile=profile,mid=mid,all_heights_standing_pass=review["standing_pass_v2_count"],
        forward_speed_rms_mean_m_s=sum(r["forward_speed_rms_m_s"] for r in selected)/len(selected),
        leg_speed_peak_rad_s=max(r["leg_speed_peak_rad_s"] for r in selected)))
    raw[profile]=result;receipts[profile]=receipt
    evidence[profile]={str(p):sha(p) for p in [run/"receipt.json",run/"evaluation/result.json",REPORT/(name+"_review001.json")]}
requested=raw["requested"];paired=raw["without_alignment_paired"]
samples_a=requested["actual_domain_samples"];samples_b=paired["actual_domain_samples"]
matched={key:samples_a[key]==samples_b[key] for key in samples_a if key!="imu_fixed_rpy_rad"}
assert all(matched.values())
assert all(value==0. for row in samples_b["imu_fixed_rpy_rad"] for value in row)
assert len({r["checkpoint_sha256"] for r in receipts.values()})==1
assert len({r["seed"] for r in receipts.values()})==1
assert all(r["physics_runtime"]==receipts["requested"]["physics_runtime"] for r in receipts.values())
report=dict(schema="requested-domain-v2-readonly-comparison-v1", observed_utc=datetime.now(timezone.utc).isoformat(),
    checkpoint_sha256=receipts["requested"]["checkpoint_sha256"],seed=261201,
    paired_non_IMU_draws_equal=matched, physics_runtime_equal=True, new_PPO=0, optimizer_constructed=False,
    rows=rows,evidence_sha256=evidence,verifier_sha256=sha(Path(__file__)),
    findings=["Nominal-only009 is sensitive to requested domains: nominal middle-height7/7, requested0/7.",
              "Paired removal of fixed IMU alignment gives1/7, still1.653m worst drift. IMU alignment does not explain all sensitivity.",
              "Earlier unpaired3/7 result changed later random draws and is excluded from causal attribution.",
              "Low/high heights were not trained in009. This experiment is not the score of a newly trained robust policy."],
    qualified=False,hardware_ready=False)
out=REPORT/"requested_domains_comparison001.json"
assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(dict(paired_checks=len(matched),native_profiles=len(rows),hardware_ready=False)))
