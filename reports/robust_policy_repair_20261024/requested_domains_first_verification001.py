from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path("/home/noir/Documents/workspace/isaac_wheeled_rl_train")
WORK = Path("/home/noir/.codex/worktrees/wheel-leg-domain-v2/isaac_wheeled_rl_train")
RUN = ROOT / "runs/scratch_standing_20261026/pilot010_requested_domains_retry001"
OUT = ROOT / "reports/robust_policy_repair_20261024/requested_domains_first_verification001.json"

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def flat(v):
    for x in v:
        if isinstance(x, list): yield from flat(x)
        else: yield x

progress = read(RUN / "progress.json")
init = read(RUN / "initialization.json")
old = read(ROOT / "runs/scratch_standing_20261026/pilot009_leg_velocity_separated/initialization.json")
audit = read(RUN / "first_update_audit.json")
samples = read(RUN / "randomization_study.initial_samples.json")
s = samples["samples"]
resolved = read(RUN / "randomization_study.resolved.json")
hashes = read(RUN / "source_hashes.json")
pid = progress["pid"]
start = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
checks = dict(
    exact_live_process=start == progress["start_ticks"],
    actual_PPO=progress["completed_updates"] >= 1,
    no_checkpoint_reads=progress["checkpoint_reads"] == [],
    fresh_actor_and_critic=init["checkpoint_loaded"] is False and init["teacher_loaded"] is False,
    empty_initial_Adam=init["optimizer_states"] == 0,
    same_random_actor_as009=init["actor_tensor_sha256"] == old["actor_tensor_sha256"],
    same_random_critic_as009=init["critic_tensor_sha256"] == old["critic_tensor_sha256"],
    native_optimizer_calls=audit["committed_optimizer_calls"] == 12,
    finite_first_update=audit["all_recorded_values_finite"],
    actor_and_critic_changed=audit["actor_changed_parameter_tensors"] == 9 and audit["critic_changed_parameter_tensors"] == 8,
    all_frozen_sources_exact=all(sha(RUN/"source_snapshot"/k) == v for k,v in hashes.items()),
    all_training_checkout_sources_exact=all(sha(WORK/k) == v for k,v in hashes.items()),
    committed_sources=progress["committed_source"]["all_source_bytes_match_commit"],
    shared_sensor_rigid_mask=s["enabled"] == s["rigid_enabled"],
    fixed_P=all(x == 1. for x in flat(s["leg_kp_scale"])),
    fixed_D=all(x == 1. for x in flat(s["leg_kd_scale"])),
    fixed_spring=all(x == 1. for x in flat(s["spring_strength"])),
    zero_IMU_wobble=all(x == 0. for x in flat(s["imu_wobble_amplitude_rad"])),
    inertia_requested=resolved["dynamics_randomization"]["inertia_scale"] == [.94,1.06],
    inertia_samples=all(.94 <= x <= 1.06 for x in flat(s["independent_inertia_scale"])),
    mass_samples=all(.92 <= x <= 1.08 for x in flat(s["mass_scale"])),
    motor_samples=all(.84 <= x <= 1.16 for x in flat(s["motor_strength"])),
    IMU_fixed_samples=all(abs(x) <= math.radians(b)+1e-8 for row in s["imu_fixed_rpy_rad"] for x,b in zip(row,[.8,.8,.8/3.])),
    explicit_policy_delay=all(x in [0,1] for x in s["policy_sensor_delay_ticks"]),
    no_legacy_overlay="gap_robustness_version" not in resolved["task"] and "domain_scale" not in resolved["task"])
assert all(checks.values()), checks
sys.path.insert(0,str(WORK/"src"))
from wheeled_tasks.chassis.randomization_study import scalar_metrics
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
events=EventAccumulator(str(RUN),size_guidance={"scalars":0}).Reload()
expected=scalar_metrics(resolved)
for key,value in expected.items():
    rows=events.Scalars("DomainStudyConfig/"+key)
    assert any(r.step==0 and math.isclose(r.value,value,rel_tol=2e-6,abs_tol=2e-8) for r in rows), key
report=dict(schema="requested-domain-v2-first-native-PPO-verification-v1",
    observed_utc=datetime.now(timezone.utc).isoformat(), run=str(RUN), source_checkout=str(WORK),
    pid=pid,start_ticks=start,completed_updates=progress["completed_updates"],
    commit=progress["committed_source"]["commit"], source_count=len(hashes),
    archive_sha256=progress["committed_source"]["archive_sha256"], checks=checks,
    all_checks_pass=True, RSL_config_scalars_verified=len(expected),
    configured_randomized_probability=.75,actual_randomized_lanes=sum(s["enabled"]),num_envs=len(s["enabled"]),
    mass_summary=samples["rigid_summary"]["metrics"],
    native_behavior_qualified=False,hardware_ready=False,
    scope="Fresh initialization, actual sampled bounds and first committed native PPO update only; no robustness score",
    evidence_sha256={n:sha(RUN/n) for n in ["initialization.json","first_update_audit.json","source_revision.json","randomization_study.initial_samples.json","randomization_study.resolved.json"]},
    verifier_sha256=sha(Path(__file__)))
assert not OUT.exists()
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(dict(checks=len(checks),all_pass=True,completed_updates=report["completed_updates"],verified_RSL_config_scalars=len(expected),actual_randomized_lanes=sum(s["enabled"]))))
