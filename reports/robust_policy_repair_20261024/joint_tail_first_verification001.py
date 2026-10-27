from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path("/home/noir/Documents/workspace/isaac_wheeled_rl_train")
WORK = Path("/home/noir/.codex/worktrees/wheel-leg-domain-v2/isaac_wheeled_rl_train")
RUN = ROOT / "runs/scratch_standing_20261026/pilot011_joint_tail_requested001"
OUT = ROOT / "reports/robust_policy_repair_20261024/joint_tail_first_verification001.json"

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def flat(v):
    for x in v:
        if isinstance(x, list): yield from flat(x)
        else: yield x

progress = read(RUN / "progress.json")
init = read(RUN / "initialization.json")
old = read(ROOT / "runs/scratch_standing_20261026/pilot010_requested_domains_retry001/initialization.json")
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
    same_random_actor_as010=init["actor_tensor_sha256"] == old["actor_tensor_sha256"],
    same_random_critic_as010=init["critic_tensor_sha256"] == old["critic_tensor_sha256"],
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
old_run=ROOT/'runs/scratch_standing_20261026/pilot010_requested_domains_retry001'
old_samples=read(old_run/'randomization_study.initial_samples.json')['samples']
old_args=read(old_run/'progress.json')['args']
new_args=progress['args'].copy()
new_args.pop('standing_reward_shape')
for values in (new_args,old_args):
    values.pop('output')
    values.pop('updates')
old_task=read(old_run/'task_contract.json')['settings']
new_task=read(RUN/'task_contract.json')['settings']
changes={k for k in set(old_task)|set(new_task) if old_task.get(k)!=new_task.get(k)}
checks.update(all19_actual_initial_domain_draws_identical=s==old_samples,
    only_reward_flag_and_bounded_budget_changed=new_args==old_args,
    only_task_version_and_reward_shape_changed=changes=={'version','standing_reward_shape'},
    joint_shape_actually_installed=progress['precise_reward']['shape']=='joint-long-tail',
    native_recovery_reward_preserved=progress['precise_reward']['native_recovery_potential_retained_during_recovery'])
import torch
sys.path.insert(0,str(WORK/'src'))
from wheeled_algo.scratch_standing import tensor_hash
initial=torch.load(RUN/'model_initial.pt',map_location='cpu',weights_only=False)
checks['actual_initial_Adam_empty']=not initial['optimizer_state_dict']['state']
checks['actual_initial_actor_and_critic_hashes_match']=all(
    tensor_hash(initial[role+'_state_dict'])==init[role+'_tensor_sha256'] for role in ('actor','critic'))
assert all(checks.values()), checks
sys.path.insert(0,str(WORK/"src"))
from wheeled_tasks.chassis.randomization_study import scalar_metrics
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
events=EventAccumulator(str(RUN),size_guidance={"scalars":0}).Reload()
expected=scalar_metrics(resolved)
for key,value in expected.items():
    rows=events.Scalars("DomainStudyConfig/"+key)
    assert any(r.step==0 and math.isclose(r.value,value,rel_tol=2e-6,abs_tol=2e-8) for r in rows), key
shape_points=events.Scalars('PrecisionStandingConfig/joint_long_tail_enabled')
assert any(point.step==0 and point.value==1. for point in shape_points)
report=dict(schema="joint-tail-requested-domain-first-native-PPO-verification-v1",
    observed_utc=datetime.now(timezone.utc).isoformat(), run=str(RUN), source_checkout=str(WORK),
    pid=pid,start_ticks=start,completed_updates=progress["completed_updates"],
    commit=progress["committed_source"]["commit"], source_count=len(hashes),
    archive_sha256=progress["committed_source"]["archive_sha256"], checks=checks,
    all_checks_pass=True, RSL_config_scalars_verified=len(expected)+1,
    configured_randomized_probability=.75,actual_randomized_lanes=sum(s["enabled"]),num_envs=len(s["enabled"]),
    mass_summary=samples["rigid_summary"]["metrics"],
    native_behavior_qualified=False,hardware_ready=False,
    scope="Fresh initialization, actual sampled bounds and first committed native PPO update only; no robustness score",
    evidence_sha256={n:sha(RUN/n) for n in ["initialization.json","first_update_audit.json","source_revision.json","randomization_study.initial_samples.json","randomization_study.resolved.json"]},
    verifier_sha256=sha(Path(__file__)))
assert not OUT.exists()
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(dict(checks=len(checks),all_pass=True,completed_updates=report["completed_updates"],verified_RSL_config_scalars=len(expected),actual_randomized_lanes=sum(s["enabled"]))))

identity=dict(schema='scratch-from-zero-launch-identity-v2',observed_utc=report['observed_utc'],
    pid=pid,start_ticks=start,command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0'),
    source_checkout=str(WORK),commit=progress['committed_source']['commit'],
    source_archive_sha256=progress['committed_source']['archive_sha256'],source_count=len(hashes),
    run=str(RUN),exec_session_id=64290,requested_updates=128,prior_closed_fresh_updates=3457,
    observed_updates=progress['completed_updates'],initialization=init,first_verification=str(OUT),
    scope='Reward-only randomized middle-height foundation experiment, not the final unified policy',hardware_ready=False)
identity_path=ROOT/'reports/robust_policy_repair_20261024/scratch_pilot011_launch_identity001.json'
assert not identity_path.exists()
identity_path.write_text(json.dumps(identity,ensure_ascii=False,indent=2)+'\n')
ledger_path=ROOT/'reports/robust_policy_repair_20261024/scratch_campaign001.json'
raw=ledger_path.read_bytes()
ledger=json.loads(raw)
assert ledger['active_pid'] is None and ledger['prior_closed_fresh_updates']==3457
ledger.update(status='training_unqualified',active_run=str(RUN),active_pid=pid,active_start_ticks=start,
    requested_updates=128,completed_updates_observed=progress['completed_updates'],
    active_completed_updates_observed=progress['completed_updates'],
    total_from_zero_completed_observed=3457+progress['completed_updates'],
    active_launch_identity=str(identity_path),latest_first_verification=str(OUT),
    do_not_modify_live_src=True,do_not_modify_training_entrypoint=True,
    latest_observation='Fresh joint long-tail reward-only experiment passed initialization, identical domain draws and first actual nativePPO audit; behavior remains unqualified.',
    next_action='Finish the bounded128 updates with frozen sources; compare first100 RSL metrics and the same checkpoint under nominal/requested native fixed evaluation.',
    observed_utc=report['observed_utc'],last_observed_utc=report['observed_utc'])
assert ledger_path.read_bytes()==raw
ledger_path.with_suffix('.jointtail.tmp').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
ledger_path.with_suffix('.jointtail.tmp').replace(ledger_path)
