"""Verify a native population-only 4096-env v16 resume; no optimizer or env steps."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT/'runs/flat_base_20261019'
sys.path.insert(0, str(ROOT/'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.separated_feedback import KIND

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--first-only',action='store_true')
args = parser.parse_args()
torch.set_num_threads(1)
load = lambda p: torch.load(p,map_location='cpu',weights_only=True)
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
trial = RUN/'pilot024'
required = (1249,) if args.first_only else (1249,1264)
output = HERE/('feedback1264_native1249.json' if args.first_only else 'feedback1264_native_milestones.json')
if output.exists():
    raise SystemExit('Preserve completed native entry-change proof')
paths = [trial/'initial_state.pt',*(trial/f'model_{n:06d}.pt' for n in required)]
if not all(p.is_file() for p in paths):
    raise SystemExit('Wait for all requested actual checkpoints; no partial receipt written')
parent_path = RUN/'pilot023/model_001248.pt'
parent = load(parent_path)
initial = load(trial/'initial_state.pt')
info, before = initial['infos'], parent['infos']
entry_path = HERE/'population_capacity_plan_revision019.json'
revision = read(entry_path)


def exact(a,b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict): return a.keys() == b.keys() and all(exact(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)): return len(a) == len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a == b


changed = sorted(n for n in set(info['source_sha256']) | set(before['source_sha256'])
    if info['source_sha256'].get(n) != before['source_sha256'].get(n))
entry = 'scripts/train_flat_base.py'
height = info.get('height_command_metrics_contract',{})
checks = {f'initial_{key}_exact_parent':exact(initial[key],parent[key]) for key in
    ('actor_state_dict','critic_state_dict','optimizer_state_dict','torch_rng_state','cuda_rng_state')}
checks.update(native_all_six_restore_checks=len(info['full_restore_verification']) == 6
        and all(info['full_restore_verification'].values()),
    exact1248_parent=sha(parent_path) == info['parent_checkpoint_sha256'] == revision['parent_checkpoint_sha256']
        == 'da285c8ddc1958af6c5e3b232c97bb3729370459235e2d90268738c2b0067d7d',
    isolated_reviewed_population_entry_change=changed == [entry]
        and before['source_sha256'] == revision['old_source_sha256']
        and info['source_sha256'] == revision['new_source_sha256']
        and info.get('plan_revision_sha256') == sha(entry_path),
    exact_population_only_plan_change=info['plan_file'] == 'reports/flat_base_continuation_20261019/population_coverage_candidate001/plan_4096.json'
        and info['plan_sha256'] == revision['new_plan_sha256']
        and before['plan_sha256'] == revision['old_plan_sha256']
        and info['changed_plan_fields'] == ['allowed_num_envs','default_num_envs']
        and info['population_transition'] == revision['population_transition']
        and info['num_envs'] == 4096 and before['num_envs'] == 512,
    inherited_rollout_frames_retained=info['training_transitions_total'] == 61341696,
    no_rate_or_teacher_change=info['optimizer_after_declared_rate_exact']
        and not info['optimizer_rate_contract']['declared_actor_rate_change']
        and info['teacher_anchor_runtime'] == before['teacher_anchor_runtime'],
    no_identity_fork_or_extra_warmup=info['actor_adaptation'] == KIND
        and info['restore_verification_reference'] == 'parent'
        and not info.get('separated_feedback_identity_extension') and info['critic_warmup_remaining'] == 0,
    parent_counters=(info['successful_updates_total'],info['actor_updates_total'],info['active_branch_actor_updates'],
        info['feedback_branch_actor_updates'],info['post_split_actor_updates'],info['inherited_shared_feedback_updates'],
        info['consumed_optimizer_updates_total']) == (1248,1068,44,690,386,304,1249),
    same_control_and_command_reference=info['control_math_sha256'] == before['control_math_sha256']
        == 'cc30b2b0572999224c8feb79ffa7afb3df1b8a28800bca4a153663b9ea8d4bf8'
        and {k:v for k,v in info['training_command_reference'].items() if k not in ('population','direct_lanes','ramped_lanes')}
            == {k:v for k,v in before['training_command_reference'].items() if k not in ('population','direct_lanes','ramped_lanes')}
        and info['training_command_reference']['population'] == read(HERE/'population_coverage_candidate001/receipt.json')['populations']['4096']['population'],
    height_metadata_present=height.get('schema') == 'flat-height-command-metrics-v1'
        and height.get('bands') == ['height_low','height_nominal','height_high']
        and height.get('height_bin_edges_m') == [(.23+.305)/2,(.305+.42)/2]
        and height.get('step_unit') == 'completed optimizer updates')
timing = info['startup']['runtime_preflight']['observed']
checks['actual50Hz_policy1000Hz_physics_feedback_PD'] = all(timing[n] == value for n,value in
    dict(physics_dt=.001,policy_dt=.02,pc_control_dt=.001,decimation=20).items())
mean = mean_from_checkpoint(initial,ROOT/'models/wheel_leg_flat_base_20261018/source.onnx')
names = ['distribution.std_param']+['mlp.'+name for name,_ in mean.named_parameters()]
ids = dict(zip(names,initial['optimizer_state_dict']['param_groups'][0]['params'],strict=True))
prefixes = ('mlp.feedback.','mlp.fast_feedback.')
learned = [name for name in names if name.startswith(prefixes)]
frozen = [name for name in names if not name.startswith(prefixes)]
slots0 = initial['optimizer_state_dict']['state']
milestones = {}
for count in required:
    path = trial/f'model_{count:06d}.pt'
    state = load(path); row = state['infos']; slots = state['optimizer_state_dict']['state']; delta = count-1248
    proof = dict(frozen_actor_std_buffers_exact=all(torch.equal(value,state['actor_state_dict'][name])
            for name,value in initial['actor_state_dict'].items() if not name.startswith(prefixes)),
        frozen_Adam_slots_exact=all(exact(slots0.get(ids[name]),slots.get(ids[name])) for name in frozen),
        eight_feedback_tensors_learned=len(learned) == 8 and all(not torch.equal(initial['actor_state_dict'][name],state['actor_state_dict'][name]) for name in learned),
        only2252_feedback_parameters_trainable=row['actor_trainability']['trainable_parameters'] == 2252
            and set(row['actor_trainability']['trainable_names']) == set(learned),
        actual12_Adam_steps_each_update=all(float(slots[i]['step']) == float(slots0[i]['step'])+12*delta
            for i in [ids[name] for name in learned]+initial['optimizer_state_dict']['param_groups'][1]['params']),
        critic_learned=any(not torch.equal(value,state['critic_state_dict'][name]) for name,value in initial['critic_state_dict'].items()),
        all_history_and_consumption_preserved=(row['successful_updates_total'],row['actor_updates_total'],
            row['active_branch_actor_updates'],row['feedback_branch_actor_updates'],row['post_split_actor_updates'],
            row['inherited_shared_feedback_updates'],row['consumed_optimizer_updates_total'],row['critic_warmup_remaining'])
            == (count,1068+delta,44,690+delta,386+delta,304,count+1,0),
        exact_group_rates=[group['lr'] for group in state['optimizer_state_dict']['param_groups']] == [3e-4,5e-4]
            and math.isclose(row['last_losses']['actor_learning_rate'],3e-4,rel_tol=1e-12)
            and math.isclose(row['last_losses']['critic_learning_rate'],5e-4,rel_tol=1e-12),
        all_losses_finite=all(math.isfinite(value) for value in row['last_losses'].values()),
        height_contract_unchanged=row.get('height_command_metrics_contract') == height,
        exact_cumulative_rollout_frames=row['training_transitions_total'] == 61341696+delta*4096*96)
    checks.update({str(count)+'_'+key:value for key,value in proof.items()})
    milestones[str(count)] = dict(checkpoint_sha256=sha(path),checks=proof)
height_rows = 0
if not args.first_only:
    events = EventAccumulator(str(trial),size_guidance={'scalars':0});events.Reload()
    tags = set(events.Tags()['scalars'])
    cache = {}
    def scalar(tag):
        if tag not in cache:
            cache[tag] = {point.step:point.value for point in events.Scalars(tag)}
        return cache[tag]
    expected = list(range(1249,1265))
    height_checks = []
    for namespace in ('FlatBehavior','FlatBirth'):
        for group,population in info['training_command_reference']['population'].items():
            views = [group]+([group+'/rmcs_direct'] if population['direct'] else [])+([group+'/legacy_ramp'] if population['ramped'] else [])
            for view in views:
                prefix = namespace+'/'+view
                whole = scalar(prefix+'/sample_frames')
                bins = [scalar(prefix+'/'+band+'/sample_frames') for band in height['bands']]
                height_checks.append(sorted(whole) == expected and all(sorted(values) == expected for values in bins))
                for count in expected:
                    total = whole[count]
                    height_checks.append(sum(values[count] for values in bins) == total)
                    for metric in ('vx_mae_m_s','yaw_mae_rad_s','height_mae_m','planar_speed_mean_m_s','terminated_frame_fraction'):
                        aggregate_tag = prefix+'/'+metric
                        all_values = scalar(aggregate_tag) if aggregate_tag in tags else {}
                        weighted = 0.
                        for band,frames in zip(height['bands'],bins,strict=True):
                            tag = prefix+'/'+band+'/'+metric
                            values = scalar(tag) if tag in tags else {}
                            if frames[count]:
                                height_checks.append(count in values and math.isfinite(values.get(count,float('nan'))))
                                weighted += frames[count]*values.get(count,float('nan'))
                            else:
                                height_checks.append(count not in values)
                        height_checks.append(math.isclose(weighted/total,all_values.get(count,float('nan')),rel_tol=3e-7,abs_tol=3e-8)
                            if total else count not in all_values)
                        height_rows += 1
    checks['all16_rollout_frame_clock_TensorBoard_points'] = all(scalar('FlatTransfer/training_transitions_total')[n] == 61341696+(n-1248)*4096*96 for n in expected)
    checks['all_height_counts_and_weighted_errors_match_existing_logger'] = bool(height_checks) and all(height_checks)
receipt = dict(schema='flat-feedback1264-population-native-verification-v1',passed=all(checks.values()),
    checks=checks,milestones=milestones,all_required_milestones_present=True,first_only=args.first_only,
    height_metric_rows_checked=height_rows,optimizer_updates_performed_by_verifier=0,behavior_qualified=False)
if not receipt['passed']:
    print(json.dumps(checks,indent=2));raise SystemExit('Actual native population-only continuation checks failed')
output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),milestones=list(milestones),height_metric_rows_checked=height_rows)))
