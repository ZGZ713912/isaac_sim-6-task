from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from wheeled_algo.scratch_standing import tensor_hash

torch.set_num_threads(2)
identity = json.loads((BASE/'scratch_pilot008_launch_identity001.json').read_text())
run = Path(identity['run'])
progress = json.loads((run/'progress.json').read_text())
manifest = json.loads((run/'source_hashes.json').read_text())
task = json.loads((run/'task_contract.json').read_text())
init = json.loads((run/'initialization.json').read_text())
audit = json.loads((run/'first_update_audit.json').read_text())
initial = torch.load(run/'model_initial.pt', map_location='cpu', weights_only=False)
first = torch.load(run/'model_0.pt', map_location='cpu', weights_only=False)
reference = torch.load(ROOT/'runs/scratch_standing_20261026/pilot007_contact_cost_zero/model_initial.pt',
                       map_location='cpu', weights_only=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


stat = Path(f"/proc/{identity['pid']}/stat").read_text()
gpu = [int(value) for value in subprocess.check_output(
    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
    if value.strip().isdigit()]
event, = run.glob('events.out.tfevents.*')
raw = event.read_bytes()
tags = ['LegTargetRate/'+name for name in ('target_delta_rms_rad', 'target_delta_peak_rad',
    'target_rate_rms_rad_s', 'cost_density_per_second', 'paid_cost_per_policy_step', 'weight_density_per_second')]
with tempfile.TemporaryDirectory() as temporary:
    copy = Path(temporary)/event.name; copy.write_bytes(raw)
    reader = EventAccumulator(str(copy), size_guidance={'scalars':0}).Reload()
    observed = {tag:[dict(step=row.step, value=row.value) for row in reader.Scalars(tag)] for tag in tags}
assert all(observed.values()), 'Wait for actual new reward tags in the RSL event'
first_values = {tag: rows[0]['value'] for tag, rows in observed.items()}
checks = dict(original_process_identity=int(stat[stat.rindex(')')+2:].split()[19]) == identity['start_ticks'],
    original_process_on_GPU=identity['pid'] in gpu,
    initial_actual_Adam_empty=initial['optimizer_state_dict']['state'] == {},
    initial_actual_clock_zero=initial['iter'] == initial['infos']['completed_updates'] == 0,
    initialization_forbids_checkpoint_teacher=init['checkpoint_loaded'] is False and init['teacher_loaded'] is False,
    no_training_checkpoint_reads=progress['checkpoint_reads'] == [],
    initial_actor_exactly007_initial=all(torch.equal(value, reference['actor_state_dict'][name])
        for name, value in initial['actor_state_dict'].items()),
    initial_critic_exactly007_initial=all(torch.equal(value, reference['critic_state_dict'][name])
        for name, value in initial['critic_state_dict'].items()),
    initial_tensor_hashes_match=all(tensor_hash(initial[role+'_state_dict']) == init[role+'_tensor_sha256']
        for role in ('actor','critic')),
    all12_first_Adam_calls_finite=audit['committed_optimizer_calls'] == 12 and audit['all_recorded_values_finite'],
    actor_and_critic_parameters_changed=audit['actor_changed_parameter_tensors'] == 9
        and audit['critic_changed_parameter_tensors'] == 8,
    actual_first_Adam_state_covers17=len(first['optimizer_state_dict']['state']) == 17,
    first_checkpoint_Adam_steps12=all(int(value['step'].item()) == 12
        for value in first['optimizer_state_dict']['state'].values()),
    actual_source_snapshot263_matches=len(manifest) == 263 and all(sha(ROOT/name) == digest
        and sha(run/'source_snapshot'/name) == digest for name, digest in manifest.items()),
    actual_cost_contract_matches=progress['standing_leg_target_rate_contract'] == task['standing_leg_target_rate_contract']
        and progress['standing_leg_target_rate_contract']['weight_density_per_second'] == .2,
    all6_real_RSL_cost_tags_present=all(rows[0]['step'] == 0 for rows in observed.values()),
    actual_cost_paid_positive=first_values['LegTargetRate/paid_cost_per_policy_step'] > 0.,
    actual_density_integrates_policy_dt=abs(first_values['LegTargetRate/paid_cost_per_policy_step']
        - .005*first_values['LegTargetRate/cost_density_per_second']) < 1e-8,
    actual_logged_weight_correct=abs(first_values['LegTargetRate/weight_density_per_second']-.2) < 1e-7,
    actual_pure_RL_rollout=progress['valid_rollout']['valid_fraction'] == 1.
        and progress['valid_rollout']['script_or_blend_transitions'] == 0)
assert all(checks.values()), checks
report = dict(schema='scratch-first-native-update-verification-v1', observed_utc=datetime.now(timezone.utc).isoformat(),
    run=str(run), pid=identity['pid'], start_ticks=identity['start_ticks'],
    checks=checks, all_checks_pass=True, completed_updates_observed=progress['completed_updates'],
    prior_completed_fresh_updates=1921, total_completed_fresh_updates=1921+progress['completed_updates'],
    event_prefix=dict(path=str(event), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
    first_RSL_cost_values=first_values, reader_sha256=sha(__file__),
    initial_actor_tensor_sha256=init['actor_tensor_sha256'], initial_critic_tensor_sha256=init['critic_tensor_sha256'],
    new_PPO_by_reader=0, behavior_qualification=False, hardware_ready=False,
    scope='This separate CPU reader compares initial tensors only; no old weights are assigned into training. '
          'Fresh optimization and actual reward payment are verified; standing improvement remains unevaluated.')
output = BASE/'scratch_pilot008_first_verification001.json'
assert not output.exists()
output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(json.dumps(dict(checks=checks, completed_updates=progress['completed_updates'],
    first_RSL_cost_values=first_values)))
