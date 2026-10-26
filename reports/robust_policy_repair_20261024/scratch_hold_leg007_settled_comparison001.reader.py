from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
a = ROOT / 'runs/scratch_standing_20261026/eval007_final512_contact_state001'
b = ROOT / 'runs/scratch_standing_20261026/eval007_hold_settled_legs001'
ra, rb = [json.loads((p / 'receipt.json').read_text()) for p in (a, b)]
gpu = [int(value) for value in subprocess.check_output(
    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
    if value.strip().isdigit()]
checks = dict(receipt_complete=rb['status'] == 'completed_unqualified',
    native_process_exited=not Path('/proc/704672').exists(), native_GPU_absent=704672 not in gpu,
    new_PPO_zero=rb['new_PPO'] == 0, optimizer_not_constructed=not rb['optimizer_constructed'],
    checkpoint_exact=ra['checkpoint_sha256'] == rb['checkpoint_sha256'],
    physics_runtime_exact=ra['physics_runtime'] == rb['physics_runtime'],
    all1600_actor_calls=rb['policy_counterfactual']['actor_calls'] == 1600,
    declared_boundary_exact=rb['policy_counterfactual']['held_after_s'] == 3.
        and rb['policy_counterfactual']['native_prefix_policy_frames'] == 601)
other = list(range(7)) + list(range(14, 21))
with np.load(a / 'evaluation/feedback_trace.npz') as x, np.load(b / 'evaluation/feedback_trace.npz') as y, \
        np.load(b / 'evaluation/unmodified_actor_actions.npz') as native:
    checks['initial_native_state_exact'] = all(np.array_equal(x[key], y[key]) for key in x.files
        if key.startswith('initial_'))
    checks['native_inputs_and_actions_601_frame_prefix_exact'] = all(np.array_equal(x[key][:601], y[key][:601])
        for key in ('observation_policy', 'action_raw_policy'))
    checks['native_wheel_outputs_unchanged'] = np.array_equal(
        y['action_raw_policy'][:, :, 4:], native['action_policy'][:, :, 4:])
    checks['post_boundary_selected_legs_held'] = np.array_equal(y['action_raw_policy'][600:, 7:14, :4],
        np.broadcast_to(native['held_leg_targets_policy'][None, 7:14, :4], (1000, 7, 4)))
    checks['targets_captured_from_original_3s_action'] = np.array_equal(
        native['held_leg_targets_policy'][7:14], x['action_raw_policy'][600, 7:14, :4])
with np.load(a / 'evaluation/native_trace.npz') as x, np.load(b / 'evaluation/native_trace.npz') as y:
    checks['entire_native_601_frame_prefix_exact'] = all(np.array_equal(x[key][:601], y[key][:601]) for key in x.files)
    checks['all_nonselected_native_trajectories_exact'] = all(np.array_equal(
        np.take(x[key], other, axis=2 if key in ('force', 'clearance', 'dq', 'torque') else 1),
        np.take(y[key], other, axis=2 if key in ('force', 'clearance', 'dq', 'torque') else 1)) for key in x.files)
    post = []
    for lane in range(7, 14):
        values = []
        for trace in (x, y):
            present = (np.linalg.norm(trace['force'][:, :, lane], axis=-1) > 2.).all(-1)
            values.append(int((~present[601:]).sum()))
        post.append(dict(pitch_deg=ra['evaluation']['cases'][lane]['pitch_deg'],
            baseline_missing_ms=values[0], intervention_missing_ms=values[1]))


def group(receipt):
    rows = receipt['evaluation']['cases'][7:14]
    return dict(missing_ms=sum(row['missing_physical_ms'] for row in rows),
        standing_pass=sum(row['standing_pass'] for row in rows),
        mean_vibration_mrad=sum(row['leg_vibration_3_95hz_rms_mrad'] for row in rows)/7,
        mean_forward_speed_rms_m_s=sum(row['forward_speed_rms_m_s'] for row in rows)/7)


report = dict(schema='scratch-settled-leg-hold-native-comparison-v1',
    observed_utc=datetime.now(timezone.utc).isoformat(), checks=checks, all_checks_pass=all(checks.values()),
    reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), baseline=str(a), intervention=str(b),
    baseline_mid=group(ra), intervention_mid=group(rb), post_intervention_cases=post,
    post_intervention_interval_s=[3.005, 8.], original_strict8s_acceptance_retained=True,
    new_PPO=0, adoption_by_training=False, behavior_qualification=False, hardware_ready=False,
    decision='Reject settled-target hold as a standing repair. Removing all subsequent leg correction '
             'still worsens support overall. This does not establish a unique excitation source; '
             'keep direct six-motor feedback and investigate learning smoother target changes.')
assert all(checks.values()), checks
output = BASE / 'scratch_hold_leg007_settled_comparison001.json'
assert not output.exists()
output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(report))
