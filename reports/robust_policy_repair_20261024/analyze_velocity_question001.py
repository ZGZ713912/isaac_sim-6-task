"""Read authenticated native traces; explain speed and standing failure scope."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import torch

sys.path[:0] = ['src', 'scripts']
from analyze_robust_getup import analyze, hf_rms
from wheeled_algo.robust_getup import quiet_gate, tilt_gate
from wheeled_algo.upright_target_limit import rebase_final_leg_pairs
from wheeled_tasks.chassis.paired_feedback import paired_position_error
from wheeled_tasks.chassis.paired_leg_targets import paired_leg_targets

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
LEGS = [0, 1, 3, 4]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, data):
    (R / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def trace(directory):
    report = json.loads((directory / 'report.json').read_text())
    assert sha(directory / 'trajectory.npz') == report['trajectory_sha256']
    return report, dict(np.load(directory / 'trajectory.npz'))


plan = json.loads((R / 'diagnosis_plan056.json').read_text())
assert json.loads((R / 'diagnosis_queue056.json').read_text())['status'] == 'completed'
assert all(sha(p) == h for p, h in plan['source_sha256'].items())
old = B / 'diagnosis055_historical_full_mean_moving'
new = B / 'diagnosis056_current9a_matched_moving'
initial_a = dict(np.load(old / 'all_precontrol_state.npz'))
initial_b = dict(np.load(new / 'all_precontrol_state.npz'))
initial_equal = {k: bool(np.array_equal(initial_a[k], initial_b[k], equal_nan=True)) for k in initial_a}
assert all(initial_equal.values())
matched = {}
for variant in ('robust', 'early_roll20_to_40', 'early_roll20_to_50'):
    a, da = trace(old / variant)
    b, db = trace(new / variant)
    context = dict(cases=[x['name'] for x in a['lanes']] == [x['name'] for x in b['lanes']],
                   material=a['material'] == b['material'], domains=a['robust_domains'] == b['robust_domains'])
    assert all(context.values()), context
    results = {}
    for label, report, data, directory in (('historical', a, da, old), ('current9a', b, db, new)):
        analysis = analyze(directory / variant)
        sides = {}
        for side in ('left', 'right'):
            rows = [x for x in report['lanes'] if x['name'] == 'moving_push_' + side]
            ids = [x['lane'] for x in rows]
            sides[side] = dict(episodes=len(rows), survived=sum(x['survived'] for x in rows),
                recovered=sum(x['recovered'] for x in rows), strict_followup=sum(x['followup_success'] for x in rows),
                physical_leg_speed_recorded_200hz_peak_rad_s=float(np.abs(data['motor_velocity'][:, ids][:, :, LEGS]).max()),
                impacts=sum(any('impact' in c for c in x['termination_causes']) for x in rows))
            if 'leg_speed_peak_rad_s' in data:
                sides[side]['physical_leg_speed_native_1ms_peak_rad_s'] = float(data['leg_speed_peak_rad_s'][:, ids].max())
        results[label] = dict(sides=sides, actual_moving_fall_followup=analysis['practical_moving_fall_recoveries'],
                              max_gap_m=report['max_gap_m'], passive_motor_peak_nm=report['passive_motor_peak_nm'])
    matched[variant] = dict(context_exact=context, results=results)
save('current9a_matched_moving_result001.json', dict(schema='matched-current-historical-moving-v1',
     initial_exact=initial_equal, variants=matched, new_PPO=0,
     scope='Whole policy functions differ. This is not a one-variable limiter or learned-weight ablation.',
     qualification=None, hardware_started=False, created_unix=time.time()))
save('diagnosis056_closed_source_seal001.json', dict(schema='native-diagnosis-source-closure-v1',
     status='completed', all_frozen_sources_verified=True, frozen_inputs=len(plan['source_sha256']),
     plan_sha256=sha(R / 'diagnosis_plan056.json'), review_sha256=sha(__file__), new_PPO=0,
     actual_experiment_consumed=2662, repair_optimizer_updates=282, qualification=None, created_unix=time.time()))

report, data = trace(old / 'robust')
runtime = json.loads((old / 'runtime_function_readback.json').read_text())
settings = runtime['historical_settings']
manifest_path = Path('model/纯底盘_v7_flat_base107/urdf/manifest.json')
manifest = json.loads(manifest_path.read_text())
nominal = torch.tensor([manifest['nominal_joint_pos'][n] for n in manifest['control_joint_names']])[LEGS]
lane = next(x['lane'] for x in report['lanes'] if x['name'] == 'moving_push_left')
speeds = np.abs(data['motor_velocity'][:, lane, LEGS])
frame, axis = np.unravel_index(speeds.argmax(), speeds.shape)
window = []
for f in range(frame - 9, frame + 2):
    current = torch.from_numpy(data['motor_position'][f - 1, lane, LEGS]).unsqueeze(0)
    raw = torch.from_numpy(data['actions'][f, lane, LEGS]).unsqueeze(0)
    target = paired_leg_targets(nominal + .25 * raw, current)
    obs = torch.from_numpy(data['observations'][f, lane]).unsqueeze(0)
    gate = tilt_gate(obs, settings['stance_aware_recovery'], settings['recovery_shape_aware'])
    window.append(dict(time_s=float(data['time'][f, lane]), decoded_target_rad=target[0].tolist(),
        pre_step_error_rad=(target - current)[0].tolist(), motor_velocity_rad_s=data['motor_velocity'][f, lane, LEGS].tolist(),
        motor_effort_nm=data['motor_effort'][f, lane, LEGS].tolist(), recovery_gate=float(gate[0, 0]),
        quiet_gate=float(quiet_gate(obs)[0, 0]), wheel_contact=data['wheel_contact'][f, lane].tolist(),
        impact_force_n=float(data['impact'][f, lane]),
        target_pair_straddles_pi=bool(((raw * .25).unflatten(-1, (2, 2)).diff(dim=-1).abs() > torch.pi).any())))
standing = {}
for name, path in (('source_c025', B / 'diagnosis036_quiet_stance050_implementation/evaluation'),
                   ('quiet017', B / 'pilot017_quiet_output_fixed/evaluation')):
    rr, dd = trace(path)
    row = next(x for x in rr['lanes'] if x['name'] == 'stand_low__roll_minus2')
    l = row['lane']
    active = dd['alive'][:, l] & ~dd['passive'][:, l]
    tail = active & (dd['time'][:, l] >= rr['seconds'] - 4.)
    last = active & (dd['time'][:, l] >= rr['seconds'] - 1.)
    masks = dict(tilt=dd['gravity'][:, l, 2] < -np.cos(np.deg2rad(15.)),
        gyro=np.linalg.norm(dd['omega'][:, l, :2], axis=-1) < .7,
        height=dd['height'][:, l] > .19,
        leg_speed=np.abs(dd['motor_velocity'][:, l, LEGS]).max(-1) < 3.,
        both_wheel_contact=dd['wheel_contact'][:, l].all(-1))
    # Preserve the existing GapStanding event metric's FFT/window definition.
    q = np.unwrap(dd['motor_position'][last, l][:, LEGS], axis=0)
    spectrum = np.fft.rfft(q - q.mean(0), axis=0)
    frequency = np.fft.rfftfreq(len(q), .005)
    spectrum[(frequency < 3.) | (frequency > 95.)] = 0.
    physical_rms = float(np.sqrt(np.mean(np.fft.irfft(spectrum, n=len(q), axis=0) ** 2)))
    standing[name] = dict(run=str(path), trajectory_sha256=rr['trajectory_sha256'],
        tail4s_stable_fraction=float(dd['stable'][tail, l].mean()),
        tail4s_native_gate_pass_fraction={k: float(v[tail].mean()) for k, v in masks.items()},
        tail1s_physical_leg_motion_3_95hz_rms_rad=physical_rms,
        tail1s_leg_speed_rms_rad_s=float(np.sqrt(np.mean(dd['motor_velocity'][last, l][:, LEGS] ** 2))))

stopped_path = B / 'stopped1501_eval001/evaluation'
sr, sd = trace(stopped_path)
stopped_speed = np.abs(sd['motor_velocity'][:, :, LEGS])
sf, sl, sj = np.unravel_index(stopped_speed.argmax(), stopped_speed.shape)
stopped_window = []
for f in range(sf - 3, sf + 2):
    current = torch.from_numpy(sd['motor_position'][f - 1, sl, LEGS]).unsqueeze(0)
    raw = torch.from_numpy(sd['actions'][f, sl, LEGS]).unsqueeze(0)
    original_target = paired_leg_targets(nominal + .25 * raw, current)
    rebased = rebase_final_leg_pairs(raw)
    corrected_target = paired_leg_targets(nominal + .25 * rebased, current)
    # The inputs are identical physical per-axis angles modulo2pi. Their old
    # arithmetic paired projection nevertheless chooses different mechanics.
    equivalence = torch.atan2(torch.sin(.25 * (rebased - raw)), torch.cos(.25 * (rebased - raw)))
    assert float(equivalence.abs().max()) < 1e-6
    stopped_window.append(dict(time_s=float(sd['time'][f, sl]),
        relative_target_pair_difference_rad=((raw * .25).unflatten(-1, (2, 2)).diff(dim=-1)).flatten().tolist(),
        original_pd_position_error_rad=paired_position_error(original_target, current)[0].tolist(),
        circular_rebased_pd_position_error_rad=paired_position_error(corrected_target, current)[0].tolist(),
        physical_input_angle_equivalence_error_rad=float(equivalence.abs().max()),
        motor_velocity_rad_s=sd['motor_velocity'][f, sl, LEGS].tolist(),
        motor_effort_nm=sd['motor_effort'][f, sl, LEGS].tolist(),
        wheel_contact=sd['wheel_contact'][f, sl].tolist()))
stopped = dict(run=str(stopped_path), checkpoint_sha256=sr['checkpoint_sha256'],
    trajectory_sha256=sr['trajectory_sha256'], lane=int(sl), case=sr['lanes'][sl]['name'],
    control_axis=LEGS[sj], joint_name=manifest['control_joint_names'][LEGS[sj]],
    recorded_peak_rad_s=float(stopped_speed[sf, sl, sj]), time_s=float(sd['time'][sf, sl]),
    phase_window=stopped_window,
    counterfactual_scope='Same recorded state/actions, CPU decoder only. Corrected target errors are not a new native episode or proof that recovery succeeds.')
out = dict(schema='native-velocity-and-standing-explanation-v1', historical_peak=dict(
    run=str(old / 'robust'), trajectory_sha256=report['trajectory_sha256'], lane=lane,
    physical_control_axis=LEGS[axis], joint_name=manifest['control_joint_names'][LEGS[axis]],
    speed_rad_s=float(speeds[frame, axis]), frame=int(frame), time_s=float(data['time'][frame, lane]),
    phase_window=window,
    reconstruction='Targets decoded from exact saved applied actions and preceding native state; fixed nominal manifest and paired_branch107_v1. Motor effort is the last 1ms torque, velocity is post-step; they are not exactly simultaneous power samples.'),
    stopped_long_training_peak=stopped, standing=standing, matched_control=matched['robust'],
    confirmed=['Leg simulation velocity_limit_sim is1e9; external position PD160/2.5 clips torque40Nm, not speed.',
        'The high-speed window has large decoded target error, torque saturation and both wheels airborne; no paired target pi-straddle in that window.',
        'The stopped1501 trace has a separate target pair winding error: physically equivalent per-axis target angles produce a wrong arithmetic paired projection and large positive PD error; common-winding rebasing changes it to a small braking error.',
        'The quiet017 lost standing case loses the native contact gate while all four other measured stable gates pass.'],
    not_identified=['No unique causal share for motor torque, stored spring energy and closed-chain/contact coupling.',
        'No one-variable proof separating learned weights, branch blend and final target limiter.',
        'No hardware speed limit inferred from a training soft penalty.'],
    new_PPO=0, qualification=None, hardware_started=False, created_unix=time.time())
save('velocity_and_standing_diagnosis001.json', out)
print(json.dumps(dict(matched=matched['robust'], historical_speed=out['historical_peak']['speed_rad_s'],
    stopped_speed=stopped['recorded_peak_rad_s'], standing=standing), indent=2))
