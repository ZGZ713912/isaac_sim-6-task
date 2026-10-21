"""Physical response measurements for the prospective 3/-2.5m/s campaign."""
from copy import deepcopy
import numpy as np

RETENTION = ('stand_305mm', 'forward_05', 'backward_05', 'rotate_1', 'rotate_1_reverse')
STEADY = ('forward_3', 'backward_2p5', 'rotate_4', 'rotate_4_reverse')
TRANSIENT = ('start_stop_3_to_2p5', 'backward_start_stop_2p5_to_3',
             'yaw_start_stop_positive', 'yaw_start_stop_negative')


def response_cases(cases):
    from wheeled_tasks.chassis.transient_response_plan import response_envelope
    rows = {c['name']: c for c in response_envelope(cases)}
    reverse = deepcopy(rows['start_stop_3_to_2p5'])
    reverse.update(name='backward_start_stop_2p5_to_3', command=[-2.5, 0., .305])
    reverse['skill'].update(command=deepcopy(reverse['command']), opposite_speed_ratio=6./5.)
    rows[reverse['name']] = reverse
    result = []
    for name in RETENTION + STEADY + TRANSIENT:
        row = deepcopy(rows[name])
        row['skill']['sample_amplitude'] = False
        if row['skill']['kind'] == 'start_stop':
            row['skill']['segment_seconds'] = 4.
        if name.startswith('yaw_start_stop'):
            row['skill']['yaw_start_stop_segment_seconds'] = 4.
        result.append(row)
    return result


def first_dwell(condition, dt, seconds=.25):
    """Time of entering a band, verified by a continuous dwell within the phase."""
    width = max(1, round(seconds / dt))
    condition = np.asarray(condition, dtype=bool)
    if len(condition) < width:
        return None
    cumulative = np.r_[0, np.cumsum(condition, dtype=np.int64)]
    hits = np.flatnonzero(cumulative[width:] - cumulative[:-width] == width)
    return float(hits[0] * dt) if len(hits) else None


def lane_metrics(data, lane, case, dt, horizon):
    ticks = data['episode_ticks'][:, lane]
    active = data['active'][:, lane].astype(bool)
    done = data['done'][:, lane].astype(bool)
    complete = bool(active[-1] and ticks[-1] >= round(horizon / dt))
    valid = active & (~done | ((ticks >= round(horizon / dt)) & complete))
    elapsed = np.maximum(ticks - 1, 0) * dt
    velocity, omega = data['velocity'][:, lane], data['omega'][:, lane]
    positions = data['position'][:, lane, :2]
    pitch = np.abs(np.arctan2(data['gravity'][:, lane, 0], -data['gravity'][:, lane, 2]))
    dq = data['motor_velocity'][:, lane]
    roll_speed = (dq[:, 2] - dq[:, 5]) * .03
    wheel_yaw = (-dq[:, 5] - dq[:, 2]) * .06 / .4373
    leg_q = np.unwrap(data['motor_position'][:, lane][:, [0, 1, 3, 4]], axis=0)
    ids = np.flatnonzero(valid)
    row = dict(complete=complete, valid_frames=len(ids),
        pitch_peak_deg=float(np.degrees(pitch[valid]).max()) if len(ids) else None,
        slip_mean_m_s=float(np.abs(roll_speed[valid] - velocity[valid, 0]).mean()) if len(ids) else None,
        wheel_yaw_slip_rad_s=float(np.abs(wheel_yaw[valid] - omega[valid, 2]).mean()) if len(ids) else None,
        leg_excursion_max_rad=float(np.ptp(leg_q[valid], axis=0).max()) if len(ids) else None,
        leg_speed_mean_rad_s=float(np.abs(dq[valid][:, [0, 1, 3, 4]]).mean()) if len(ids) else None,
        closure_gap_peak_m=float(data['gap'][active, lane].max()) if active.any() else None,
        phases=[])
    if case['original_case'] not in TRANSIENT:
        return row
    from analyze_flat_raw_requests import requested_planar
    request = requested_planar(ticks, case, dt)
    yaw_task = case['original_case'].startswith('yaw_')
    actual = omega[:, 2] if yaw_task else velocity[:, 0]
    speed = np.linalg.norm(velocity[:, :2], axis=-1)
    for phase in range(4):
        selected = (elapsed >= phase*4.) & (elapsed < (phase+1)*4.) & valid
        indices = np.flatnonzero(selected)
        full_phase = len(indices) >= round(4./dt) - 2
        entry = dict(index=phase, full_phase=full_phase, observed_frames=len(indices))
        if not len(indices):
            entry.update(kind='stop' if phase % 2 else 'launch', censored=True)
            row['phases'].append(entry)
            continue
        target = float(request[indices[0], 1 if yaw_task else 0])
        entry['target'] = target
        if phase % 2:
            settled = (speed[indices] <= .05) & (np.abs(omega[indices, 2]) <= .1)
            time = first_dwell(settled, dt)
            entry.update(kind='stop', settle_time_s=time, censored=time is None)
            end = indices[-1] if time is None else indices[min(round(time/dt), len(indices)-1)]
            start = max(int(indices[0])-1, 0)
            entry['travel_distance_m'] = float(np.linalg.norm(np.diff(positions[start:end+1], axis=0), axis=-1).sum())
            final = indices[-max(1, round(.5/dt)):]
            entry['final_planar_m_s'] = float(speed[final].mean()) if full_phase else None
            entry['final_yaw_abs_rad_s'] = float(np.abs(omega[final, 2]).mean()) if full_phase else None
        else:
            signed = actual[indices] * np.sign(target)
            time = first_dwell(signed >= .9*abs(target), dt, .1)
            entry.update(kind='launch', response90_s=time, censored=time is None,
                         overshoot=float(max(0., signed.max() - abs(target))))
        row['phases'].append(entry)
    return row


def condition_summary(lanes):
    phases = [p for lane in lanes for p in lane['phases']]
    stops = [p for p in phases if p['kind'] == 'stop']
    launches = [p for p in phases if p['kind'] == 'launch']
    def mean(rows, key):
        values = [row.get(key) for row in rows]
        return float(np.mean(values)) if values and all(v is not None for v in values) else None
    return dict(complete=sum(l['complete'] for l in lanes), requested=len(lanes),
        stop_count=len(stops), censored_stops=sum(p['censored'] for p in stops),
        censored_launches=sum(p['censored'] for p in launches),
        settle_time_s=mean(stops, 'settle_time_s'), response90_s=mean(launches, 'response90_s'),
        stop_distance_m=mean(stops, 'travel_distance_m'),
        final_planar_m_s=mean(stops, 'final_planar_m_s'),
        final_yaw_abs_rad_s=mean(stops, 'final_yaw_abs_rad_s'),
        pitch_peak_deg=mean(lanes, 'pitch_peak_deg'),
        leg_excursion_max_rad=mean(lanes, 'leg_excursion_max_rad'),
        leg_speed_mean_rad_s=mean(lanes, 'leg_speed_mean_rad_s'),
        slip_mean_m_s=mean(lanes, 'slip_mean_m_s'))


def assess_response(native, summaries, target_role='trained'):
    """Prospective admission; no loss-based or survivor-only promotion."""
    parent = summaries['parent']
    candidate = summaries[target_role]
    reasons, retention_lost, improvements = [], [], []
    arms = {value['role']: key for key, value in native['checkpoint_comparison'].items()}
    old_arm, new_arm = arms['parent'], arms[target_role]
    for condition, row in candidate.items():
        old = parent[condition]
        name = row['case_name']
        native_new = native['result']['cases'][f'g{new_arm}_{condition}']
        native_old = native['result']['cases'][f'g{old_arm}_{condition}']
        if row['complete'] != row['requested']:
            reasons.append(condition + ': incomplete episodes')
        if native_new['closure_gap_max_m'] > .003 or native_new['tilt_max_deg'] > 15.:
            reasons.append(condition + ': mechanics or tilt')
        if name in RETENTION + STEADY:
            lost = [key for key, passed in native_old['checks'].items() if passed and not native_new['checks'].get(key)]
            if lost:
                retention_lost.append(dict(condition=condition, checks=lost))
            for metric, tolerance in [('vx_mae_m_s', .02), ('yaw_mae_rad_s', .05), ('height_mae_m', .003)]:
                if native_new[metric] > native_old[metric] + tolerance:
                    reasons.append(condition + ': retention ' + metric)
        if name in STEADY:
            if native_new['vx_mae_m_s'] > .3 or native_new['yaw_mae_rad_s'] > .3:
                reasons.append(condition + ': steady command tracking')
        if name in TRANSIENT:
            if row['censored_stops'] or row['censored_launches']:
                reasons.append(condition + ': censored response')
            if row['final_planar_m_s'] is None or row['final_planar_m_s'] > .05:
                reasons.append(condition + ': stop translation')
            if row['final_yaw_abs_rad_s'] is None or row['final_yaw_abs_rad_s'] > .1:
                reasons.append(condition + ': stop yaw')
            for metric in ('settle_time_s', 'response90_s', 'stop_distance_m'):
                before, after = old.get(metric), row.get(metric)
                if before is not None and after is not None and before > .02:
                    improvements.append((before-after)/before)
                elif before is None and after is not None:
                    improvements.append(1.)
    if retention_lost:
        reasons.append('lost passing baseline retention checks')
    gain = float(np.median(improvements)) if improvements else None
    if gain is None or gain < .1:
        reasons.append('median matched response gain below 10 percent')
    return dict(admitted=not reasons, reasons=reasons, lost_retention_checks=retention_lost,
        median_response_relative_gain=gain, response_metric_comparisons=len(improvements),
        qualification=False, hardware_ready=False)
