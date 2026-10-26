"""Describe full-window drift and reference availability in paired development failures."""
import json
import sys
import time
from pathlib import Path

sys.path[:0] = ['src', 'scripts']
import numpy as np
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
names = ['stand_low__roll_plus2', 'stand__roll_plus2',
         'stand_high__pitch_minus2', 'stand_high__combined']
rows = []
provenance = []
for label, run in [('current9a', 'diagnosis039_current9a_development'),
                   ('quiet050', 'diagnosis038_quiet_stance050_development')]:
    path = B / run / 'evaluation'
    report = json.loads((path / 'report.json').read_text())
    trace = path / 'trajectory.npz'
    assert digest(trace) == report['trajectory_sha256']
    a = np.load(trace)
    provenance.append(dict(trace=str(trace), sha256=digest(trace)))
    for lane in report['lanes']:
        if lane['name'] not in names:
            continue
        i = lane['lane']
        ids = np.flatnonzero(a['alive'][:, i])
        t = a['time'][ids, i]
        xy = a['local_position'][ids, i, :2]
        radius = np.linalg.norm(xy - xy[0], axis=1)
        active = a['standing_reference_active'][ids, i]
        valid = a['standing_reference_output_valid'][ids, i]
        coordinate = a['observations'][ids, i, 14:16] / 20.
        rows.append(dict(source=label, name=lane['name'], observed_until_s=float(t[-1]),
            final_delta_xy_m=(xy[-1] - xy[0]).tolist(), max_excursion_m=float(radius.max()),
            max_excursion_time_s=float(t[radius.argmax()]),
            reference_first_active_s=float(t[np.flatnonzero(active)[0]]) if active.any() else None,
            reference_valid_fraction=float(valid.mean()),
            severe_reference_resets=int(a['standing_reference_severe_reset_count'][ids, i].max()),
            final_two_wheel_odometric_coordinates_m=coordinate[-1].tolist(),
            integrated_measured_physical_yaw_rad=float(np.trapezoid(a['omega'][ids, i, 2], t)),
            excursions_at_integer_seconds_m=[float(radius[np.abs(t - second).argmin()]) for second in range(1, 9)]))
out = dict(schema='paired-standing-reference-observation-v1', rows=rows, provenance=provenance,
    interpretation='Reference is available for most of these traces; initial and intermediate excursion can fail even if late body rates are small. Two wheel odometric coordinates are not world XY or proof of slip/observer error. No causal or qualification claim.',
    new_PPO=0, qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'standing_reference_observed_audit001.json', out)
print(json.dumps(dict(rows=len(rows), traces=len(provenance), minimum_reference_valid_fraction=min(x['reference_valid_fraction'] for x in rows),
    peak_excursion_times_s=[x['max_excursion_time_s'] for x in rows], new_PPO=0), indent=2))
