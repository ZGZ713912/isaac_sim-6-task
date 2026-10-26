"""Verify and publish native1ms support, geometry and standing vibration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interruptions(missing):
    edges = np.diff(np.r_[False, missing, False].astype(np.int8))
    start, end = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return dict(interruptions=int(len(start)), maximum_interruption_ms=int((end - start).max()) if len(start) else 0,
                first_missing_support_s=float((start[0] + 1) * .001) if len(start) else None)


def analyze(run, output):
    folder = run / 'evaluation'
    meta = json.loads((folder / 'native_wheel_contact_1ms.json').read_text())
    report = json.loads((folder / 'report.json').read_text())
    contacts, trajectory = folder / 'native_wheel_contact_1ms.npz', folder / 'trajectory.npz'
    if digest(contacts) != meta['sha256'] or digest(trajectory) != report['trajectory_sha256']:
        raise ValueError('Native support/trajectory evidence changed')
    data, trace = np.load(contacts), np.load(trajectory)
    force, gap = data['force'], data['clearance']
    if force.shape[:2] != (len(trace['time']), 5) or force.shape[:-1] != gap.shape:
        raise ValueError('Native contact axes or decimation changed')
    if not np.isfinite(force).all() or not np.isfinite(gap).all():
        raise ValueError('Nonfinite native support evidence')
    rows = []
    clock = json.loads((run / 'progress.json').read_text())['actor_settings'].get('origin_updates', 0)
    writer = SummaryWriter(str(folder / 'metrics'), filename_suffix='.strict-standing-contact')
    for lane in report['lanes']:
        i = lane['lane']
        f, g = force[:, :, i].reshape(-1, 2, 3), gap[:, :, i].reshape(-1, 2)
        supported = (np.linalg.norm(f, axis=-1) > 2.).all(-1)
        row = dict(name=lane['name'], lane=i, survived=lane['survived'])
        for name, lo in [('whole', 0), ('tail4s', max(0, len(g) - 4000))]:
            s, gg, ff = supported[lo:], g[lo:], f[lo:]
            row[name] = dict(dual_support_fraction=float(s.mean()), all_ms_supported=bool(s.all()),
                maximum_clearance_m=float(gg.max()), fraction_ms_clearance_over_0_1mm=float((gg > .0001).any(-1).mean()),
                minimum_vertical_load_n=float(ff[..., 2].min()),
                **interruptions(~s))
        row['first_20ms_dual_support_fraction'] = float(supported[:20].mean())
        row['first_20ms_maximum_clearance_m'] = float(g[:20].max())
        row['continuous_support_and_geometry'] = bool(lane['survived'] and supported.all() and g.max() <= .0001)
        tail = trace['time'][:, i] >= report['seconds'] - min(4., report['seconds'])
        q = trace['motor_position'][tail, i][:, [0, 1, 3, 4]]
        spectrum = np.fft.rfft(q - q.mean(0), axis=0)
        frequencies = np.fft.rfftfreq(len(q), .005)
        spectrum[(frequencies < 3.) | (frequencies > 95.)] = 0.
        row['tail_leg_motion_3_95hz_rms_rad'] = float(np.sqrt(np.mean(np.fft.irfft(spectrum, n=len(q), axis=0)**2)))
        row['tail_leg_speed_rms_rad_s'] = float(np.sqrt(np.mean(trace['motor_velocity'][tail, i][:, [0, 1, 3, 4]]**2)))
        row['whole_physical_leg_speed_peak_rad_s'] = float(trace['leg_speed_peak_rad_s'][:, i].max())
        for window in ('whole', 'tail4s'):
            for key, value in row[window].items():
                if isinstance(value, (int, float)):
                    writer.add_scalar('StrictStandingEvaluation/' + row['name'] + '/' + window + '/' + key, value, clock)
        for key in ('tail_leg_motion_3_95hz_rms_rad', 'tail_leg_speed_rms_rad_s', 'whole_physical_leg_speed_peak_rad_s'):
            writer.add_scalar('StrictStandingEvaluation/' + row['name'] + '/' + key, row[key], clock)
        rows.append(row)
    writer.close()
    result = dict(schema='strict-standing-1ms-analysis-v1', run=str(run), seconds=report['seconds'], rows=rows,
        force_threshold_n=2., geometry_reporting_resolution_m=.0001, raw_clearance_retained=True,
        no_startup_grace=True, reset=meta.get('reset'), contact_data_sha256=meta['sha256'], trajectory_sha256=digest(trajectory),
        passed_continuous_support_and_geometry=sum(r['continuous_support_and_geometry'] for r in rows),
        cases=len(rows), qualification=None, hardware_ready=False)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), cases=len(rows), passed_support=result['passed_continuous_support_and_geometry'],
        first20ms_supported=sum(r['first_20ms_dual_support_fraction'] == 1. for r in rows),
        late100percent=sum(r['tail4s']['all_ms_supported'] for r in rows), reset=meta.get('reset'))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    analyze(a.run, a.output)
