"""Pair native states/domains before comparing strict support and vibration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(before_path, after_path, output, kd_factor):
    before, after = [json.loads(p.read_text()) for p in (before_path, after_path)]
    runs = [Path(x['run']) for x in (before, after)]
    reports = [json.loads((p / 'evaluation/report.json').read_text()) for p in runs]
    starts = [np.load(p / 'evaluation_precontrol_state.npz') for p in runs]
    checks = dict(same_seed=reports[0]['seed'] == reports[1]['seed'],
        same_seconds=before['seconds'] == after['seconds'],
        same_native_rates=all(reports[0][k] == reports[1][k] for k in ('policy_hz', 'physics_hz')),
        same_case_order=[r['name'] for r in before['rows']] == [r['name'] for r in after['rows']],
        exact_initial_state=starts[0].files == starts[1].files and all(np.array_equal(starts[0][k], starts[1][k]) for k in starts[0].files),
        exact_domains_materials_and_scene=all(all(x[k] == y[k] for k in ('material_coefficients', 'world_origin_m', 'pose_index', 'command', 'imu_mount'))
            for x, y in zip(reports[0]['lanes'], reports[1]['lanes'], strict=True)),
        same_grounded_reset=reports[0]['evaluation_overrides']['grounded_standing_reset'] == reports[1]['evaluation_overrides']['grounded_standing_reset'])
    checks['all_other_domain_parameters_exact'] = all(
        x['first_reset_domain'].keys() == y['first_reset_domain'].keys() and all(x['first_reset_domain'][k] == y['first_reset_domain'][k]
            for k in x['first_reset_domain'] if k != 'leg_kd_scale')
        for x, y in zip(reports[0]['lanes'], reports[1]['lanes'], strict=True))
    checks['only_declared_kd_factor'] = all(np.allclose(
        np.asarray(x['first_reset_domain']['leg_kd_scale']) * kd_factor,
        np.asarray(y['first_reset_domain']['leg_kd_scale']), rtol=0., atol=2e-7)
        for x, y in zip(reports[0]['lanes'], reports[1]['lanes'], strict=True))
    if not all(checks.values()):
        raise ValueError('Native comparison pairing failed: ' + repr(checks))
    rows = []
    for x, y in zip(before['rows'], after['rows'], strict=True):
        rows.append(dict(name=x['name'], before=x, after=y,
            whole_contact_fraction_delta=y['whole']['dual_support_fraction'] - x['whole']['dual_support_fraction'],
            tail_contact_fraction_delta=y['tail4s']['dual_support_fraction'] - x['tail4s']['dual_support_fraction'],
            vibration_rms_ratio=y['tail_leg_motion_3_95hz_rms_rad'] / max(1e-12, x['tail_leg_motion_3_95hz_rms_rad'])))
    result = dict(schema='native-paired-leg-kd-standing-comparison-v1', intentional_native_kd_factor=kd_factor, checks=checks, rows=rows,
        before_analysis_sha256=digest(before_path), after_analysis_sha256=digest(after_path),
        before_passed=before['passed_continuous_support_and_geometry'], after_passed=after['passed_continuous_support_and_geometry'],
        cases=len(rows), improved_contact_cases=sum(r['whole_contact_fraction_delta'] > 1e-10 for r in rows),
        degraded_contact_cases=sum(r['whole_contact_fraction_delta'] < -1e-10 for r in rows),
        lower_vibration_cases=sum(r['vibration_rms_ratio'] < 1. for r in rows),
        qualification=None, hardware_ready=False)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))
    for r in rows:
        print(r['name'], 'support', round(r['before']['whole']['dual_support_fraction'], 4), '->', round(r['after']['whole']['dual_support_fraction'], 4),
              'vibration_ratio', round(r['vibration_rms_ratio'], 3), 'late_gap_mm', round(r['after']['tail4s']['maximum_clearance_m'] * 1000, 4))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kd-factor', type=float, required=True)
    a = p.parse_args()
    compare(a.before, a.after, a.output, a.kd_factor)
