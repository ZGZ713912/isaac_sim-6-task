#!/usr/bin/env python3
"""Report the paired heading intervention, including incomplete episodes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_flat_raw_requests import requested_planar


def complete_window_integrals(rate, valid, ticks, dt, begin, end):
    """Body gyro integrals for lanes that actually completed the whole window."""
    clock = (ticks - 1) * dt
    window = (clock >= begin) & (clock < end)
    expected = round((end - begin) / dt)
    output = []
    for lane in range(rate.shape[1]):
        mask = valid[:, lane] & window[:, lane]
        if int(mask.sum()) != expected:
            output.append(None)
        else:
            output.append(float(rate[mask, lane].sum() * dt))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the existing heading analysis')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report_file = args.input/'report.json'
    report = json.loads(report_file.read_text())
    override_file = args.input/'heading_memory_override.json'
    override = json.loads(override_file.read_text())
    completion_file = args.input/'heading_memory_completion001.json'
    completion = json.loads(completion_file.read_text()) if completion_file.is_file() else override
    if completion_file.is_file() and (
            completion['initial_override_sha256'] != sha(override_file)
            or completion['traces_sha256'] != sha(args.input/'traces.npz')
            or not completion['intervention_actor_inputs_observed']
            or not completion['same_native_onnx_graph']):
        raise ValueError('Recovered completion does not match the native evidence')
    if (report['status'] != 'evaluated' or not completion['same_candidate_weights']
            or completion['base_report_sha256'] != sha(report_file)
            or report['checkpoint_comparison']['2']['sha256'] != report['checkpoint_comparison']['3']['sha256']):
        raise ValueError('Require completed, authenticated same-weight heading intervention')
    contract = json.loads((args.input/'contract.json').read_text())
    cases = {c['name']: c for c in contract['evaluation']['cases']}
    dt = 1/report['policy_hz']
    summaries = {}
    with np.load(args.input/'traces.npz') as data:
        groups, ticks = data['groups'], data['episode_ticks']
        active, done = data['active'].astype(bool), data['done'].astype(bool)
        valid = active & ~done
        vx, yaw = data['velocity'][..., 0], data['omega'][..., 2]
        position = data['position']
        full = active[-1] & (ticks[-1] >= round(report['horizon_seconds']/dt))
        for name, case in cases.items():
            ids = np.flatnonzero(groups == name)
            graded = report['result']['cases'][name]
            if int(full[ids].sum()) != graded['full_horizon_episodes']:
                raise ValueError('Native trace and grader completion differ')
            raw = requested_planar(ticks[:, ids], case, dt)
            mask = valid[:, ids] & (ticks[:, ids]*dt >= case['warmup_seconds'])
            supported = bool(mask.any())
            row = dict(full_episodes=int(full[ids].sum()), requested_episodes=len(ids),
                       passed_all_checks=graded['passed'], termination_reasons=graded['termination_reasons'],
                       max_abs_transverse_y_m=float(np.abs(position[:, ids, 1][active[:, ids]]).max()),
                       post_warmup_frames=int(mask.sum()),
                       raw_vx_mae_m_s=float(np.abs(vx[:, ids]-raw[..., 0])[mask].mean()) if supported else None,
                       raw_yaw_mae_rad_s=float(np.abs(yaw[:, ids]-raw[..., 1])[mask].mean()) if supported else None)
            if case['original_case'] == 'start_stop_3':
                row['first_stop_8_to_16s_body_yaw_gyro_integral_rad'] = complete_window_integrals(
                    yaw[:, ids], valid[:, ids], ticks[:, ids], dt, 8., 16.)
            summaries[name] = row
    result = dict(schema='flat-stop-heading-analysis-v1', qualification_eligible=False, qualified=False,
                  full_training_admitted=False, new_optimizer_updates=0, policy_hz=report['policy_hz'],
                  seed=report['seed'], source_report_sha256=sha(report_file), override_sha256=sha(override_file),
                  recovered_completion_sha256=sha(completion_file) if completion_file.is_file() else None,
                  analyzer_sha256=sha(Path(__file__)),
                  scope='Same actor weights; only command heading memory differs between arm2 and arm3. Gyro integrals are body-axis measures, not absolute yaw. Incomplete windows are null; raw-request errors are conditional on valid frames.',
                  summaries=summaries)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(args.input), filename_suffix='.heading_memory_summary') as writer:
        for name, values in summaries.items():
            arm = name.split('_', 1)[0][1:]
            step = report['checkpoint_comparison'][arm]['successful_updates']
            for metric in ('full_episodes', 'requested_episodes', 'max_abs_transverse_y_m',
                           'raw_vx_mae_m_s', 'raw_yaw_mae_rad_s', 'post_warmup_frames'):
                if values[metric] is not None:
                    writer.add_scalar(f'FlatHeadingMemory/{name}/{metric}', values[metric], step)
            integral = values.get('first_stop_8_to_16s_body_yaw_gyro_integral_rad', [])
            supported = [abs(v) for v in integral if v is not None]
            if supported:
                writer.add_scalar(f'FlatHeadingMemory/{name}/first_stop_abs_gyro_integral_mean_rad',
                                  sum(supported)/len(supported), step)
    print(json.dumps(dict(status='heading_memory_analysis_complete', cases=len(summaries))))


if __name__ == '__main__':
    main()
