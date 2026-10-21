#!/usr/bin/env python3
"""Account for every paired response episode and export metrics to events."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from flat_transient_metrics import assess_response, condition_summary, lane_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-role', default='trained')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous response evidence')
    source = args.input/'report.json'
    report = json.loads(source.read_text())
    if report['status'] != 'evaluated' or not report.get('transient_response_protocol'):
        raise ValueError('Require completed native response protocol')
    contract = json.loads((args.input/'contract.json').read_text())
    cases = {c['name']: c for c in contract['evaluation']['cases']}
    summaries, episodes = {}, {}
    with np.load(args.input/'traces.npz') as archive:
        # NPZ access decompresses a whole member; cache each diagnostic once.
        data = {key: archive[key] for key in ('groups', 'episode_ticks', 'active', 'done',
            'velocity', 'omega', 'position', 'gravity', 'motor_velocity', 'motor_position', 'gap')}
        groups = data['groups']
        for index, arm in report['checkpoint_comparison'].items():
            rows, details = {}, {}
            for name, case in cases.items():
                if not name.startswith(f'g{index}_'):
                    continue
                ids = np.flatnonzero(groups == name)
                lanes = [lane_metrics(data, int(lane), case, 1/report['policy_hz'], report['horizon_seconds']) for lane in ids]
                row = condition_summary(lanes)
                if row['complete'] != report['result']['cases'][name]['full_horizon_episodes']:
                    raise ValueError('Independent response completion differs from native grading')
                row['case_name'] = case['original_case']
                rows[name.split('_', 1)[1]] = row
                details[name] = lanes
            summaries[arm['role']], episodes[arm['role']] = rows, details
    result = dict(schema='flat-transient-native-response-v1', native_report_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        seed=report['seed'], policy_hz=report['policy_hz'], physics_hz=report['physics_hz'],
        checkpoint_comparison=report['checkpoint_comparison'], summaries=summaries, episodes=episodes,
        admission=assess_response(report, summaries, args.target_role),
        timing='post-step observations, request from preceding policy tick; .25s continuous combined translation/yaw stop dwell; 90% launch sustained .1s',
        stop_distance='XY path length from preceding sample to settled-band entry; censored paths never count as successful stops',
        missing_data='incomplete phases have null final metrics; no survivor-only admission')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(args.input), filename_suffix='.transient_response') as writer:
        for arm in report['checkpoint_comparison'].values():
            for condition, row in summaries[arm['role']].items():
                for metric, value in row.items():
                    if isinstance(value, (float, int)):
                        writer.add_scalar(f'FlatTransientEvaluation/{arm["role"]}/{condition}/{metric}', value, arm['successful_updates'])
    print(json.dumps(dict(status='response_analyzed', admission=result['admission'])))


if __name__ == '__main__':
    main()
