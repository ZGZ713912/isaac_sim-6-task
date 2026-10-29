#!/usr/bin/env python3
"""Keep human motion requests separate from RMCS's feasible actor reference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def requested_planar(ticks, case, policy_dt):
    """Reconstruct the request used for this action, before heading projection.

    Diagnostics contain the incremented episode tick. The action was computed
    using the command assembled at the preceding policy tick. Amplitude sampling
    is absent from the deterministic profile_command used by this replay.
    """
    elapsed = np.maximum(ticks - 1, 0).astype(np.float32) * np.float32(policy_dt)
    command = np.asarray(case['command'][:2], dtype=np.float32)
    request = np.broadcast_to(command, (*ticks.shape, 2)).copy()
    skill = case.get('skill', {})
    seconds = skill.get('yaw_start_stop_segment_seconds')
    if seconds is not None:
        if skill.get('kind') != 'rotate':
            raise ValueError('Unsupported yaw request profile')
        phase = np.floor(elapsed / seconds).astype(int) % 4
        request[..., 1] *= np.where(phase == 0, 1., np.where(phase == 2, -1., 0.))
    elif skill.get('kind') == 'start_stop':
        phase = np.floor(elapsed / skill.get('segment_seconds', 3.)).astype(int) % 4
        request[..., 0] *= np.where(phase == 0, 1., np.where(phase == 2, -skill.get('opposite_speed_ratio', 1.), 0.))
    elif skill.get('kind') not in ('stand', 'forward', 'backward', 'rotate', 'curve', 'height'):
        raise ValueError('Unsupported raw motion profile')
    return request


def motion_stats(actual, issued, request, mask):
    count = int(mask.sum())
    if not count:
        return dict(frames=0, actual_mean=None, requested_mean=None,
                    issued_mean=None, raw_mae=None, issued_mae=None,
                    changed_reference_fraction=None)
    actual, issued, request = actual[mask], issued[mask], request[mask]
    return dict(frames=count, actual_mean=float(actual.mean()),
                requested_mean=float(request.mean()), issued_mean=float(issued.mean()),
                raw_mae=float(np.abs(actual-request).mean()),
                issued_mae=float(np.abs(actual-issued).mean()),
                changed_reference_fraction=float((np.abs(issued-request) > 1e-4).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the previous raw-request analysis')
    source = args.input/'report.json'
    report = json.loads(source.read_text())
    if report['status'] != 'evaluated' or report['probe_mode'] != 'checkpoint_compare':
        raise ValueError('Require a completed native checkpoint comparison')
    contract = json.loads((args.input/'contract.json').read_text())
    policy_dt = 1/report['policy_hz']
    cases = {c['name']: c for c in contract['evaluation']['cases']}
    summaries = {}
    with np.load(args.input/'traces.npz') as data:
        groups = data['groups']
        ticks = data['episode_ticks']
        active = data['active'].astype(bool)
        valid = active & ~data['done'].astype(bool)
        commands = data['commands']
        velocity, omega = data['velocity'][..., 0], data['omega'][..., 2]
        full = active[-1] & (ticks[-1] >= round(report['horizon_seconds']/policy_dt))
        for index, arm in report['checkpoint_comparison'].items():
            rows = {}
            for name, case in cases.items():
                if not name.startswith(f'g{index}_'):
                    continue
                ids = np.flatnonzero(groups == name)
                if int(full[ids].sum()) != report['result']['cases'][name]['full_horizon_episodes']:
                    raise ValueError('Trace completion and frozen case grading disagree')
                request = requested_planar(ticks[:, ids], case, policy_dt)
                elapsed = ticks[:, ids] * policy_dt
                warmup = case.get('warmup_seconds', contract['evaluation']['warmup_seconds'])
                masks = dict(after_warmup=valid[:, ids] & (elapsed >= warmup),
                    last_five_seconds_complete_episodes=valid[:, ids] & full[ids][None, :]
                    & (elapsed >= report['horizon_seconds']-5.))
                row = dict(full_episodes=int(full[ids].sum()), requested_episodes=len(ids))
                for label, mask in masks.items():
                    row[label] = dict(
                        vx_m_s=motion_stats(velocity[:, ids], commands[:, ids, 0], request[..., 0], mask),
                        yaw_rad_s=motion_stats(omega[:, ids], commands[:, ids, 1], request[..., 1], mask))
                if case.get('skill', {}).get('kind') == 'start_stop':
                    seconds = case['skill'].get('segment_seconds', 3.)
                    action_elapsed = np.maximum(ticks[:, ids]-1, 0) * policy_dt
                    phase = (action_elapsed//seconds).astype(int) % 4
                    ground = case['contact_material_profile']['ground_dynamic_friction']
                    wheel = contract['contact_material']['wheel_dynamic_friction']
                    mu = (ground+wheel)/2
                    if contract['contact_material']['combine_mode'] != 'average':
                        raise ValueError('Traction diagnostic assumes the frozen average material combination')
                    allowance = max(1., abs(case['command'][0])/(mu*9.81)+.5)
                    mask = valid[:, ids] & ((phase == 1) | (phase == 3))
                    mask &= (action_elapsed % seconds) >= allowance
                    row['supplementary_stop'] = dict(effective_mu=mu, allowance_seconds=allowance,
                        model='quasi-static |delta_v|/(mu*g) plus .5s; diagnostic only, not a safety bound',
                        vx_m_s=motion_stats(velocity[:, ids], commands[:, ids, 0], request[..., 0], mask))
                rows[name] = row
            summaries[arm['role']] = rows
    result = dict(schema='flat-raw-request-diagnostic-v1', qualified=False,
        qualification_eligible=False, new_optimizer_updates=0,
        source_report_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scoring='Report raw request and issued RMCS reference separately. All errors are conditional on valid frames; missing frames are null. Original grading is unchanged.',
        summaries=summaries)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(args.input), filename_suffix='.raw_request_summary') as writer:
        for arm in report['checkpoint_comparison'].values():
            for name, row in summaries[arm['role']].items():
                for unit, values in row['after_warmup'].items():
                    for metric, value in values.items():
                        if value is not None:
                            writer.add_scalar(f'FlatRawRequest/{arm["role"]}/{name}/{unit}/{metric}', value,
                                              arm['successful_updates'])
    print(json.dumps(dict(status='raw_request_analysis_complete', arms=list(summaries))))


if __name__ == '__main__':
    main()
