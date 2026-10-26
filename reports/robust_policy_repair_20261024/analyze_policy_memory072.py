"""Read-only local feedback derivatives on authenticated native standing states.

The previous-action Jacobian holds all sensor inputs fixed. Its spectral radius
does not establish stability of the robot, sensors, delay or contact dynamics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from export_robust_getup import load_mean
from wheeled_tasks.chassis.scut_observation import POLICY_FROM_CONTROL


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run, output):
    torch.set_num_threads(1)
    folder = run / 'evaluation'
    report = json.loads((folder / 'report.json').read_text())
    progress = json.loads((run / 'progress.json').read_text())
    checkpoint = Path(progress['checkpoint'])
    assert sha(checkpoint) == progress['checkpoint_sha256'] == report['checkpoint_sha256']
    assert sha(folder / 'trajectory.npz') == report['trajectory_sha256']
    trace = np.load(folder / 'trajectory.npz')
    mean, info = load_mean(checkpoint)
    rows = []
    writer = SummaryWriter(str(folder / 'metrics'), filename_suffix='.local-memory-jacobian')
    for lane in report['lanes']:
        i = lane['lane']
        indices = np.flatnonzero(trace['time'][:, i] >= report['seconds'] - 4.)[::20]
        obs = torch.tensor(trace['observations'][indices, i], requires_grad=True)
        actions = mean(obs)
        actual = trace['actions'][indices, i][:, POLICY_FROM_CONTROL]
        error = float(np.max(np.abs(actions.detach().numpy() - actual)))
        assert error < 5e-5, (lane['name'], error)
        jac = torch.stack([torch.autograd.grad(actions[:, j].sum(), obs,
            retain_graph=True)[0] for j in range(6)], 1).detach().numpy()
        radius = np.abs(np.linalg.eigvals(jac[:, :, 22:28])).max(-1)
        # Physical leg targets are action[0:4] * .25 radians. Encoder velocity
        # observations are .1 * rad/s and gyro observations .5 * rad/s.
        velocity_gain = np.linalg.svd(.025 * jac[:, :4, 16:20], compute_uv=False)[:, 0]
        gyro_gain = np.linalg.svd(.125 * jac[:, :4, 4:7], compute_uv=False)[:, 0]
        row = dict(name=lane['name'], frames=len(indices), action_replay_error=error,
            memory_radius_median=float(np.median(radius)), memory_radius_max=float(radius.max()),
            memory_fraction_radius_gt_one=float((radius > 1.).mean()),
            leg_velocity_to_target_gain_max_seconds=float(velocity_gain.max()),
            gyro_to_leg_target_gain_max_seconds=float(gyro_gain.max()))
        rows.append(row)
        for key, value in row.items():
            if isinstance(value, float):
                writer.add_scalar('LocalPolicyFeedback/' + lane['name'] + '/' + key, value,
                    info['completed_updates'])
    writer.close()
    result = dict(schema='read-only-local-policy-memory-derivatives-v1', checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), trajectory_sha256=report['trajectory_sha256'],
        learning_clock=info['completed_updates'], rows=rows, new_PPO=0,
        scope='Sensors held fixed: local actor memory only; excludes plant, contact, sensor dynamics and delay.',
        stability_or_causality_claim=False, qualification=None)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), cases=len(rows),
        maximum_memory_radius=max(row['memory_radius_max'] for row in rows),
        cases_with_radius_gt_one=sum(row['memory_fraction_radius_gt_one'] > 0. for row in rows))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.output)
