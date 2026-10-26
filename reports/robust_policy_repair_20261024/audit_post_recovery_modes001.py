"""Describe recorded motion modes without attributing contact failure to one mode."""
import json
import sys
import time
from pathlib import Path

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from analyze_robust_getup import hf_rms
from train_robust_getup import digest, write_json
from wheeled_algo.robust_getup import quiet_gate

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
models = []
for name in ('diagnosis025_motion_damping080_fixed', 'diagnosis041_entry_protected_wheels'):
    run = B / name
    report = json.loads((run / 'evaluation/report.json').read_text())
    trace = run / 'evaluation/trajectory.npz'
    assert digest(trace) == report['trajectory_sha256']
    z = np.load(trace)
    rows = []
    writer = SummaryWriter(str(run / 'evaluation/metrics'), filename_suffix='.post-recovery-modes')
    for lane in report['lanes']:
        if not lane['name'].startswith(('left_symmetric__', 'right_symmetric__', 'inverted_symmetric__')) or not lane['recovered']:
            continue
        i = lane['lane']
        mask = z['alive'][:, i] & ~z['passive'][:, i] & (z['time'][:, i] >= 4.)
        if mask.sum() < 400:
            continue
        q = np.unwrap(z['motor_position'][mask, i][:, [0, 1, 3, 4]], axis=0).reshape(-1, 2, 2)
        u = (z['actions'][mask, i][:, [0, 1, 3, 4]] * .25).reshape(-1, 2, 2)
        obs = torch.from_numpy(z['observations'][mask, i].copy())
        contact_bad = ~z['wheel_contact'][mask, i].all(-1)
        row = dict(name=lane['name'], recovered=True, followup=lane['followup_success'],
            frames=int(mask.sum()), contact_break_frames=int(contact_bad.sum()),
            quiet_gate_peak=float(quiet_gate(obs).max()),
            requested_forward_peak_m_s=float(np.abs(z['requested_command'][mask, i, 0]).max()),
            actual_forward_command_peak_m_s=float(np.abs(z['commands'][mask, i, 0]).max()),
            knee_motion_3_95hz_rms_rad=hf_rms((q[:, :, 1]-q[:, :, 0])*.5, .005, 3.),
            common_motion_3_95hz_rms_rad=hf_rms(q.mean(-1), .005, 3.),
            knee_target_3_95hz_rms_rad=hf_rms((u[:, :, 1]-u[:, :, 0])*.5, .005, 3.),
            common_target_3_95hz_rms_rad=hf_rms(u.mean(-1), .005, 3.),
            contact_break_clearance_peak_m=float(z['wheel_clearance_peak'][mask, i][contact_bad].max()) if contact_bad.any() else None,
            contact_break_air_run_peak_s=float(z['wheel_air_run_peak'][mask, i][contact_bad].max()) if contact_bad.any() else None)
        for key, value in row.items():
            if isinstance(value, (float, int)):
                writer.add_scalar('PostRecoveryModes/'+row['name']+'/'+key, value, 1501)
        rows.append(row)
    writer.close()
    models.append(dict(run=str(run), trajectory_sha256=digest(trace), rows=rows))
out = dict(schema='recorded-post-recovery-modes-v1', models=models,
    interpretation='Descriptive FFT/contact evidence only. Both paired common rotation and knee opening vary. Quiet authority is zero on these late moving frames; this does not exclude earlier quiet feedback changing arrival states. No unique causal claim.',
    next_hypothesis='Test knee-only motion increment damping while retaining common rotation, existing target bounds, recovery authority and wheels.',
    new_PPO=0, repair_optimizer_updates=258, actual_experiment_consumed=2638,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'post_recovery_modes001.json', out)
print(json.dumps(dict(models=[dict(run=m['run'], cases=len(m['rows']), all_quiet_zero=all(r['quiet_gate_peak']==0. for r in m['rows'])) for m in models]), indent=2))
