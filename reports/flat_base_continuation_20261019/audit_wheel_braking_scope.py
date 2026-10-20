"""Read existing native stop failures and test the new head's actual gradient scope."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint

torch.set_num_threads(1)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
out = HERE / 'wheel_braking_scope001.json'
if out.exists():
    raise SystemExit('Preserve the completed scope audit')
checkpoint = RUN / 'pilot027_from1280/model_001296.pt'
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
mean = mean_from_checkpoint(state, ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx')
rows, checks = {}, {}
for cohort in ('wheel_from1280_default001', 'wheel_from1280_maximum001'):
    evaluation_path = RUN / cohort / 'evaluation.json'
    evaluation = read(evaluation_path)
    checks[cohort + '_exact_checkpoint'] = evaluation['checkpoint_sha256'] == sha(checkpoint)
    trace_path = RUN / cohort / 'traces.npz'
    cases = {c['name']: c for c in evaluation['settings']['cases']}
    with np.load(trace_path) as trace:
        for case in ('start_stop_1', 'start_stop_2', 'start_stop_3'):
            lanes = trace['groups'] == case
            active = trace['active'][:, lanes]
            obs = trace['observations'][:, lanes]
            speed = np.linalg.norm(trace['velocity'][:, lanes, :2], axis=-1)
            request_zero = (abs(trace['raw_requested_commands'][:, lanes, :2]) < 1e-6).all(-1)
            ages = np.zeros(request_zero.shape, dtype=np.int64)
            for tick in range(len(ages)):
                ages[tick] = np.where(request_zero[tick], ages[tick - 1] + 1 if tick else 1, 0)
            settle = max(cases[case]['skill'].get('stop_settle_seconds', 1.),
                cases[case]['skill'].get('velocity_transition_seconds', 0.))
            settled = active & request_zero & (ages * .02 > settle)
            mask = settled & (speed > cases[case]['settled_stop_speed_max'])
            with torch.no_grad():
                gate = mean.authority_gate(torch.from_numpy(obs.reshape(-1, 35).copy())).numpy().reshape(active.shape)
            closed = mask & (gate == 0.)
            frames, closed_frames = int(mask.sum()), int(closed.sum())
            gradient_norm = None
            extra_max = None
            if closed_frames:
                selected = torch.from_numpy(obs[closed].copy())
                extra = mean.authority_actions(selected)
                gradients = torch.autograd.grad(extra[:, 4:].sum(), tuple(mean.wheel_authority.parameters()))
                gradient_norm = sum(float(g.double().square().sum()) for g in gradients) ** .5
                extra_max = float(extra.detach().abs().max())
            key = cohort + '/' + case
            official = evaluation['result']['cases'][case]['settled_stop_speed']
            peak = float(speed[settled].max())
            checks[key + '_settled_peak_matches_native_grade'] = abs(peak - official) < 1e-7
            checks[key + '_closed_failure_direct_head_gradient_zero'] = not closed_frames or (
                extra_max == 0. and gradient_norm == 0.)
            checks[key + '_settle_clock_and_coverage'] = settle == 1. and int(lanes.sum()) == 4
            rows[key] = dict(
                evaluation_sha256=sha(evaluation_path), trace_sha256=sha(trace_path),
                settled_sample_frames=int(settled.sum()), failed_settled_sample_frames=frames,
                failed_frames_with_zero_new_head_gate=closed_frames,
                fraction_of_failed_frames_with_zero_gate=closed_frames / frames if frames else None,
                native_settled_speed_peak_m_s=official, reconstructed_settled_speed_peak_m_s=peak,
                zero_gate_new_head_abs_action_max=extra_max, zero_gate_new_head_gradient_norm=gradient_norm,
            )
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
out.write_text(json.dumps(dict(schema='flat-wheel-v6-native-braking-learning-scope-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks, rows=rows,
    checkpoint_sha256=sha(checkpoint), actor_source_sha256=sha(ROOT / 'src/wheeled_algo/wheel_authority.py'),
    native_stop_clock_source_sha256=sha(ROOT / 'src/wheeled_tasks/chassis/skill_commands.py'),
    reader_sha256=sha(Path(__file__)), new_optimizer_updates=0, new_native_steps=0,
    heldout_seed_used=False, behavior_qualified=False,
    mask='Existing raw zero-command intervals lasting >1s; reconstructed peak must equal the native grade.',
    scope='Recorded pre-action observations aligned to each post-step velocity. Zero gate blocks direct new-head learning at these inputs; earlier gated actions may still change later trajectories. This is not proof of global unlearnability or a sole cause.'),
    indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(passed=True, checks=len(checks), rows=rows), indent=2))
