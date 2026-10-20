"""Measure learned wheel-head corrections on fixed original1280 high-speed inputs."""
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
load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
out = HERE / 'wheel_from1280_same_inputs001.json'
if out.exists():
    raise SystemExit('Preserve completed read-only probe')
parent_path = RUN / 'pilot027_from1280/initial_state.pt'
final_path = RUN / 'pilot027_from1280/model_001296.pt'
proof_path = HERE / 'wheel_from1280_native_final001.json'
proof = json.loads(proof_path.read_text())
if not proof['passed'] or proof['milestones']['1296']['checkpoint_sha256'] != sha(final_path):
    raise SystemExit('Require the authenticated final sixteen-update state')
source = ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx'
parent, final = load(parent_path), load(final_path)
before, after = (mean_from_checkpoint(s, source) for s in (parent, final))
checks = dict(
    only_new_wheel_state_changed=all(torch.equal(v, final['actor_state_dict'][n])
        for n, v in parent['actor_state_dict'].items() if not n.startswith('mlp.wheel_authority.')),
    original_wheel_bound=after.authority_bound.tolist() == [6., 6.],
)
rows = {}
for cohort in ('rmcs_default1280_envelope001', 'rmcs_maximum1280_envelope001'):
    path = RUN / cohort / 'traces.npz'
    evaluation = json.loads((RUN / cohort / 'evaluation.json').read_text())
    with np.load(path) as trace:
        for case in ('backward_3', 'forward_3'):
            lanes = trace['groups'] == case
            ticks = trace['episode_ticks'][:, lanes]
            mask = trace['active'][:, lanes] & (ticks >= 250) & (ticks < 450)
            obs = torch.from_numpy(trace['observations'][:, lanes][mask].copy())
            recorded = torch.from_numpy(trace['actions'][:, lanes][mask].copy())
            with torch.no_grad():
                initial, learned = before(obs), after(obs)
                extra = after.authority_actions(obs)
                gate = after.authority_gate(obs)
            key = cohort + '/' + case
            reproduction = (initial - recorded).abs().max().item()
            addition_error = ((learned - initial) - extra).abs().max().item()
            checks[key + '_fixed_input_scope'] = len(obs) == 800 and torch.all(gate == 1.).item()
            checks[key + '_native_initial_reproduction'] = reproduction < 2e-5
            checks[key + '_only_measured_extension'] = addition_error < 2e-6
            delta = .3 * (extra[:, 4] - extra[:, 5])
            rows[key] = dict(
                trace_sha256=sha(path), evaluation_sha256=sha(RUN / cohort / 'evaluation.json'),
                reference_checkpoint_sha256=evaluation['checkpoint_sha256'],
                sample_frames=len(obs), episode_time_window_s=[5., 9.],
                initial_action_native_max_error=reproduction,
                final_minus_initial_equals_wheel_head_max_error=addition_error,
                maximum_abs_extra_raw_wheel_action=extra[:, 4:].abs().max().item(),
                fraction_of_extra_raw_bound_used=extra[:, 4:].abs().max().item() / 6.,
                initial_unclipped_forward_rim_target_mean_m_s=(.3 * (initial[:, 4] - initial[:, 5])).mean().item(),
                final_unclipped_forward_rim_target_mean_m_s=(.3 * (learned[:, 4] - learned[:, 5])).mean().item(),
                learned_unclipped_forward_rim_target_correction_mean_m_s=delta.mean().item(),
                correction_min_m_s=delta.min().item(), correction_max_m_s=delta.max().item(),
                actual_old_body_velocity_mean_m_s=float(trace['velocity'][:, lanes][mask][:, 0].mean()),
            )
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
out.write_text(json.dumps(dict(schema='flat-wheel-from1280-fixed-input-learning-probe-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks,
    parent_initial_state_sha256=sha(parent_path), final_checkpoint_sha256=sha(final_path),
    native_state_proof_sha256=sha(proof_path), reader_sha256=sha(Path(__file__)), rows=rows,
    new_optimizer_updates=0, new_native_steps=0, behavior_qualified=False,
    conversion='Policy wheel actions *10rad/s, wheel radius .06m, forward rim mean .3*(aL-aR); unclipped requested targets only.',
    scope='Same saved original1280 observations; target corrections do not establish achieved velocity, closed-loop improvement, or a global learning limit.'), indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(passed=True, checks=len(checks), rows=rows), indent=2))
