"""Measure necessary same-input wheel-target authority on saved native states."""
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
torch.set_num_threads(2)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
checkpoint = RUN / 'pilot025/model_001280.pt'
mean = mean_from_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=True),
                            ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx')
out = HERE / 'high_reverse1280_same_input_bounds001.json'
if out.exists():
    raise SystemExit('Preserve earlier authority diagnostic')
rows = {}
traces = {}
for cohort in ('rmcs_default1280_envelope001', 'rmcs_maximum1280_envelope001'):
    path = RUN / cohort / 'traces.npz'
    traces[cohort] = sha(path)
    trace = np.load(path)
    times = np.arange(trace['observations'].shape[0]) * .02
    for name in ('backward_2', 'backward_3', 'forward_3'):
        lanes = trace['groups'] == name
        obs = trace['observations'][:, lanes]
        active = trace['active'][:, lanes].astype(bool)
        for window, lower, upper in (('whole', 0., 100.), ('five_to_nine_seconds', 5., 9.)):
            selected = active & (times[:, None] >= lower) & (times[:, None] < upper)
            x = torch.from_numpy(obs[selected])
            saved = torch.from_numpy(trace['actions'][:, lanes][selected])
            with torch.inference_mode():
                base = mean.command.base(x)
                command = mean.command.residual_actions(x)
                actions = mean(x)
                gate = mean.tracking_feedback_gate(x).squeeze(-1)
                rim = lambda a: .3 * (a[:, 4] - a[:, 5])
                center = rim(base + command)
                radius = .3 * mean.output_scale[4:6].sum() * gate
                minimum, maximum = center - radius, center + radius
                wanted = x[:, 0]
                actual_encoder = .3 * (x[:, 20] - x[:, 21])
                near_bound = (mean.feedback_actions(x)[:, 4:6].abs()
                              >= .95 * mean.output_scale[4:6] * gate[:, None].clamp_min(1e-9))
            rows[cohort + '/' + name + '/' + window] = dict(
                frames=len(x), all_active_action_CPU_max_abs_error=float((actions - saved).abs().max()),
                body_forward_mean_m_s=float(trace['velocity'][:, lanes][selected, 0].mean()),
                encoder_forward_mean_m_s=float(actual_encoder.mean()),
                wheel_target_forward_mean_m_s=float(rim(actions).mean()),
                base_wheel_target_forward_mean_m_s=float(rim(base).mean()),
                command_correction_forward_mean_m_s=float(rim(command).mean()),
                mean_gate=float(gate.mean()),
                minimum_available_target_mean_m_s=float(minimum.mean()),
                minimum_available_target_min_m_s=float(minimum.min()),
                maximum_available_target_mean_m_s=float(maximum.mean()),
                reverse_command_unable_to_request_nonpositive_target_fraction=float(
                    ((minimum > 0.) & (wanted < 0.)).float().mean()),
                exact_command_target_outside_same_input_bounds_fraction=float(
                    ((wanted < minimum) | (wanted > maximum)).float().mean()),
                at95percent_feedback_bound_fraction_per_wheel=near_bound.float().mean(0).tolist())
    trace.close()
checks = dict(
    CPU_reproduces_native_actions=all(r['all_active_action_CPU_max_abs_error'] < 5e-6 for r in rows.values()),
    high_reverse_steady_all_target_minima_positive=rows[
        'rmcs_maximum1280_envelope001/backward_3/five_to_nine_seconds']['minimum_available_target_min_m_s'] > 0.,
    steady_reverse_not_at_current_tanh_limit=all(v == 0. for v in rows[
        'rmcs_maximum1280_envelope001/backward_3/five_to_nine_seconds']['at95percent_feedback_bound_fraction_per_wheel']))
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
out.write_text(json.dumps(dict(schema='flat-high-reverse-same-input-wheel-authority-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks,
    checkpoint_sha256=sha(checkpoint), trace_sha256=traces, rows=rows,
    action_conversion='policy L/R wheel actions *10rad/s; forward mean rim speed .3*(aL-aR), radius .06m',
    scope='Necessary instantaneous target bounds at recorded states. Body speed is not a motor target; leg control and changed closed-loop states may escape this region. No global impossibility, tanh-saturation or unique-cause claim.',
    new_optimizer_updates=0, new_native_steps=0, evaluation_traces_used_for_optimization=False,
    behavior_qualified=False), indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(passed=True, checks=checks, high_reverse_steady=rows[
    'rmcs_maximum1280_envelope001/backward_3/five_to_nine_seconds']), indent=2))
