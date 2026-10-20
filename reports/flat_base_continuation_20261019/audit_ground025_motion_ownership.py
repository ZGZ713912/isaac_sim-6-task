"""Separate motion coverage, output capacity, and actual learning ownership."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.static_feedback import StaticFeedbackActor
from wheeled_tasks.chassis.flat_base_continuation import digest, encode

torch.set_num_threads(1)
candidate = HERE / 'static_grip_candidate002'
target = candidate / 'motion_ownership001.json'
if target.exists():
    raise SystemExit('Preserve the completed motion ownership audit')
parent_path = ROOT / 'runs/flat_base_20261019/pilot028_wheel_rate1328/model_001328.pt'
child_path = ROOT / 'runs/flat_base_20261019/pilot029_static_grip1338/model_001338.pt'
trace_path = ROOT / 'runs/flat_base_20261019/ground025_parent1328_default_envelope001/traces.npz'
plan_path = candidate / 'plan.json'
plan = json.loads(plan_path.read_text())
load = lambda path: torch.load(path, map_location='cpu', weights_only=True)
parent_state, child_state = load(parent_path), load(child_path)
source = ROOT / plan['actor_bundle'] / 'source.onnx'
parent, child = [mean_from_checkpoint(state, source) for state in (parent_state, child_state)]
unchanged = {name: torch.equal(value, child_state['actor_state_dict'][name])
             for name, value in parent_state['actor_state_dict'].items()}
assert all(value for name, value in unchanged.items() if not name.startswith('mlp.static_feedback.'))
assert child_state['infos']['actor_trainability']['trainable_parameters'] == 1126
assert all(name.startswith('mlp.static_feedback.')
           for name in child_state['infos']['actor_trainability']['trainable_names'])
coverage = []
for row in plan['train_groups']:
    skill = row['skill']
    if skill['mode'] != 1:
        continue
    settings = skill['height_sampling']
    assert settings == dict(range_m=[.23, .42], nominal_fraction=.5, endpoint_fraction=.5)
    coverage.append(dict(name=row['name'], fraction=row['fraction'], command=skill['command'],
                         height_sampling=settings))

with np.load(trace_path) as trace:
    active = trace['active'].astype(bool)
    obs = trace['observations'][active]
    selected = torch.from_numpy(obs[np.linspace(0, len(obs) - 1, min(4096, len(obs)), dtype=int)].copy())
    moving = child.moving_mix(selected).squeeze(-1) == 1
    assert moving.any()
    with torch.inference_mode():
        same_moving_function = torch.equal(parent(selected[moving]), child(selected[moving]))
    assert same_moving_function
    early_rows = []
    for lane, name in enumerate(trace['groups']):
        ended = np.flatnonzero(active[:, lane] & trace['done'][:, lane].astype(bool))
        if not len(ended):
            continue
        tick = int(ended[0])
        if float(trace['episode_ticks'][tick, lane]) / 50 > 1.5:
            continue
        value = torch.from_numpy(trace['observations'][tick, lane].copy()).unsqueeze(0)
        with torch.inference_mode():
            base = child.command.base(value)
            command = child.command(value) - base
            mix = child.moving_mix(value)
            feedback = child.feedback_actions(value)
            wheel = child.authority_actions(value)
            mean = child(value)
            assert torch.allclose(base + command + feedback + wheel, mean, atol=1e-6, rtol=1e-6)
            assert torch.allclose(mean[0], torch.from_numpy(trace['actions'][tick, lane]), atol=2e-6, rtol=2e-6)
        early_rows.append(dict(case=str(name), lane=lane, terminal_seconds=float(trace['episode_ticks'][tick, lane]) / 50,
            visible_command=value[0, :4].tolist(), moving_mix=float(mix),
            fast_mix=float(child.fast_mix(value)), authority_gate=float(child.authority_gate(value)),
            raw_wheel_actions=dict(original=base[0, 4:].tolist(), command_residual=command[0, 4:].tolist(),
                motion_feedback=feedback[0, 4:].tolist(), wheel_authority=wheel[0, 4:].tolist(),
                total=mean[0, 4:].tolist()),
            static_head_gradient_multiplier=float(1 - mix)))
    assert early_rows and all(row['static_head_gradient_multiplier'] == 0 for row in early_rows)

target.write_bytes(encode(dict(schema='flat-ground025-motion-learning-ownership-audit-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), reader_sha256=digest(Path(__file__)),
    parent_checkpoint_sha256=digest(parent_path), child_checkpoint_sha256=digest(child_path),
    trace_sha256=digest(trace_path), plan_sha256=digest(plan_path),
    planned_motion_coverage=coverage,
    coverage_scope='Declared sampler distribution; not a measured per-update height histogram.',
    actual_trainable_branch='static_feedback', actual_trainable_parameters=1126,
    all_other_actor_tensors_equal=True, same_input_full_moving_function_equal=same_moving_function,
    actual_moving_input_probe_count=int(moving.sum()), early_motion_terminals=early_rows,
    conclusion='The existing plan includes both motion-height endpoints. Current static-only ownership has zero direct gradient on these failed full-moving inputs; it cannot directly repair the motion feedback or wheel head.',
    limitations='This output decomposition does not establish the physical cause of instability. Closed-loop startup and stopping can change through the adapted static head.',
    new_optimizer_updates=0, qualified=False)))
print(json.dumps(dict(motion_groups=len(coverage), moving_probe_count=int(moving.sum()),
    early_motion_terminals=len(early_rows), same_input_full_moving_function_equal=same_moving_function)))
