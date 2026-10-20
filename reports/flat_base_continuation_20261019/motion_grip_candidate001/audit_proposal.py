"""CPU-only preparation; source adoption and native training remain separate."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_tasks.chassis.flat_base_continuation import digest, encode

torch.set_num_threads(1)
target = HERE / 'cpu_preparation001.json'
if target.exists():
    raise SystemExit('Preserve the CPU-only proposal preparation')
module_path = HERE / 'proposed_motion_grip.py'
spec = importlib.util.spec_from_file_location('proposed_motion_grip', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
parent_path = ROOT / 'runs/flat_base_20261019/pilot029_static_grip1338/model_001338.pt'
expected_parent = '39f7f2d364d82905741478fd254c7fdde3d9c101a613cfba1ba79d14a483fd73'
assert digest(parent_path) == expected_parent
state = torch.load(parent_path, map_location='cpu', weights_only=True)
plan_path = ROOT / state['infos']['plan_file']
plan = json.loads(plan_path.read_text())
assert digest(plan_path) == state['infos']['plan_sha256']
parent = mean_from_checkpoint(state, ROOT / plan['actor_bundle'] / 'source.onnx')
child = module.MotionGripActor(parent, expected_parent)
assert list(parent.state_dict()) == list(child.state_dict())
assert list(dict(parent.named_parameters())) == list(dict(child.named_parameters()))
assert all(torch.equal(value, child.state_dict()[name]) for name, value in parent.state_dict().items())
trace_path = ROOT / 'runs/flat_base_20261019/ground025_static1338_default_envelope001/traces.npz'
with np.load(trace_path) as trace:
    obs = trace['observations'][trace['active'].astype(bool)]
    selected = torch.from_numpy(obs[np.linspace(0, len(obs)-1, min(4096, len(obs)), dtype=int)].copy())
with torch.no_grad():
    identity = torch.equal(parent(selected), child(selected))
assert identity
target_output = child(selected).detach() + .01
(child(selected) - target_output).square().mean().backward()
gradients = {name: float(parameter.grad.norm())
             for name, parameter in child.named_parameters() if parameter.requires_grad}
assert len(gradients) == 12 and all(value > 0 for value in gradients.values())
assert all(parameter.grad is None for name, parameter in child.named_parameters()
           if not name.startswith(tuple(name + '.' for name in module.BRANCHES)))
draft = deepcopy(plan)
draft.update(actor_adaptation=module.KIND,
             motion_grip_parent=dict(checkpoint=str(parent_path.relative_to(ROOT)), sha256=expected_parent),
             additional_critic_warmup_updates=2,
             training_scope='low_grip_motion_feedback_and_wheel_adaptation_preserve_adapted_static')
draft_path = HERE / 'proposed_plan.json'
draft_path.write_bytes(encode(draft))
target.write_bytes(encode(dict(schema='flat-motion-grip-cpu-proposal-v1',
    status='cpu_prepared_not_adopted', recorded_at=datetime.now(timezone.utc).isoformat(),
    parent_checkpoint_sha256=expected_parent,
    proposal_file=str(module_path.relative_to(ROOT)), proposal_sha256=digest(module_path),
    test_report_sha256=digest(HERE / 'cpu_proposal002.xml'),
    changed_plan_fields=sorted(key for key in plan.keys() | draft.keys() if plan.get(key) != draft.get(key)),
    proposed_plan_sha256=digest(draft_path),
    actual_input_probe=dict(trace_sha256=digest(trace_path), frames=len(selected),
        entry_function_equal=identity, parameter_order_equal=True,
        trainable_parameters=sum(p.numel() for p in child.parameters() if p.requires_grad),
        gradient_norms=gradients,
        scope='Squared-output CPU gradient probe only; not PPO or evidence of behavioral improvement'),
    proposed_bounded_pilot=dict(critic_updates=2, actor_updates=8, final_successful_update=1348,
        population=4096, actor_learning_rate=.0001, policy_hz=50, physics_hz=1000),
    frozen='original and command branches, adapted static head, std and their existing Adam slots',
    source_adopted=False, native_training_started=False, new_optimizer_updates=0,
    full_training_admitted=False, qualified=False,
    prerequisites=['Seal all four in-flight1338 replay cohorts before changing source',
        'Integrate authenticated parent loading and explicit trainer/plan/export routes',
        'Pass complete CPU routes and native initial/first-critic/first-actor state verification',
        'Evaluate actual post-PPO behavior before any further update block'],
    reader_sha256=digest(Path(__file__)))))
print(json.dumps(dict(status='cpu_prepared_not_adopted',
    real_observations=len(selected), entry_function_equal=identity,
    moving_head_gradients=gradients)))
