"""Execute the real training parser/factory and replay complete recorded banks."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from train_robust_getup import prepare
from wheeled_algo.function_root_checkpoint import validate_training_settings
from wheeled_algo.robust_getup import RobustGetupActor

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
checkpoint = B / 'function_root_entry001/model_initial.pt'
expected, info = load_mean(checkpoint)
tree = ast.parse(Path('scripts/train_robust_getup.py').read_text())
main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
parser_end = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'args' for t in node.targets))
parser_code = compile(ast.Module(body=deepcopy(main.body[:parser_end + 1]), type_ignores=[]),
                      'actual-training-parser', 'exec')
command = json.loads((R / 'plan007_entry_support0_001.json').read_text())['training_command'][2:]
command[command.index('--resume') + 1] = str(checkpoint)
command[command.index('--output') + 1] = str(R / 'training_function_parity002')
command[command.index('--actor-lr') + 1] = '.00001'
sys.argv = ['actual-training-parser', *command]
namespace = dict(argparse=argparse, Path=Path, ROOT=Path.cwd())
exec(parser_code, namespace)
args = namespace['args']
args.device = 'cpu'
args.output.mkdir(parents=True, exist_ok=False)
_, task = prepare(args)
root_state = torch.load(info['training_function_root']['checkpoint'], map_location='cpu', weights_only=False)
validate_training_settings(task, root_state['infos']['settings'])
block = None
for node in ast.walk(main):
    if isinstance(node, ast.Try):
        for i, item in enumerate(node.body):
            if (isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Tuple)
                    and [getattr(t, 'id', None) for t in item.targets[0].elts] == ['actor_type', 'extra']):
                stop = next(j for j in range(i + 1, len(node.body))
                            if isinstance(node.body[j], ast.Assign)
                            and getattr(node.body[j].targets[0], 'id', None) == 'initialization')
                block = node.body[i:stop]
assert block
factory_namespace = dict(args=args, task=task, RobustGetupActor=RobustGetupActor,
                         load_mean=lambda _: deepcopy(expected.parent), torch=torch)
exec(compile(ast.Module(body=deepcopy(block), type_ignores=[]), 'actual-training-factory', 'exec'), factory_namespace)
actual = factory_namespace['mean']
actual.load_state_dict(expected.state_dict(), strict=True)
checks = dict(class_exact=type(actual) is type(expected), settings_inherited=all(task[k] == info['settings'][k]
    for k in ('gap_entry_encoder_version', 'motion_leg_damping_version', 'motion_leg_damping_alpha')),
    weights_and_buffers_exact=all(torch.equal(v, actual.state_dict()[k]) for k, v in expected.state_dict().items()))
rows = []
for name in ('diagnosis012_final_arithmetic_fixed/evaluation', 'diagnosis012_final_arithmetic_nominal/robust',
             'diagnosis012_final_arithmetic_rmcs/evaluation', 'diagnosis024_entry_phase_fit_fixed/evaluation'):
    path = B / name / 'trajectory.npz'
    observations = np.load(path)['observations'].reshape(-1, 35)
    different = 0
    peak = 0.
    with torch.inference_mode():
        for values in np.array_split(observations, (len(observations) + 511) // 512):
            batch = torch.from_numpy(values.copy())
            a, b = actual(batch), expected(batch)
            different += int((a != b).any(-1).sum())
            peak = max(peak, float((a - b).abs().max()))
    rows.append(dict(trace=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     frames=len(observations), different_frames=different, action_error_peak=peak))
checks['all_recorded_frames_exact'] = all(x['different_frames'] == 0 for x in rows)
out = dict(schema='actual-training-decoder-function-parity-v2', checks=checks,
    checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    training_source_sha256=hashlib.sha256(Path('scripts/train_robust_getup.py').read_bytes()).hexdigest(),
    actor_class=type(actual).__name__, rows=rows,
    task_differences={k: [root_state['infos']['settings'].get(k), task.get(k)]
        for k in set(root_state['infos']['settings']) | set(task)
        if root_state['infos']['settings'].get(k) != task.get(k)},
    scope='CPU execution of AST-extracted real parser and actor factory; complete recorded observation banks. Native rollout and optimizer restoration are separate checks.',
    new_PPO=0, actual_experiment_consumed=2603, qualification=None, hardware_started=False,
    created_unix=time.time())
(R / 'training_function_parity002.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2), flush=True)
assert all(checks.values()), checks
