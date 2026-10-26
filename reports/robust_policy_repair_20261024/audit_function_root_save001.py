"""Exercise the actual nested save hook with a zero-update CPU runner stub."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import json
import time

sys.path[:0] = ['src', 'scripts']
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest, PARENT

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
source = Path('runs/robust_policy_repair_20261024/function_root_entry001/model_initial.pt')
state = torch.load(source, map_location='cpu', weights_only=False)
tree = ast.parse(Path('scripts/train_robust_getup.py').read_text())
hook = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            and node.name == 'save' and any(isinstance(n, ast.Assign) and
            any(isinstance(t, ast.Name) and t.id == 'info' for t in n.targets) for n in node.body))
directory = R / 'function_root_save_smoke001'
directory.mkdir(exist_ok=False)
args = SimpleNamespace(resume=source, bank=Path('reports/robust_policy_repair_20261024/pose_bank_rmcs003.json'),
    initialize=Path('runs/unified_getup_20261021/pilot002/model_final.pt'), device='cpu',
    actor_variant='robust', quiet_only=False, quiet_velocity_only=False, focus_recovery_quiet=False,
    arc_only=False, recovery_only=True, entry_recovery_only=True)
generator = SimpleNamespace(get_state=lambda: state['infos']['environment_rng'].clone())
env = SimpleNamespace(domain=SimpleNamespace(rng=generator), generator=generator)
report = dict(completed_updates=1501, new_optimizer_updates=0, task_contract_sha256='CPU-save-hook-smoke')
hashes = deepcopy(state['infos']['source_sha256'])
actor = SimpleNamespace(state_dict=lambda: state['actor_state_dict'])
runner = SimpleNamespace(alg=SimpleNamespace(actor=actor))


def original_save(path, infos):
    result = deepcopy(state)
    result['infos'] = infos
    torch.save(result, path)


namespace = dict(args=args, report=report, state=state, task=state['infos']['settings'], env=env,
    torch=torch, Path=Path, deepcopy=deepcopy, PARENT=PARENT, digest=digest,
    initialization=state, hashes=hashes, runner=runner, original_save=original_save)
exec(compile(ast.Module(body=[deepcopy(hook)], type_ignores=[]), 'actual-native-save-hook', 'exec'), namespace)
target = directory / 'model_initial.pt'
namespace['save'](target)
a, ai = load_mean(source)
b, bi = load_mean(target)
checks = dict(schema_v3=bi['schema'] == 'robust-getup-checkpoint-v3',
    runtime_state_exact=all(torch.equal(value, b.state_dict()[key]) for key, value in a.state_dict().items()),
    zero_updates=bi['function_root_learning']['new_optimizer_updates'] == 0,
    learning_clock_exact=bi['completed_updates'] == ai['completed_updates'],
    unchanged_function_root=bi['training_function_root'] == ai['training_function_root'],
    atomic_pending_removed=not target.with_suffix('.pt.pending').exists())
out = dict(schema='actual-save-hook-function-root-smoke-v1', checks=checks,
    checkpoint=str(target), sha256=digest(target), source_sha256=digest(source),
    scope='Actual trainer save-hook AST with a zero-update CPU runner stub; no native rollout or PPO',
    new_PPO=0, actual_experiment_consumed=2603, qualification=None,
    hardware_started=False, created_unix=time.time())
(R / 'function_root_save_smoke001.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
assert all(checks.values())
