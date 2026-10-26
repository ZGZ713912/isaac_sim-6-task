"""Import the real zero-update collector and check its process callback."""
import argparse
import ast
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time
import types

sys.path[:0] = ['src', 'scripts']
import torch
from rsl_rl.runners import OnPolicyRunner
from train_robust_getup import digest, write_json
from wheeled_algo.motion_output_checkpoint import validate_training_settings

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
collector = R / 'collect_followup_reward_credit002.py'
tree = ast.parse(collector.read_text())
tree.body = tree.body[:-1]
sys.argv = [str(collector), '--diagnostic-rollout-only', '--output', '/tmp/followup-credit-import-only002']
namespace = {'__file__': str(collector.resolve()), '__name__': 'followup_collector_import_fixture'}
exec(compile(tree, str(collector), 'exec'), namespace)
checks = dict(import_without_training=not Path('/tmp/followup-credit-import-only002').exists())
values={'state': {1: {'step': torch.tensor(2., device='cuda'), 'exp_avg': torch.ones(3, device='cuda')}}, 'params': [1]}
portable=namespace['cpu_state'](values)
checks['CUDA_optimizer_readback_copied_to_CPU_exact'] = portable['state'][1]['step'].device.type=='cpu' and torch.equal(portable['state'][1]['exp_avg'],torch.ones(3))
learn_tree = ast.parse(textwrap.dedent(inspect.getsource(OnPolicyRunner.learn)))
calls = {}
for node in ast.walk(learn_tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('compute_returns', 'update'):
        calls.setdefault(node.func.attr, []).append(node.lineno)
checks['native_GAE_precedes_outer_optimizer'] = max(calls['compute_returns']) < min(calls['update'])
try:
    namespace['forbid_update'](None)
except RuntimeError:
    checks['optimizer_explicitly_forbidden'] = True

last = dict(native_reward=torch.tensor([1., 2., 3.]), bonus=torch.tensor([.025, 0., .00625]),
    native_followup_hold=torch.tensor([2., 0., 1.]), in_motion_fall=torch.tensor([True, False, True]))
fake_env = types.SimpleNamespace(training_reward_last=last)
storage = types.SimpleNamespace(step=0, rewards=torch.zeros(1, 3, 1))
alg = types.SimpleNamespace(device='cpu', gamma=.998,
    transition=types.SimpleNamespace(values=torch.tensor([[4.], [5.], [6.]])))
mask = types.SimpleNamespace(storage=storage, algorithm=alg)
extras = {'time_outs': torch.tensor([True, False, False])}

def process_fixture(self, obs, rewards, dones, info):
    actual = rewards.clone()
    actual += self.algorithm.gamma * torch.squeeze(
        self.algorithm.transition.values * info['time_outs'].unsqueeze(1), 1)
    self.storage.rewards[0, :, 0].copy_(actual)

namespace['env'] = fake_env
namespace['native_process'] = process_fixture
namespace['process'](mask, None, last['native_reward'] + last['bonus'], torch.zeros(3, dtype=torch.bool), extras)
expected = last['native_reward'].clone()
expected += alg.gamma * torch.squeeze(alg.transition.values * extras['time_outs'].unsqueeze(1), 1)
checks['original_reward_timeout_bootstrap_exact'] = torch.equal(mask.continuity_rows[0]['base_reward'], expected)
checks['timer_bonus_and_moving_flags_recorded_exact'] = all(torch.equal(mask.continuity_rows[0][k], last[v])
    for k, v in [('bonus', 'bonus'), ('native_hold', 'native_followup_hold'), ('moving_fall', 'in_motion_fall')])

source = Path('scripts/train_robust_getup.py').resolve()
module = types.ModuleType('followup_prepare_actual')
module.__file__ = str(source)
exec(compile(source.read_text(), str(source), 'exec'), module.__dict__)
native_parse = argparse.ArgumentParser.parse_args
captured = None

class Captured(Exception):
    pass

command = json.loads((R / 'motion_prepare_actual_preflight001.json').read_text())['command']
command[command.index('--output') + 1] = str(B / 'followup_prepare_CPU002')
command += ['--followup-continuity-weight', '5']

def capture(self, *args, **kwargs):
    global captured
    captured = native_parse(self, *args, **kwargs)
    raise Captured()

try:
    argparse.ArgumentParser.parse_args = capture
    sys.argv = ['train'] + command
    try:
        module.main()
    except Captured:
        pass
finally:
    argparse.ArgumentParser.parse_args = native_parse
cfg, task = module.prepare(captured)
source_info = torch.load(captured.resume, map_location='cpu', weights_only=False)['infos']
validate_training_settings(task, source_info['settings'])
old_task = json.loads((B / 'pilot015_motion_output/task_contract.json').read_text())['settings']
checks['all_native_task_settings_equal_round015'] = task == old_task
checks['reward_only_training_scope_requested'] = (captured.followup_continuity_weight == 5.
    and captured.motion_output_only and captured.motion_recovery_focus and not cfg['record_diagnostics'])
result = dict(schema='followup-reward-zero-collector-preflight-v1', checks=checks,
    collector_sha256=digest(collector), trainer_sha256=digest(source), command=command,
    native_runner=str(inspect.getsourcefile(OnPolicyRunner)), native_runner_sha256=digest(inspect.getsourcefile(OnPolicyRunner)),
    native_GAE_update_order=calls, new_PPO=0, simulator_started=False, qualification=None, created_unix=time.time())
write_json(R / 'followup_collector_preflight002.json', result)
print(json.dumps(result, indent=2))
assert all(checks.values()), checks
