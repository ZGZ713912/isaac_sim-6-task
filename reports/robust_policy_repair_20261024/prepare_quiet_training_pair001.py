"""Prepare, but do not admit or start, a paired24-parameter quiet pilot."""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest, prepare, write_json
from wheeled_algo.gap_recovery_exploration import GapRecoveryNoiseMean
from wheeled_algo.periodic_entry_actor import exact_state_equal
from wheeled_algo.robust_getup import RobustGetupActor

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
pre = json.loads((R / 'quiet_reward_preflight001.json').read_text())
records = []
for row in pre['checkpoints']:
    source = Path(row['checkpoint'])
    assert digest(source) == row['sha256']
    mean, info = load_mean(source)
    state = torch.load(source, map_location='cpu', weights_only=False)
    distribution = GapRecoveryNoiseMean(deepcopy(mean), info['settings']['quiet_exploration_fraction'],
        info['settings'].get('quiet_wheel_exploration_fraction'), info['settings']['gap_recovery_leg_exploration_fraction'])
    names = [(n,p) for n,p in distribution.named_parameters() if p.requires_grad]
    group = next(x for x in state['optimizer_state_dict']['param_groups'] if x['name'] == 'actor')
    assert len(group['params']) == len(names) == 28
    changes = []
    initialized = deepcopy(state)
    for pid,(name,param) in zip(group['params'], names, strict=True):
        if name.startswith('mean.quiet_velocity.'):
            assert param.shape == (4,2)
            old = initialized['optimizer_state_dict']['state'].pop(pid, None)
            changes.append(dict(name=name, optimizer_id=pid, prior_state_existed=old is not None,
                prior_step=float(old['step']) if old else None))
    assert len(changes) == 3
    expected = deepcopy(state['optimizer_state_dict'])
    for item in changes:
        expected['state'].pop(item['optimizer_id'], None)
    assert exact_state_equal(expected, initialized['optimizer_state_dict'])
    assert all(exact_state_equal(state[k],initialized[k]) for k in state if k not in ('infos','optimizer_state_dict'))
    initialized['infos']['quiet_feedback_optimizer_initialization'] = dict(
        schema='explicit-quiet-feedback-Adam-reset-v1', source=str(source.resolve()), source_sha256=digest(source),
        reset=changes, reason='Fresh shared optimization after the documented zero-PPO diagonal sign edit; not a function or coordinate-equivalence claim.',
        other_Adam_critic_weights_RNG_clock='exact', new_PPO=0)
    target = B / ('quiet_training_' + row['name'] + '001') / 'model_initial.pt'
    target.parent.mkdir(parents=True, exist_ok=False)
    torch.save(initialized,target)
    restored,_ = load_mean(target)
    assert exact_state_equal(restored.state_dict(), mean.state_dict())
    records.append(dict(name=row['name'], checkpoint=str(target), sha256=digest(target), reset=changes))

# Execute the real parser, prepare function and actor factory, without Isaac or PPO.
tree = ast.parse(Path('scripts/train_robust_getup.py').read_text())
main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'main')
end = next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
           and any(isinstance(t,ast.Name) and t.id=='args' for t in n.targets))
parser_code = compile(ast.Module(body=deepcopy(main.body[:end+1]),type_ignores=[]),'actual-quiet-parser','exec')
command = ['--output',str(R/'quiet_training_factory_preflight001'),'--bank',str(R/'pose_bank_rmcs003.json'),
    '--resume',records[0]['checkpoint'],'--updates','8','--num-envs','256','--rollout-steps','512','--seed','241510',
    '--isolated-origins','--gap-robustness','--gap-severity','.8','--gap-actuator-timing','native',
    '--actor-lr','.001','--friction-min','.3','--domain-scale','.35','--eval-seconds','2',
    '--curriculum-updates','512','--curriculum-origin','569','--quiet-only','--quiet-velocity-only',
    '--standing-support-training','--early-standing-position','--standing-drift-objective']
sys.argv = ['actual-quiet-parser',*command]
ns = dict(argparse=argparse,Path=Path,ROOT=Path.cwd())
exec(parser_code,ns)
args=ns['args'];args.device='cpu';args.output.mkdir(parents=True,exist_ok=False)
_,task=prepare(args)
expected,info=load_mean(records[0]['checkpoint'])
block=None
for node in ast.walk(main):
    if isinstance(node,ast.Try):
        for i,item in enumerate(node.body):
            if (isinstance(item,ast.Assign) and isinstance(item.targets[0],ast.Tuple)
                and [getattr(t,'id',None) for t in item.targets[0].elts]==['actor_type','extra']):
                stop=next(j for j in range(i+1,len(node.body)) if isinstance(node.body[j],ast.Assign)
                          and getattr(node.body[j].targets[0],'id',None)=='initialization')
                block=node.body[i:stop]
assert block
factory=dict(args=args,task=task,RobustGetupActor=RobustGetupActor,
             load_mean=lambda _:deepcopy(expected.parent),torch=torch)
exec(compile(ast.Module(body=deepcopy(block),type_ignores=[]),'actual-quiet-factory','exec'),factory)
actual=factory['mean'];actual.load_state_dict(expected.state_dict(),strict=True)
distribution=GapRecoveryNoiseMean(actual,task['quiet_exploration_fraction'],task.get('quiet_wheel_exploration_fraction'),
                                task['gap_recovery_leg_exploration_fraction'])
actual.freeze_except_quiet_velocity();distribution.log_std.requires_grad_(False)
trainable=[(n,p.numel()) for n,p in distribution.named_parameters() if p.requires_grad]
assert len(trainable)==3 and sum(x[1] for x in trainable)==24
traces=[]
with torch.inference_mode():
    for name in ('diagnosis013_low_diagonal_fixed/evaluation','diagnosis015_development21_baseline/evaluation'):
        path=B/name/'trajectory.npz';obs=np.load(path)['observations'].reshape(-1,35)
        different=0
        for values in np.array_split(obs,(len(obs)+511)//512):
            x=torch.from_numpy(values.copy())
            different+=int((actual(x)!=expected(x)).any(-1).sum())
        assert different==0
        traces.append(dict(path=str(path),sha256=digest(path),frames=len(obs),different=different))
write_json(R/'quiet_training_preflight001.json',dict(schema='quiet-pair-training-preparation-v1',
    checkpoints=records,training_arguments=command,trainable=trainable,task=task,traces=traces,
    training_source_sha256=digest(Path('scripts/train_robust_getup.py')),
    checks=dict(actual_factory_and_decoder_exact=True,only24feedback_parameters_trainable=True,
        explicit_shared_three_slot_Adam_reset=True,all_other_state_exact=True),
    status='prepared_not_admitted',new_PPO=0,actual_experiment_consumed=2605,qualification=None,
    hardware_started=False,created_unix=time.time()))
print(json.dumps(dict(checkpoints=records,trainable=trainable,traces=traces),indent=2))
