"""Exercise the unadopted v5 loader, actual Adam extension and independent plan contract."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
torch.set_num_threads(2)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


command = module('wheeled_algo.command_residual',HERE/'src/wheeled_algo/command_residual.py')
# The staged source is intended for this exact production path after adoption.
command.__file__ = str(ROOT/'src/wheeled_algo/command_residual.py')
static = module('wheeled_algo.static_feedback',HERE/'src/wheeled_algo/static_feedback.py')
flat = module('wheeled_tasks.chassis._static_candidate',HERE/'src/wheeled_tasks/chassis/flat_base_continuation.py')
plan_path = HERE/'plan_static.json'
plan = flat.read_plan(ROOT,plan_path)
worker,seed = flat.write_bundle(ROOT,HERE/'config4096',4096,plan_file=plan_path)
from wheeled_tasks.chassis.preflight import design_preflight
control = json.loads((ROOT/plan['control_file']).read_text())
preflight = design_preflight(worker,control,ROOT,stage='speed',role='train',seed=seed,num_envs=4096)
source = ROOT/'models/wheel_leg_flat_base_20261018/source.onnx'
parent_path = ROOT/'runs/flat_base_20261019/pilot025/model_001280.pt'
parent = torch.load(parent_path,map_location='cpu',weights_only=True)
recall = plan['static_feedback_recall']
recalled = command.mean_from_checkpoint(torch.load(ROOT/recall['checkpoint'],map_location='cpu',weights_only=True),source)
moving = command.mean_from_checkpoint(parent,source)
mean = static.StaticFeedbackActor(moving,recalled,recall['sha256'])


class Wrapped(nn.Module):
    def __init__(self,value):
        super().__init__()
        self.distribution = nn.Module()
        self.distribution.register_parameter('std_param',nn.Parameter(
            parent['actor_state_dict']['distribution.std_param'].clone(),requires_grad=False))
        self.mlp = value


old_actor,actor = Wrapped(moving),Wrapped(mean)
critic = nn.Module()
critic.mlp = nn.Sequential(nn.Linear(83,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),
    nn.Linear(128,64),nn.ELU(),nn.Linear(64,1))
critic.load_state_dict(parent['critic_state_dict'])
optimizer = torch.optim.Adam([dict(params=actor.parameters(),lr=.0003),dict(params=critic.parameters(),lr=.0005)])
rng = torch.get_rng_state().clone()
transformed,extension = static.extend_static_feedback_parent(parent,actor,critic,optimizer,list(dict(old_actor.named_parameters())))
actor.load_state_dict(transformed['actor_state_dict'])
optimizer.load_state_dict(transformed['optimizer_state_dict'])
proof = static.configure_static_feedback_training(actor,True)
checks = dict(independent4096_contract_passed=preflight['status'] == 'static_passed',
    static_recall_in_worker_and_design=worker['static_feedback_recall'] == recall,
    extension_RNG_unchanged=torch.equal(rng,torch.get_rng_state()),
    trainable2252=proof['trainable_parameters'] == 2252,
    static_tensors_frozen=all(not p.requires_grad for p in actor.mlp.static_feedback.parameters()),
    old_actor_weights_exact=all(torch.equal(v,transformed['actor_state_dict'][n]) for n,v in parent['actor_state_dict'].items()),
    all_other_parent_checkpoint_fields_exact=all(
        (torch.equal(v,transformed[n]) if torch.is_tensor(v) else
         all(torch.equal(a,b) for a,b in zip(v,transformed[n],strict=True)) if n == 'cuda_rng_state' else
         v == transformed[n]) for n,v in parent.items()
        if n not in ('actor_state_dict','optimizer_state_dict','critic_state_dict')))
transformed['infos']['actor_adaptation'] = static.KIND
transformed['infos']['static_feedback_recall'] = deepcopy(recall)
loaded = command.mean_from_checkpoint(transformed,source)
probe = torch.zeros(64,35)
probe[:,3] = 1.525
probe[:,9] = -1.
probe[:,28] = 1.
probe[16:32,0] = .5
probe[32:48,0] = -.5
probe[48:,2] = 1.
with torch.inference_mode():
    checks['actual_composed_checkpoint_loader_exact'] = torch.equal(mean(probe),loaded(probe))
bad = deepcopy(transformed)
bad['infos']['static_feedback_recall']['sha256'] = '0'*64
try:
    command.mean_from_checkpoint(bad,source)
except ValueError:
    checks['wrong_static_checkpoint_rejected'] = True
else:
    checks['wrong_static_checkpoint_rejected'] = False
bad = deepcopy(transformed)
bad['actor_state_dict']['mlp.static_feedback.2.bias'][0] += .01
try:
    command.mean_from_checkpoint(bad,source)
except ValueError:
    checks['changed_frozen_static_head_rejected'] = True
else:
    checks['changed_frozen_static_head_rejected'] = False
static.configure_static_feedback_training(actor,False)
checks['critic_only_scope_freezes_entire_actor'] = not any(p.requires_grad for p in actor.parameters())
if not all(checks.values()):
    print(json.dumps(checks,indent=2))
    raise SystemExit('Candidate integration proof failed; preserve all staged files')
out = HERE/'integration_cpu001.json'
if out.exists():
    raise SystemExit('Preserve previous integration evidence')
result = dict(schema='flat-static-feedback-v5-candidate-integration-CPU-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(),passed=True,checks=checks,
    candidate_source_sha256={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*.py') if 'parent_source' not in p.parts},
    parent_checkpoint_sha256=sha(parent_path),candidate_plan_sha256=sha(plan_path),
    static_preflight=preflight,extension_contract=extension,
    adopted=False,new_native_steps=0,new_optimizer_updates=0,behavior_qualified=False)
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=True,checks=checks,adopted=False),indent=2))
