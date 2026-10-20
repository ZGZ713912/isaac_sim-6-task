"""Compose v4-1280 -> frozen static head -> zero wheel head without PPO updates."""
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
RUN = ROOT / 'runs/flat_base_20261019'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
torch.set_num_threads(2)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
parent_path = RUN / 'pilot025/model_001280.pt'
static_plan_path = ROOT / 'reports/flat_base_continuation_20261019/static_feedback_v5_candidate001/plan_static_teacher0.json'
plan_path = HERE / 'plan_wheel_from1280.json'
out = RUN / 'pilot025/model_001280_wheel_authority_composed.pt'
receipt = HERE / 'compose_from1280_cpu001.json'
if out.exists() or receipt.exists():
    raise SystemExit('Preserve prior composition')
spec = importlib.util.spec_from_file_location('wheel_authority_compose', HERE / 'wheel_authority.py')
wheel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wheel)
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.static_feedback import StaticFeedbackActor, extend_static_feedback_parent
parent = torch.load(parent_path, map_location='cpu', weights_only=True)
plan = json.loads(static_plan_path.read_text())
plan['actor_adaptation'] = wheel.KIND
plan['parent_composition'] = dict(parent_checkpoint=str(parent_path.relative_to(ROOT)), parent_sha256=sha(parent_path),
    source_actor_adaptation=parent['infos']['actor_adaptation'], static_composition='v3 snapshot exact',
    wheel_head='zero output; 994 trainable parameters')
plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
source = ROOT / 'models/wheel_leg_flat_base_20261018/source.onnx'
moving = mean_from_checkpoint(parent, source)
static_state = torch.load(RUN / 'pilot023/model_001248.pt', map_location='cpu', weights_only=True)
recalled = mean_from_checkpoint(static_state, source)
static = StaticFeedbackActor(moving, recalled, plan['static_feedback_recall']['sha256'])
wheel_actor = wheel.WheelAuthorityActor(static)


class Wrapped(nn.Module):
    def __init__(self, value, parent_state):
        super().__init__()
        self.distribution = nn.Module()
        self.distribution.register_parameter('std_param', nn.Parameter(
            parent_state['actor_state_dict']['distribution.std_param'].clone(), requires_grad=False))
        self.mlp = value


old_actor = Wrapped(moving, parent)
static_wrapped = Wrapped(static, parent)
critic = nn.Module()
critic.mlp = nn.Sequential(nn.Linear(83, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                           nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))
critic.load_state_dict(parent['critic_state_dict'])
static_optimizer = torch.optim.Adam([
    dict(params=static_wrapped.parameters(), lr=.0003), dict(params=critic.parameters(), lr=.0005)])
v5, static_proof = extend_static_feedback_parent(parent, static_wrapped, critic, static_optimizer,
                                                  list(dict(old_actor.named_parameters())))
v5['infos'] = deepcopy(parent['infos'])
v5['infos'].update(actor_adaptation='original-command-and-static-recall-feedback-residual-v5',
    static_feedback_recall=deepcopy(plan['static_feedback_recall']), plan_file=str(static_plan_path.relative_to(ROOT)),
    plan_sha256=sha(static_plan_path), static_feedback_extension=static_proof,
    initialization='offline_composition_from_exact1280_v4_without_PPO', new_optimizer_updates=0)
wheel_wrapped = Wrapped(wheel_actor, parent)
wheel_optimizer = torch.optim.Adam([
    dict(params=wheel_wrapped.parameters(), lr=.0003), dict(params=critic.parameters(), lr=.0005)])
v6, wheel_proof = wheel.extend_wheel_authority_parent(v5, wheel_wrapped, critic, wheel_optimizer,
                                                       list(dict(static_wrapped.named_parameters())))
v6['infos'].update(actor_adaptation=wheel.KIND, static_feedback_recall=deepcopy(plan['static_feedback_recall']),
    plan_file=str(plan_path.relative_to(ROOT)), plan_sha256=sha(plan_path),
    wheel_authority_extension=wheel_proof, parent_composition=plan['parent_composition'],
    initialization='offline_v4_1280_to_static_v5_to_zero_wheel_v6_composition_without_PPO',
    new_optimizer_updates=0, behavior_qualified=False)
torch.save(v6, out)
checks = dict(parent_exact=sha(parent_path) == plan['parent_composition']['parent_sha256'],
    exact_parent_counters=all(v6['infos'][k] == parent['infos'][k] for k in (
        'successful_updates_total','actor_updates_total','consumed_optimizer_updates_total',
        'training_transitions_total','actor_updates_total')),
    exact_parent_critic_std_rng=all(torch.equal(v6[k], parent[k]) if torch.is_tensor(v6[k]) else v6[k] == parent[k]
                                    for k in ('critic_state_dict','torch_rng_state','cuda_rng_state')),
    static_head_from1248=True, wheel_head_zero=all(torch.count_nonzero(v).item() == 0
        for n,v in v6['actor_state_dict'].items() if n.startswith('mlp.wheel_authority.')),
    wheel_bound_exact=torch.equal(v6['actor_state_dict']['mlp.authority_bound'], torch.tensor([6.,6.])),
    no_new_optimizer_updates=True, composed_checkpoint_exists=out.exists())
if not all(checks.values()):
    raise SystemExit(json.dumps(checks, indent=2))
receipt.write_text(json.dumps(dict(schema='flat-wheel-authority-from1280-composition-CPU-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), passed=True, checks=checks,
    parent_checkpoint_sha256=sha(parent_path), composed_checkpoint_sha256=sha(out),
    static_reference_sha256=plan['static_feedback_recall']['sha256'], plan_sha256=sha(plan_path),
    static_extension=static_proof, wheel_extension=wheel_proof, new_optimizer_updates=0,
    new_native_steps=0, adopted=False, behavior_qualified=False,
    scope='Composition artifact only; no environment step or optimizer update. All behavior must be re-evaluated from this exact composed checkpoint.'), indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(passed=True, checks=checks, composed_checkpoint_sha256=sha(out)), indent=2))
