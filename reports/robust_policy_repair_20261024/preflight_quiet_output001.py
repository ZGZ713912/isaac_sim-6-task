"""Actual parser, decoder, training scope and saved-state tamper controls."""
from copy import deepcopy
from pathlib import Path
import argparse
import ast
import json
import sys
import tempfile
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
import train_robust_getup as train
from export_robust_getup import load_mean
from wheeled_algo.gap_recovery_exploration import GapRecoveryNoiseMean
from wheeled_algo.periodic_entry_actor import exact_state_equal
from wheeled_algo.quiet_output_checkpoint import LEARNED_KEYS, validate_training_settings

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
checkpoint = B / 'quiet_output_root001/model_initial.pt'
source = B / 'quiet_stance050003/model_probe.pt'
new, info = load_mean(checkpoint)
old, prior_info = load_mean(source)
a, b = [torch.load(p, map_location='cpu', weights_only=False) for p in (source, checkpoint)]
output = B / 'quiet_output_prepare001'
assert not output.exists()
command = ['--output', str(output), '--bank', str(R / 'pose_bank_rmcs003.json'),
    '--resume', str(checkpoint), '--updates', '8', '--num-envs', '256', '--rollout-steps', '512',
    '--seed', '241510', '--isolated-origins', '--gap-robustness', '--gap-severity', '.8',
    '--gap-actuator-timing', 'native', '--actor-lr', '.0001', '--friction-min', '.3',
    '--domain-scale', '.35', '--eval-seconds', '2', '--curriculum-updates', '512',
    '--curriculum-origin', '569', '--quiet-only', '--quiet-output-only',
    '--standing-support-training', '--early-standing-position', '--standing-drift-objective']
main = next(n for n in ast.parse(Path('scripts/train_robust_getup.py').read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == 'main')
stop = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
            and ast.unparse(n.value.func) == 'args.output.mkdir')
parser_code = compile(ast.Module(body=deepcopy(main.body[:stop]), type_ignores=[]),
                      'actual-quiet-parser-and-validation', 'exec')
ns = dict(vars(train))
sys.argv = ['actual-train-parser', *command]
exec(parser_code, ns)
args = ns['args']
output.mkdir()
_, task = train.prepare(args)
validate_training_settings(task, prior_info['settings'])
delta = {k: dict(before=prior_info['settings'].get(k), after=task.get(k))
         for k in prior_info['settings'].keys() | task.keys() if prior_info['settings'].get(k) != task.get(k)}
with np.load(B / 'diagnosis038_quiet_stance050_development/evaluation/trajectory.npz') as data:
    values = data['observations'].reshape(-1, 35)
    observations = torch.from_numpy(values[np.linspace(0, len(values)-1, 8192).astype(int)].copy())
settings = info['settings']
distributions = [GapRecoveryNoiseMean(m, settings['quiet_exploration_fraction'],
    settings['quiet_wheel_exploration_fraction'], settings['gap_recovery_leg_exploration_fraction']) for m in (old, new)]
for distribution in distributions:
    distribution.log_std.data.copy_(b['actor_state_dict']['mlp.log_std'])
with torch.inference_mode():
    before, after = [m(observations) for m in distributions]
# Execute the exact new native freeze branch on the authenticated mean.
scope = next(n for n in ast.walk(main) if isinstance(n, ast.If)
             and ast.unparse(n.test) == 'args.resume and quiet_function_root')
actor = torch.nn.Module()
actor.mlp = distributions[1]
class Runner: pass
runner = Runner(); runner.alg = Runner(); runner.alg.actor = actor
scope_ns = dict(args=args, quiet_function_root=True, runner=runner,
                quiet_heads=[new.quiet, new.quiet_low, new.quiet_high])
exec(compile(ast.Module(body=[deepcopy(scope)], type_ignores=[]), 'actual-native-quiet-freeze', 'exec'), scope_ns)
trainable = {k: p.numel() for k, p in actor.named_parameters() if p.requires_grad}
new(observations).square().mean().backward()
gradients = {k: float(p.grad.norm()) if p.grad is not None else 0.
             for k, p in actor.named_parameters() if p.requires_grad}
rejections = {}
with tempfile.TemporaryDirectory(prefix='quiet_output_decoder_', dir=R) as directory:
    path = Path(directory) / 'mutated.pt'
    for kind in ('module_sha', 'root_sha', 'scope', 'quiet_hidden', 'feedback', 'fallen', 'motion', 'std', 'clock'):
        bad = deepcopy(b)
        if kind == 'module_sha': bad['infos']['source_sha256']['src/wheeled_algo/quiet_output_checkpoint.py'] = '0'*64
        elif kind == 'root_sha': bad['infos']['policy_function_root']['sha256'] = '0'*64
        elif kind == 'scope': bad['infos']['policy_function_learning']['scope'] = 'all-quiet'
        elif kind == 'clock': bad['infos']['completed_updates'] += 1
        else:
            key = dict(quiet_hidden='mlp.mean.quiet.0.bias', feedback='mlp.mean.quiet_velocity.mid.weight',
                fallen='mlp.mean.recovery.4.bias', motion='mlp.mean.arc.4.bias', std='mlp.log_std')[kind]
            bad['actor_state_dict'][key].add_(.01)
        torch.save(bad, path)
        try: load_mean(path)
        except ValueError as error: rejections[kind] = str(error)
        else: raise AssertionError('Decoder accepted ' + kind)
historical = {}
for name, relative in dict(V2='motion_damping080002/model_probe.pt', V3='pilot013_projected_entry/model_final.pt',
    V4='quiet_stance050003/model_probe.pt', V5='pilot014_fallen_output/model_final.pt',
    V6='pilot016_followup_reward/model_final.pt').items():
    _, previous = load_mean(B / relative)
    historical[name] = previous['schema']
checks = dict(actual_parser_accepts_explicit_V7=ns['quiet_function_root'] is True,
    strict_V7_decode=info['schema'] == 'robust-getup-checkpoint-v7',
    all_nonmetadata_initial_state_exact=exact_state_equal({k:v for k,v in a.items() if k!='infos'}, {k:v for k,v in b.items() if k!='infos'}),
    actual_training_settings_allow_only_declared_fields=set(delta) <= {'version', 'motion_fraction', 'quiet_training', 'quiet_height_probabilities'},
    original_runtime_class=type(old) is type(new),
    original_mean_bitexact=torch.equal(before[:, 0], after[:, 0]),
    original_conditioned_std_bitexact=torch.equal(before[:, 1], after[:, 1]),
    actual_native_freeze_six_existing_outputs=set(trainable) == set(LEARNED_KEYS),
    only1170_actor_parameters=sum(trainable.values()) == 1170,
    all_six_outputs_have_finite_nonzero_gradient=all(np.isfinite(v) and v > 0 for v in gradients.values()),
    nine_saved_state_tamper_controls_rejected=len(rejections) == 9,
    historical_V2_to_V6_still_decode=len(historical) == 5)
out = dict(schema='quiet-output-preflight-v1', checks=checks, command=command,
    checkpoint=str(checkpoint), checkpoint_sha256=train.digest(checkpoint), source=str(source),
    source_sha256=train.digest(source), training_sha256=train.digest('scripts/train_robust_getup.py'),
    export_sha256=train.digest('scripts/export_robust_getup.py'),
    contract_sha256=train.digest('src/wheeled_algo/quiet_output_checkpoint.py'),
    settings_delta=delta, trainable=trainable, gradients=gradients, rejections=rejections,
    historical=historical, paired_inputs=len(observations), new_PPO=0,
    qualification=None, simulator_started=False, created_unix=time.time())
train.write_json(R / 'quiet_output_preflight001.json', out)
print(json.dumps(out, indent=2))
assert all(checks.values()), checks
