"""Exercise strict V6 decode, exact neutral outputs and historical compatibility."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from export_robust_getup import load_mean, digest
from train_robust_getup import write_json
from wheeled_algo.gap_recovery_exploration import GapRecoveryNoiseMean
from wheeled_algo.periodic_entry_actor import exact_state_equal

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
cp = B/'motion_output_root001/model_initial.pt'
source = B/'quiet_stance000003/model_probe.pt'
new, info = load_mean(cp)
old, _ = load_mean(source)
state = torch.load(cp, map_location='cpu', weights_only=False)
prior = torch.load(source, map_location='cpu', weights_only=False)
frames = np.load(B/'diagnosis025_motion_damping080_fixed/evaluation/trajectory.npz')['observations'].reshape(-1, 35)
obs = torch.from_numpy(frames[np.linspace(0, len(frames)-1, 4096).astype(int)].copy())
settings = info['settings']
distributions = [GapRecoveryNoiseMean(mean, settings['quiet_exploration_fraction'],
    settings['quiet_wheel_exploration_fraction'], settings['gap_recovery_leg_exploration_fraction']) for mean in (old, new)]
for distribution in distributions:
    distribution.log_std.data.copy_(state['actor_state_dict']['mlp.log_std'])
with torch.inference_mode():
    before, after = [distribution(obs) for distribution in distributions]
checks = dict(strict_V6_decode=True, original_runtime_class=type(old) is type(new),
    all_nonmetadata_saved_state_exact=exact_state_equal({k:v for k,v in state.items() if k!='infos'}, {k:v for k,v in prior.items() if k!='infos'}),
    mean_bit_exact=torch.equal(before[:, 0], after[:, 0]),
    conditioned_std_bit_exact=torch.equal(before[:, 1], after[:, 1]))
historical = {}
for schema, path in [('V2', 'motion_damping080002/model_probe.pt'),
                     ('V3', 'pilot013_projected_entry/model_final.pt'),
                     ('V4', 'quiet_stance050003/model_probe.pt'),
                     ('V5', 'pilot014_fallen_output/model_final.pt')]:
    _, previous_info = load_mean(B/path)
    historical[schema] = dict(checkpoint=str(B/path), sha256=digest(B/path), decoded_schema=previous_info['schema'])
checks['historical_V2_V3_V4_V5_decode'] = len(historical)==4
rejected = {}
with tempfile.TemporaryDirectory(prefix='motion_decoder_', dir=R) as directory:
    path = Path(directory)/'mutated.pt'
    for kind in ('module_sha', 'function_root_sha', 'scope', 'motion_weight', 'fallen_weight', 'wrong_schema'):
        bad = deepcopy(state)
        if kind=='module_sha':
            bad['infos']['source_sha256']['src/wheeled_algo/motion_output_checkpoint.py']='0'*64
        elif kind=='function_root_sha':
            bad['infos']['policy_function_root']['sha256']='0'*64
        elif kind=='scope':
            bad['infos']['policy_function_learning']['scope']='all-actor'
        elif kind=='motion_weight':
            bad['actor_state_dict']['mlp.mean.arc.4.bias'].add_(.01)
        elif kind=='fallen_weight':
            bad['actor_state_dict']['mlp.mean.recovery.4.bias'].add_(.01)
        else:
            bad['infos']['schema']='robust-getup-checkpoint-v5'
        torch.save(bad, path)
        try:
            load_mean(path)
        except ValueError as error:
            rejected[kind]=str(error)
        else:
            raise AssertionError('Actual decoder accepted '+kind)
checks['six_actual_decoder_tamper_controls_rejected'] = len(rejected)==6
out = dict(schema='motion-output-actual-decoder-preflight-v1', checks={k:bool(v) for k,v in checks.items()},
    checkpoint=str(cp), checkpoint_sha256=digest(cp), source_checkpoint=str(source), source_sha256=digest(source),
    paired_recorded_inputs=len(obs), historical=historical, rejections=rejected,
    new_PPO=0, simulator_started=False, qualification=None, created_unix=time.time())
write_json(R/'motion_decoder_preflight001.json', out)
print(json.dumps(out, indent=2))
assert all(checks.values()), checks
