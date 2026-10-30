#!/usr/bin/env python3
"""Frozen-trajectory accounting; component scores do not predict a PPO update."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib, json
import numpy as np

parser = argparse.ArgumentParser(description="Reconstruct frozen native reward components and exploratory noise-score intervals; zero PPO")
parser.add_argument("--run-dir", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--event-dir", required=True, type=Path, help="The checkpoint's existing RSL-RL run")
args = parser.parse_args()
run, output = args.run_dir, args.output
assert not output.exists()
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
d = read(run / 'diagnosis.json')
assert d['status'] == 'completed' and d['ppo_updates'] == 0 and all(d['unchanged'].values())
assert d['native_ppo_timing']['lam'] == 1.
assert read(args.event_dir/'completion.json')['checkpoint_sha256'] == d['checkpoint_sha256']
assert sha(run/'rollouts.npz') == d['arrays_sha256']
gamma = d['native_ppo_timing']['gamma']
steps = d['native_ppo_timing']['num_steps_per_env']
a = np.load(run / 'rollouts.npz', allow_pickle=False)
learning, bootstrap, done = [a[k].astype(bool) for k in ('learning', 'bootstrap', 'done')]
mask = a['post_warmup'].astype(bool)
z = a['gaussian_noise'].astype(float)
raw = a['residuals'].astype(float)
tail = np.load(run / 'return_tails.npz', allow_pickle=False)['critic_value'].astype(float)
assert raw.shape[1] == 512 and len(raw) == len(tail) * steps

def discounted(reward, tails=None):
    result = np.zeros_like(reward, dtype=float)
    for block, start in enumerate(range(0, len(reward), steps)):
        future = np.zeros(512) if tails is None else tails[block].copy()
        for t in reversed(range(start, start + steps)):
            cont = learning[t] & bootstrap[t] & ~done[t]
            if t + 1 < start + steps:
                cont &= learning[t + 1]
            future = np.where(learning[t], reward[t] + np.where(cont, gamma * future, 0.), 0.)
            result[t] = future
    return result

def category(key):
    name = key.removeprefix('reward_')
    if name in ('dense_height_broad', 'dense_height_precision'): return 'height_tracking'
    if name.startswith(('dense_velocity_', 'dense_spin_velocity_', 'dense_lateral_')): return 'planar_tracking'
    if name.startswith('dense_yaw_'): return 'yaw_tracking'
    if name.startswith('dense_stationary_') or name == 'stand_still_lin_vel': return 'stationary_tracking'
    if name in ('leg_action_smoothness', 'wheel_action_smoothness', 'action_rate'): return name
    return 'other_density'

density = {}
for key in a.files:
    if key.startswith('reward_'):
        name = category(key)
        density.setdefault(name, np.zeros(raw.shape))
        density[name] += a[key].astype(float)
components = {key: discounted(value*.005) for key,value in density.items()}
components['failure_event'] = discounted(a['failure_event'].astype(float))
components['success_event'] = discounted(a['success_event'].astype(float))
components['time_limit_bootstrap_reward'] = discounted(a['storage_reward'].astype(float)-a['raw_reward'].astype(float))
components['rollout_tail_value'] = discounted(np.zeros_like(raw), tail)
components['negative_pre_value'] = np.where(learning, -a['stored_value'].astype(float), 0.)
reconstruction = sum(components.values())
maxerror = float(np.abs(reconstruction[learning]-raw[learning]).max())
assert maxerror < .001, maxerror

# Center each component with the same original whole-rollout population.
# Their sum reconstructs normalized residuals before any PPO clipping/epochs.
normalized = {k:np.zeros_like(v) for k,v in components.items()}
normalized_native_error = 0.
normalized_reconstruction_error = 0.
normalized_roundoff_bound = 0.
for start in range(0,len(raw),steps):
    sl = slice(start,start+steps); m=learning[sl]
    sd = raw[sl][m].std(ddof=1)+1e-8
    for k,v in components.items():
        normalized[k][sl] = np.where(m,(v[sl]-v[sl][m].mean())/sd,0.)
    rebuilt = sum(v[sl] for v in normalized.values())
    independent=(reconstruction[sl][m]-reconstruction[sl][m].mean())/sd
    normalized_reconstruction_error=max(normalized_reconstruction_error,float(np.abs(rebuilt[m]-independent).max()))
    raw_norm=(raw[sl][m]-raw[sl][m].mean())/sd
    bound=2*float(np.abs(reconstruction[sl][m]-raw[sl][m]).max())/sd+float(np.abs(raw_norm-a['advantages'][sl][m]).max())
    assert float(np.abs(rebuilt[m]-a['advantages'][sl][m]).max())<=bound+1e-10
    normalized_roundoff_bound=max(normalized_roundoff_bound,bound)
    normalized_native_error=max(normalized_native_error,float(np.abs(rebuilt[m]-a['advantages'][sl][m]).max()))
assert normalized_reconstruction_error < 1e-10,normalized_reconstruction_error

commands=a['commands']; stationary=(np.abs(commands[:,:,:2]) < .001).all(-1)
selections=dict(all_post_warmup=mask,
    low_height_stationary=mask & stationary & (commands[:,:,2] < .27),
    high_height_stationary=mask & stationary & (commands[:,:,2] > .38),
    forward=mask & (commands[:,:,0] > .4) & (np.abs(commands[:,:,1]) < .001),
    backward=mask & (commands[:,:,0] < -.4) & (np.abs(commands[:,:,1]) < .001))
populations={}
for population,m in selections.items():
    count=m.sum(0);ids=np.flatnonzero(count);n=int(count.sum());assert n>0
    # Whole environment trajectories are resampled, retaining temporal dependence.
    # A single saved RNG/checkpoint is not independent training-seed evidence.
    rng=np.random.default_rng(290935)
    indices=rng.integers(0,len(ids),(1000,len(ids)))
    denominator=count[ids][indices].sum(1)
    rows={}
    for k,v in normalized.items():
        entry={}
        for score_name,score in [('mean_score_z',z),('std_score_z2_minus1',z*z-1.)]:
            summed=(np.where(m[:,:,None],v[:,:,None]*score,0.)).sum(0)
            mean=summed.sum(0)/n
            boot=summed[ids][indices].sum(1)/denominator[:,None]
            bounds=np.quantile(boot,[.025,.975],axis=0)
            entry[score_name]=dict(mean=mean.tolist(),env_cluster_95pct=[bounds[0].tolist(),bounds[1].tolist()])
        rows[k]=entry
    total=sum(normalized.values())
    rows['total']={}
    for score_name,score in [('mean_score_z',z),('std_score_z2_minus1',z*z-1.)]:
        summed=(np.where(m[:,:,None],total[:,:,None]*score,0.)).sum(0)
        boot=summed[ids][indices].sum(1)/denominator[:,None]
        bounds=np.quantile(boot,[.025,.975],axis=0)
        rows['total'][score_name]=dict(mean=(summed.sum(0)/n).tolist(),env_cluster_95pct=[bounds[0].tolist(),bounds[1].tolist()])
    # Algebra only: retain recorded states, critic and masks; do not apply an update.
    candidate=raw-.9*components['leg_action_smoothness']
    populations[population]=dict(samples=n,environment_clusters=len(ids),
        command_mean=commands[m].mean(0).tolist(),actual_height_mean_m=float(a['height'][m].mean()),
        residual_rms=float(np.sqrt(np.mean(raw[m]**2))),
        offline_leg_smoothness_multiplier_point1_residual_rms=float(np.sqrt(np.mean(candidate[m]**2))),
        density_per_second={k:float(v[m].mean()) for k,v in density.items()},
        normalized_component_noise_scores=rows)
report=dict(schema='v6-local-component-credit-audit-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
    diagnosis=str(run/'diagnosis.json'),diagnosis_sha256=sha(run/'diagnosis.json'),arrays_sha256=sha(run/'rollouts.npz'),
    analysis_script_sha256=sha(Path(__file__)),checkpoint_sha256=d['checkpoint_sha256'],
    learning_clock=d['curriculum_actor_update'],ppo_updates=0,models_and_optimizer_unchanged=d['unchanged'],
    maximum_raw_component_sum_error=maxerror,maximum_normalized_component_sum_error=normalized_native_error,
    maximum_float64_centered_component_identity_error=normalized_reconstruction_error,
    native_float32_roundoff_propagation_bound=normalized_roundoff_bound,
    native_timing=d['native_ppo_timing'],populations=populations,
    interpretation='Linear component decomposition at recorded states/critic. Score estimates precede PPO clipping, gradients through the network, Adam and epochs. Cluster intervals resample whole environment trajectories within one checkpoint/RNG, are exploratory, and are not independent training-seed causal evidence. Component energies are not orthogonal explained variance. Offline multiplier point1 is algebra only and predicts neither retraining nor mechanics. No training-contract change or update occurred.')
# Append these measurements to the same checkpoint's original RSL-RL event namespace.
from torch.utils.tensorboard import SummaryWriter
with SummaryWriter(str(args.event_dir)) as writer:
    for population, data in populations.items():
        prefix = f"LocalDiagnostics/ComponentCredit/{population}"
        for key in ('samples', 'environment_clusters', 'residual_rms', 'actual_height_mean_m'):
            writer.add_scalar(prefix+'/'+key, data[key], report['learning_clock'])
        for component, scores in data['normalized_component_noise_scores'].items():
            for score, values in scores.items():
                for axis, mean in enumerate(values['mean']):
                    tag = f"{prefix}/{component}/{score}/axis{axis}"
                    writer.add_scalar(tag+'/mean', mean, report['learning_clock'])
                    writer.add_scalar(tag+'/cluster95_low', values['env_cluster_95pct'][0][axis], report['learning_clock'])
                    writer.add_scalar(tag+'/cluster95_high', values['env_cluster_95pct'][1][axis], report['learning_clock'])
report['original_event_directory'] = str(args.event_dir.resolve())
output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for name,p in populations.items():
    s=p['normalized_component_noise_scores']
    print(name,'samples',p['samples'],'clusters',p['environment_clusters'],'rms',p['residual_rms'])
    for k in ('height_tracking','leg_action_smoothness','failure_event','negative_pre_value','total'):
        print(k,'mean',np.round(s[k]['mean_score_z']['mean'],6),'std',np.round(s[k]['std_score_z2_minus1']['mean'],6))
print('output',output,'reconstruction_errors',maxerror,normalized_native_error)
