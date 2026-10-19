"""Read archived low-backward action limits and log the sampled diagnostics."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT/'runs/flat_base_20261019'
sys.path.insert(0, str(ROOT/'src'))
from wheeled_algo.command_residual import mean_from_checkpoint

torch.set_num_threads(2)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
output = HERE/'feedback1248_backward_limits001.json'
if output.exists():
    raise SystemExit('Preserve the completed archived-action readout')
versions, checks, initial = {}, {}, {}
prefix = 'FlatActionAudit/flat-base-paired107-fixed-v1/physics_1000Hz/seed_191029/backward_05/post3s'
for count, trial in ((1120,'pilot021'),(1184,'pilot022'),(1248,'pilot023')):
    checkpoint = RUN/trial/f'model_{count:06d}.pt'
    path = RUN/f'deployment{count}_basic001/evaluation.json'
    trace = path.with_name('traces.npz')
    evaluation = read(path)
    case = evaluation['result']['cases']['backward_05']
    checks[str(count)+'_completed_same_checkpoint'] = evaluation['status'] == 'evaluated' and evaluation['checkpoint_sha256'] == sha(checkpoint)
    checks[str(count)+'_all_four_full_at_matching_clock'] = case['full_horizon_episodes'] == 4 and evaluation['physics_hz'] == 1000 and evaluation['policy_hz'] == 50 and evaluation['seed'] == 191029
    with np.load(trace) as z:
        groups = z['groups']; ids = np.flatnonzero(groups == 'backward_05')
        data = {k:z[k][:,ids].copy() for k in ('observations','actions','active','episode_ticks','velocity')}
    initial[count] = data['observations'][0].copy()
    selected = data['active'].astype(bool) & (data['episode_ticks']*.02 > 3.)
    obs = torch.from_numpy(data['observations'][selected].copy())
    recorded = torch.from_numpy(data['actions'][selected].copy())
    actor = mean_from_checkpoint(torch.load(checkpoint,map_location='cpu',weights_only=True),
        ROOT/'models/wheel_leg_flat_base_20261018/source.onnx').eval()
    with torch.inference_mode():
        predicted = actor(obs)
        command = actor.command(obs)
        correction = actor.feedback_actions(obs)
        gate = actor.tracking_feedback_gate(obs)
        raw = torch.tanh(actor.feedback((obs[:,:28]-actor.input_center)*actor.input_scale))
        mix = actor.fast_mix(obs)
    checks[str(count)+'_1400_matching_command_height_samples'] = (len(obs) == 1400 and bool((obs[:,0] == -.5).all())
        and bool((obs[:,2] == 0).all()) and bool(torch.allclose(obs[:,3],torch.full_like(obs[:,3],1.525),atol=1e-6,rtol=0)))
    checks[str(count)+'_CPU_matches_native_actions'] = bool(torch.allclose(predicted,recorded,atol=2e-5,rtol=2e-5))
    checks[str(count)+'_same_input_decomposition'] = bool(torch.allclose(predicted,command+correction,atol=2e-5,rtol=2e-5))
    checks[str(count)+'_low_branch_selected'] = bool((mix == 0).all())
    metrics = dict(sample_frames=len(obs),root_vx_signed_mean_m_s=float(data['velocity'][selected][:,0].mean()),
        encoder_vx_signed_mean_m_s=float((.3*(obs[:,20]-obs[:,21])).mean()),
        feedback_gate_mean=float(gate.mean()),feedback_gate_min=float(gate.min()),feedback_gate_max=float(gate.max()),
        fast_mix_max=float(mix.max()),cpu_native_action_max_abs_error=float((predicted-recorded).abs().max()),
        feedback_forward_setpoint_increment_m_s=float((.3*(correction[:,4]-correction[:,5])).mean()))
    for index in range(6):
        metrics[f'policy_action{index}/low_tanh_signed_mean'] = float(raw[:,index].mean())
        metrics[f'policy_action{index}/low_tanh_abs_max'] = float(raw[:,index].abs().max())
        metrics[f'policy_action{index}/low_tanh_fraction_ge95percent'] = float((raw[:,index].abs() >= .95).float().mean())
    checks[str(count)+'_exact27_unique_finite_metrics'] = len(metrics) == 27 and all(np.isfinite(v) for v in metrics.values())
    old_events = {p:sha(p) for p in (RUN/trial).glob('events.out.tfevents.*')}
    with SummaryWriter(str(RUN/trial),filename_suffix='.backward_limits001') as writer:
        for tag, value in metrics.items():
            writer.add_scalar(prefix+'/'+tag,value,count)
    current_events = sorted(set((RUN/trial).glob('events.out.tfevents.*'))-set(old_events))
    checks[str(count)+'_all_previous_event_bytes_preserved'] = len(current_events) == 1 and all(sha(p) == digest for p,digest in old_events.items())
    accumulator = EventAccumulator(str(current_events[0]),size_guidance={'scalars':0});accumulator.Reload()
    expected = {prefix+'/'+tag for tag in metrics}
    checks[str(count)+'_TensorBoard_all27_unique_tags_roundtrip'] = set(accumulator.Tags()['scalars']) == expected and all(
        len(accumulator.Scalars(prefix+'/'+tag)) == 1
        and accumulator.Scalars(prefix+'/'+tag)[0].step == count
        and np.isclose(accumulator.Scalars(prefix+'/'+tag)[0].value,value,rtol=2e-7,atol=2e-8)
        for tag,value in metrics.items())
    versions[str(count)] = dict(checkpoint_sha256=sha(checkpoint),evaluation_sha256=sha(path),traces_sha256=sha(trace),
        metrics=metrics,event_file=str(current_events[0].relative_to(ROOT)),event_sha256=sha(current_events[0]))
checks['same_initial_observations_bit_exact'] = all(np.array_equal(initial[1120],initial[n]) for n in (1184,1248))
checks['current128_execution_sources_preserved'] = all(sha(ROOT/n) == digest for n,digest in read(RUN/'feedback1248_queue001.json')['source_sha256'].items())
report = dict(schema='flat-feedback1248-backward-archived-limit-readout-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
    passed=all(checks.values()),checks=checks,reader_sha256=sha(Path(__file__)),versions=versions,
    tensorboard_namespace=prefix,window='Active 50Hz archived samples after3s in backward_05; all four native first episodes full.',
    policy_action_order=['Lhip','Lknee','Rhip','Rknee','Lwheel','Rwheel'],
    units='speed m/s; gate, mix and tanh dimensionless; counts policy frames; action error raw policy units',
    scope='Same-input output-range diagnostics only; no closed-loop counterfactual, causal or unsampled1000Hz effort claim.',
    native_steps=0,optimizer_updates=0,used_to_fit_weights=False,behavior_qualified=False)
if not report['passed']:
    print(json.dumps(checks,indent=2));raise SystemExit('Archived action readout checks failed; preserve outputs for inspection')
output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),versions={k:v['metrics'] for k,v in versions.items()})))
