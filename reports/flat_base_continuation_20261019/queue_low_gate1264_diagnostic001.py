"""Run two short, serial counterfactuals after the1264 queue is sealed."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019'
PYTHON = '/home/noir/miniconda3/envs/isaacsim6/bin/python'
STATE = RUN / 'low_gate1264_queue001.json'
REVIEW = HERE / 'low_gate1264_diagnostic_review001.json'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
phases = (('low_gate1264_full_native001', 'full'),
          ('low_gate1264_floor_native001', 'low_gate_floor'))
if STATE.exists() or REVIEW.exists() or any((RUN/n).exists() or (RUN/(n+'.log')).exists() for n,_ in phases):
    raise SystemExit('Preserve all previous gate-isolation outputs')

previous_path = RUN / 'feedback1264_population_queue001.json'
previous = read(previous_path)
seal_path = HERE / 'feedback1264_population_completion001.json'
seal = read(seal_path)
if (previous['status'] != 'complete' or not seal['passed'] or seal['queue_sha256'] != sha(previous_path)
        or any(Path('/proc/'+str(previous[k])).exists() for k in ('pid','worker_pid'))):
    raise SystemExit('Wait for the complete1264 native queue to exit and be sealed')
checkpoint = RUN / 'pilot024/model_001264.pt'
basic = read(RUN / 'deployment1264_basic001/evaluation.json')
if (basic['status'] != 'evaluated' or basic['checkpoint_sha256'] != sha(checkpoint)
        or basic['result']['cases']['backward_05']['vx_mae_m_s'] <= .05):
    raise SystemExit('This counterfactual is specific to the unresolved1264 backward error')
candidate = HERE / 'diagnostic_low_gate_candidate001/evaluate.py'
cpu = read(candidate.with_name('cpu001.json'))
if not cpu['passed'] or cpu['candidate_sha256'] != sha(candidate):
    raise SystemExit('Authenticated same-input gate-only intervention required')
frozen = dict(previous['source_sha256'])
frozen[str(candidate.relative_to(ROOT))] = sha(candidate)
state = dict(schema='flat-low-gate1264-serial-diagnostic-queue-v1',
    status='starting', pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
    checkpoint_sha256=sha(checkpoint), completed_pilot_seal_sha256=sha(seal_path),
    cpu_proof_sha256=sha(candidate.with_name('cpu001.json')),
    source_sha256=frozen, history=[], optimizer_updates=0, checkpoint_saved=False,
    behavior_qualified=False, hardware_motion_started=False,
    scope='Two4-second32-environment50Hz/1000Hz development counterfactuals only')


def save():
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2)+'\n')
    temp.replace(STATE)


try:
    save()
    for name,variant in phases:
        if any(sha(ROOT/n) != value for n,value in frozen.items()):
            raise ValueError('Frozen diagnostic source changed')
        state.update(status='running', phase=name, variant=variant)
        with (RUN/(name+'.log')).open('x') as log:
            worker = subprocess.Popen([PYTHON, str(candidate), '--checkpoint', str(checkpoint),
                '--output', str(RUN/name), '--deployment-pd', '--diagnostic-ablation', variant,
                '--diagnostic-seconds', '4'], cwd=ROOT,
                env={**os.environ, 'OMNI_KIT_ACCEPT_EULA':'YES'}, stdout=log, stderr=subprocess.STDOUT)
            state['worker_pid'] = worker.pid
            save()
            code = worker.wait()
        state['history'].append(dict(phase=name, exit_code=code))
        save()
        if code:
            raise RuntimeError('Gate-isolation child failed: '+name)
    import numpy as np
    evaluations = {variant: read(RUN/name/'evaluation.json') for name,variant in phases}
    traces = {variant: np.load(RUN/name/'traces.npz') for name,variant in phases}
    full,off = (evaluations[n] for n in ('full','low_gate_floor'))
    prefix = min(len(t['active']) for t in traces.values())
    paired_active = (traces['full']['active'][:prefix].astype(bool)
                     & traces['low_gate_floor']['active'][:prefix].astype(bool))
    checks = dict(both_native_children_exit0=len(state['history'])==2 and all(r['exit_code']==0 for r in state['history']),
        same_checkpoint=all(d.get('checkpoint_sha256')==sha(checkpoint) for d in evaluations.values()),
        same_reset_seed=full['seed']==off['seed']==191029,
        both50Hz_policy1000Hz_physics=all(d['policy_hz']==50 and d['physics_hz']==1000 for d in evaluations.values()),
        both_finished=all(d['status']=='evaluated' for d in evaluations.values()),
        both_explicitly_ineligible=all(d['result']['diagnostic_only'] and not d['result']['qualification_eligible']
            and not d['result']['passed'] and not d['diagnostic_counterfactual']['qualification_eligible'] for d in evaluations.values()),
        zero_optimizer_updates=all(d['new_optimizer_updates']==d['diagnostic_counterfactual']['optimizer_updates']==0 for d in evaluations.values()),
        same_initial_observations=np.array_equal(traces['full']['observations'][0],traces['low_gate_floor']['observations'][0]),
        same_initial_leg_actions=np.allclose(traces['full']['actions'][0,:,:4],
            traces['low_gate_floor']['actions'][0,:,:4],atol=2e-5,rtol=2e-5),
        same_group_assignment=np.array_equal(traces['full']['groups'],traces['low_gate_floor']['groups']),
        same_command_inputs_on_common_active_frames=np.array_equal(
            traces['full']['observations'][:prefix][paired_active][:,:4],
            traces['low_gate_floor']['observations'][:prefix][paired_active][:,:4]),
        no_frozen_source_changes=all(sha(ROOT/n)==v for n,v in frozen.items()),
        checkpoint_bytes_unchanged=sha(checkpoint)==state['checkpoint_sha256'])
    checks = {name:bool(value) for name,value in checks.items()}
    metrics = {}
    for name in full['result']['cases']:
        rows = {}
        for variant,data in evaluations.items():
            case = data['result']['cases'][name]
            rows[variant] = {k:case[k] for k in ('vx_mae_m_s','yaw_mae_rad_s','height_mae_m',
                'stand_drift_max_m','full_horizon_episodes','termination_reasons')}
            values = traces[variant]
            mask = ((values['groups'][None,:]==name) & values['active'].astype(bool)
                    & (values['episode_ticks']*.02 > 2.))
            vx = values['velocity'][mask][:,0]
            rows[variant]['root_vx_signed_mean_after2s_m_s'] = float(vx.mean()) if len(vx) else None
        metrics[name] = rows
    report = dict(schema='flat-low-gate1264-short-counterfactual-review-v1',
        recorded_at=datetime.now(timezone.utc).isoformat(), passed=all(checks.values()), checks=checks,
        checkpoint_sha256=sha(checkpoint), evaluator_sha256=sha(candidate),
        evaluation_sha256={variant:sha(RUN/name/'evaluation.json') for name,variant in phases},
        trace_sha256={variant:sha(RUN/name/'traces.npz') for name,variant in phases},
        metrics=metrics, optimizer_updates=0, behavior_qualified=False, hardware_motion_started=False,
        scope='Paired4-second gate-only intervention with fixed weights; no10-second standing, full envelope, heldout or release conclusion')
    for values in traces.values():
        values.close()
    REVIEW.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if not report['passed']:
        raise RuntimeError('Gate-isolation pairing/provenance checks failed; preserve the negative review')
    state.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(), review_sha256=sha(REVIEW))
    save()
except Exception:
    state.update(status='failed', error=traceback.format_exc())
    save()
    raise
