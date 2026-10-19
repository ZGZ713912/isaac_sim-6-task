"""Complete the4-way feedback factorial after the wheel diagnostic is sealed."""
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
STATE = RUN / 'leg_feedback1120_queue001.json'
REVIEW = HERE / 'feedback1120_factorial_review001.json'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
phases = (('leg_feedback1120_off_native001', 'leg_feedback_off'),
          ('all_feedback1120_off_native001', 'all_feedback_off'))
if STATE.exists() or REVIEW.exists() or any((RUN/n).exists() or (RUN/(n+'.log')).exists() for n,_ in phases):
    raise SystemExit('Preserve all previous feedback-factorial outputs')

previous_path = RUN / 'wheel_feedback1120_queue001.json'
previous = read(previous_path)
seal_path = HERE / 'wheel_feedback1120_queue_completion001.json'
seal = read(seal_path)
if (previous['status'] != 'complete' or not seal['passed'] or seal['queue_sha256'] != sha(previous_path)
        or any(Path('/proc/'+str(previous[k])).exists() for k in ('pid','worker_pid'))):
    raise SystemExit('Wait for the complete1120 native queue to exit and be sealed')
checkpoint = RUN / 'pilot021/model_001120.pt'
basic = read(RUN / 'deployment1120_basic001/evaluation.json')
if (basic['status'] != 'evaluated' or basic['checkpoint_sha256'] != sha(checkpoint)
        or basic['result']['cases']['backward_05']['vx_mae_m_s'] <= .05):
    raise SystemExit('This counterfactual is specific to the unresolved1120 backward error')
candidate = HERE / 'diagnostic_leg_feedback_candidate/evaluate.py'
cpu = read(candidate.with_name('cpu001.json'))
if not cpu['passed'] or cpu['candidate_sha256'] != sha(candidate) or cpu['checkpoint_sha256'] != sha(checkpoint):
    raise SystemExit('Authenticated same-input leg/all-feedback interventions required')
frozen = dict(previous['source_sha256'])
frozen[str(candidate.relative_to(ROOT))] = sha(candidate)
state = dict(schema='flat-leg-feedback1120-serial-factorial-queue-v1',
    status='starting', pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
    checkpoint_sha256=sha(checkpoint), completed_wheel_diagnostic_seal_sha256=sha(seal_path),
    cpu_proof_sha256=sha(candidate.with_name('cpu001.json')),
    source_sha256=frozen, history=[], optimizer_updates=0, checkpoint_saved=False,
    behavior_qualified=False, hardware_motion_started=False,
    scope='Two new4-second modes complete the4-way feedback factorial;0PPO')


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
            raise RuntimeError('Feedback-factorial child failed: '+name)
    import numpy as np
    paths = {'full':'wheel_feedback1120_full_native001',
        'wheel_feedback_off':'wheel_feedback1120_off_native001',
        **{variant:name for name,variant in phases}}
    evaluations = {variant:read(RUN/name/'evaluation.json') for variant,name in paths.items()}
    traces = {variant:np.load(RUN/name/'traces.npz') for variant,name in paths.items()}
    full,base = evaluations['full'],traces['full']
    normalized_settings = lambda d:{k:v for k,v in d['settings'].items() if k not in ('protocol_id','ablation')}
    paired_commands = True
    for variant,values in traces.items():
        prefix = min(len(base['active']),len(values['active']))
        paired = base['active'][:prefix].astype(bool) & values['active'][:prefix].astype(bool)
        paired_commands &= np.array_equal(base['observations'][:prefix][paired][:,:4],
            values['observations'][:prefix][paired][:,:4])
    checks = dict(two_new_native_children_exit0=len(state['history'])==2 and all(r['exit_code']==0 for r in state['history']),
        four_same_checkpoint=all(d.get('checkpoint_sha256')==sha(checkpoint) for d in evaluations.values()),
        same_reset_seed=all(d['seed']==191029 for d in evaluations.values()),
        all50Hz_policy1000Hz_physics=all(d['policy_hz']==50 and d['physics_hz']==1000 for d in evaluations.values()),
        all_finished=all(d['status']=='evaluated' for d in evaluations.values()),
        same_case_settings_except_diagnostic_labels=all(normalized_settings(d)==normalized_settings(full) for d in evaluations.values()),
        same_original_mechanical_contract=all(all(d[k]==full[k] for k in ('source_onnx_sha256','asset_manifest_sha256','model_spec_sha256')) for d in evaluations.values()),
        all_explicitly_ineligible=all(d['result']['diagnostic_only'] and not d['result']['qualification_eligible']
            and not d['result']['passed'] and not d['diagnostic_counterfactual']['qualification_eligible'] for d in evaluations.values()),
        zero_optimizer_updates=all(d['new_optimizer_updates']==d['diagnostic_counterfactual']['optimizer_updates']==0 for d in evaluations.values()),
        all_same_initial_observations=all(np.array_equal(base['observations'][0],v['observations'][0]) for v in traces.values()),
        initial_wheel_actions_retained_when_only_legs_disabled=np.allclose(base['actions'][0,:,4:],traces['leg_feedback_off']['actions'][0,:,4:],atol=2e-5,rtol=2e-5),
        initial_leg_actions_retained_when_only_wheels_disabled=np.allclose(base['actions'][0,:,:4],traces['wheel_feedback_off']['actions'][0,:,:4],atol=2e-5,rtol=2e-5),
        same_group_assignment=all(np.array_equal(base['groups'],v['groups']) for v in traces.values()),
        same_command_inputs_on_common_active_frames=paired_commands,
        all130_frozen_sources_retained=len(frozen)==130 and all(sha(ROOT/n)==v for n,v in frozen.items()),
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
    report = dict(schema='flat1120-four-way-feedback-short-factorial-v1',
        recorded_at=datetime.now(timezone.utc).isoformat(),passed=all(checks.values()),checks=checks,
        checkpoint_sha256=sha(checkpoint),evaluator_sha256=sha(candidate),
        previous_wheel_diagnostic_seal_sha256=sha(seal_path),cpu_proof_sha256=sha(candidate.with_name('cpu001.json')),
        evaluation_sha256={variant:sha(RUN/name/'evaluation.json') for variant,name in paths.items()},
        trace_sha256={variant:sha(RUN/name/'traces.npz') for variant,name in paths.items()},
        metrics=metrics,optimizer_updates=0,behavior_qualified=False,hardware_motion_started=False,
        scope='Four development4-second interventions; no10-second, full-envelope, heldout or release conclusion')
    for values in traces.values():
        values.close()
    REVIEW.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not report['passed']:
        raise RuntimeError('Factorial pairing/provenance checks failed; preserve negative review')
    state.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(), review_sha256=sha(REVIEW))
    save()
except Exception:
    state.update(status='failed', error=traceback.format_exc())
    save()
    raise
