"""Seal the unchanged-horizon native static-recall counterfactual."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT/'runs/flat_base_20261019'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
out = HERE/'static_recall1280_maximum_review001.json'
if out.exists():
    raise SystemExit('Preserve the completed native diagnostic review')
queue_path = RUN/'static_recall1280_maximum_queue001.json'
queue = read(queue_path)
if queue['status'] != 'complete' or any(Path('/proc/'+str(queue[k])).exists() for k in ('pid','worker_pid')):
    raise SystemExit('Wait for the diagnostic and both owners to exit')
contract_path = HERE/'static_recall1280_maximum_contract001.json'
contract = read(contract_path)
candidate_path = RUN/'static_recall1280_maximum_envelope001/evaluation.json'
candidate = read(candidate_path)
parent_path = RUN/'rmcs_maximum1280_envelope001/evaluation.json'
parent = read(parent_path)
old_path = RUN/'rmcs_maximum1248_envelope001/evaluation.json'
old = read(old_path)
checks = dict(native_child_exit0=queue['exit_code'] == 0,
    exact_contract=sha(contract_path) == queue['contract_sha256'],
    all_frozen_sources_exact=all(sha(ROOT/name) == value for name,value in queue['source_sha256'].items()),
    exact_completed_evaluation=sha(candidate_path) == queue['evaluation_sha256'] and candidate['status'] == 'evaluated',
    original_full_horizons_and_thresholds=candidate['settings'] == parent['settings'] == old['settings'],
    same_seed_clocks_and_mechanics=all(candidate[k] == parent[k] == old[k]
        for k in ('seed','policy_hz','physics_hz','source_onnx_sha256','asset_manifest_sha256','model_spec_sha256')),
    same1280_checkpoint=candidate['checkpoint_sha256'] == queue['checkpoint_sha256'] == parent['checkpoint_sha256'],
    unchanged_input_checkpoint=sha(RUN/'pilot025/model_001280.pt') == queue['checkpoint_sha256'],
    exact_static1248_checkpoint=sha(RUN/'pilot023/model_001248.pt') == queue['recalled_checkpoint_sha256'],
    forced_diagnostic_only=not candidate['result']['passed'] and candidate['result']['diagnostic_only']
        and not candidate['result']['qualification_eligible'],
    zero_optimizer_writes_and_no_model_output=queue['optimizer_updates'] == 0
        and candidate['new_optimizer_updates'] == 0 and not list((RUN/'static_recall1280_maximum_envelope001').glob('*.pt')),
    heldout_unused_and_no_hardware=not queue['heldout_seed_used'] and not queue['hardware_motion_started'])
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
from wheeled_algo.command_residual import mean_from_checkpoint
from compare_flat_base import compare
candidate_source = HERE/'static_recall_candidate001/static_recall.py'
spec = importlib.util.spec_from_file_location('static_recall_native_review',candidate_source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(2)
source = ROOT/'models/wheel_leg_flat_base_20261018/source.onnx'
moving = mean_from_checkpoint(torch.load(RUN/'pilot025/model_001280.pt',map_location='cpu',weights_only=True),source)
static = mean_from_checkpoint(torch.load(RUN/'pilot023/model_001248.pt',map_location='cpu',weights_only=True),source)
mean = module.StaticRecallActor(moving,static)
trace_path = RUN/'static_recall1280_maximum_envelope001/traces.npz'
trace = np.load(trace_path)
parent_trace = np.load(RUN/'rmcs_maximum1280_envelope001/traces.npz')
old_trace = np.load(RUN/'rmcs_maximum1248_envelope001/traces.npz')
checks['same_case_and_initial_observations'] = (np.array_equal(trace['groups'],parent_trace['groups'])
    and np.array_equal(trace['groups'],old_trace['groups'])
    and np.array_equal(trace['observations'][0],parent_trace['observations'][0])
    and np.array_equal(trace['observations'][0],old_trace['observations'][0]))
observations = torch.from_numpy(trace['observations'][trace['active'].astype(bool)])
saved = torch.from_numpy(trace['actions'][trace['active'].astype(bool)])
with torch.inference_mode():
    error = float((mean(observations)-saved).abs().max())
checks['CPU_composition_reproduces_all_active_native_actions'] = error < 5e-6
static_equality = {}
for name in ('stand_230mm','stand_305mm','stand_430mm'):
    lanes = np.flatnonzero(trace['groups'] == name)
    old_lanes = np.flatnonzero(old_trace['groups'] == name)
    active = trace['active'][:,lanes].astype(bool)
    equal = {}
    for key in ('observations','actions','velocity','position','height','gravity'):
        equal[key] = np.array_equal(trace[key][:,lanes][active],old_trace[key][:,old_lanes][active])
    static_equality[name] = equal
trace.close()
parent_trace.close()
old_trace.close()
events = EventAccumulator(str(RUN/'pilot025'),size_guidance={'scalars':0})
events.Reload()
tags = set(events.Tags()['scalars'])
cases = candidate['result']['cases']
expected = {candidate['tensorboard_namespace']+'/passed_cases':sum(c['passed'] for c in cases.values())}
for name,case in cases.items():
    for metric in ('survival_rate','height_mae_m','vx_mae_m_s','yaw_mae_rad_s',
                   'stand_drift_max_m','planar_speed_mean_m_s','closure_gap_max_m'):
        expected[candidate['tensorboard_namespace']+'/'+name+'/'+metric] = case[metric]
event_checks = []
for tag,value in expected.items():
    points = events.Scalars(tag) if tag in tags else []
    event_checks.append(len(points) == 1 and points[0].step == 1280
        and math.isclose(points[0].value,value,rel_tol=2e-7,abs_tol=2e-8))
checks['metrics_in_existing_RSL_events'] = all(event_checks)
if not all(checks.values()):
    print(json.dumps(checks,indent=2))
    raise SystemExit('Native diagnostic identity or reproduction failed; preserve all outputs')
result = dict(schema='flat-static-recall1280-maximum-height-native-review-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(),passed=True,checks=checks,
    queue_sha256=sha(queue_path),contract_sha256=sha(contract_path),evaluation_sha256=sha(candidate_path),
    trace_sha256=sha(trace_path),checkpoint_sha256=queue['checkpoint_sha256'],
    recalled_checkpoint_sha256=queue['recalled_checkpoint_sha256'],
    passed_cases=sum(c['passed'] for c in cases.values()),requested_cases=len(cases),
    full_horizon_episodes=sum(c['full_horizon_episodes'] for c in cases.values()),
    requested_episodes=sum(c['requested_episodes'] for c in cases.values()),
    all_active_CPU_native_max_abs_error=error,static_trace_bitwise_comparison=static_equality,
    comparisons=dict(parent1280=compare(parent,candidate),recalled1248=compare(old,candidate)),
    cases=cases,optimizer_updates=0,behavior_qualified=False,heldout_seed_used=False,hardware_motion_started=False,
    scope='Fixed-weight full-horizon diagnostic; no checkpoint or qualification. CPU equality and trajectory equality are reported separately; braking must retain its original verdict.')
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),passed_cases=result['passed_cases'],
    static_trace_bitwise_comparison=static_equality,all_active_CPU_native_max_abs_error=error),indent=2))
