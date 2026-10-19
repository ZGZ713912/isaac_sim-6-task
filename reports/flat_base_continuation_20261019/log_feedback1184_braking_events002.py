"""Log archived braking diagnostics into existing RSL-RL TensorBoard log directories."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'runs/flat_base_20261019'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
receipt_path=HERE/'feedback1184_braking_event_receipt002.json'
if receipt_path.exists():raise SystemExit('Preserve completed braking event receipt')
readout_path=HERE/'feedback1184_braking_readout001.json'
readout=read(readout_path)
if not readout['passed'] or len(readout['checks'])!=53 or not all(readout['checks'].values()):
    raise SystemExit('Require the checked archived action readout')
namespace='FlatBrakingAudit/flat-base-paired107-rmcs-default-height-envelope-v1/physics_1000Hz/seed_191029'
files={};checks={};scalar_counts={}
for step,trial in ((1120,'pilot021'),(1184,'pilot022')):
    directory=RUN/trial
    before=read(HERE/f'feedback{step}_training_event_review001.json')['event_file_sha256']
    if any(sha(ROOT/n)!=v for n,v in before.items()):raise SystemExit('An earlier sealed event file changed')
    if list(directory.glob('events.out.tfevents.*.braking_readout002')):
        raise SystemExit('Preserve every earlier braking event file')
    expected={}
    for name,case in readout['versions'][str(step)]['cases'].items():
        for stop in case['stops']:
            direction='forward_braking' if stop['incoming_sign']>0 else 'backward_braking'
            prefix=f'{namespace}/{name}/{direction}'
            for metric in ('speed_peak_after1s_m_s','speed_peak_after3s_m_s','mean_speed_after3s_m_s',
                    'first2s_low_MLP_abs_tanh_max_wheels','first2s_low_MLP_abs_tanh_max_legs',
                    'first2s_low_MLP_wheel_fraction_at_or_above95percent',
                    'first2s_low_MLP_leg_fraction_at_or_above95percent','first2s_tracking_gate_min'):
                expected[prefix+'/'+metric]=stop[metric]
            settling=stop['actual_per_lane_settles_by_end_of_segment_s']
            if any(v is None for v in settling):raise SystemExit('This receipt requires all observed stops to settle')
            for metric,value in (('settle_time_max_s',max(settling)),('settle_time_min_s',min(settling)),
                    ('settle_time_mean_s',sum(settling)/len(settling))):expected[prefix+'/'+metric]=value
    required_prefixes={f'{namespace}/{case}/{direction}' for case in ('start_stop_1','start_stop_2','start_stop_3') for direction in ('forward_braking','backward_braking')}
    if len(expected)!=66 or {tag.rsplit('/',1)[0] for tag in expected}!=required_prefixes:
        raise SystemExit('Require exactly three cases times two directions times11metrics')
    checks[str(step)+'_all_six_case_direction_pairs_and66_metrics']=True
    with SummaryWriter(str(directory),filename_suffix='.braking_readout002') as writer:
        for tag,value in expected.items():writer.add_scalar(tag,value,step)
    added=list(directory.glob('events.out.tfevents.*.braking_readout002'))
    if len(added)!=1:raise SystemExit('Exactly one diagnostic event file expected per existing log directory')
    accumulator=EventAccumulator(str(added[0]),size_guidance={'scalars':0});accumulator.Reload()
    actual=accumulator.Tags()['scalars']
    checks[str(step)+'_exact_tags_steps_and_finite_values']=set(actual)==set(expected) and all(
        len(accumulator.Scalars(tag))==1 and accumulator.Scalars(tag)[0].step==step
        and np.isfinite(accumulator.Scalars(tag)[0].value)
        and np.isclose(accumulator.Scalars(tag)[0].value,value,atol=1e-7,rtol=2e-7)
        for tag,value in expected.items())
    checks[str(step)+'_all_previous_sealed_event_files_unchanged']=all(sha(ROOT/n)==v for n,v in before.items())
    files[str(added[0].relative_to(ROOT))]=sha(added[0]);scalar_counts[str(step)]=len(expected)
receipt=dict(schema='flat-feedback1184-archived-braking-tensorboard-receipt-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
    passed=all(checks.values()),checks=checks,readout_sha256=sha(readout_path),logger_source_sha256=sha(Path(__file__)),
    source='Existing checked native replay traces; derived diagnostic events in original RSL log directories',
    tensorboard_namespace=namespace,event_step_unit='completed optimizer updates in exact associated checkpoint',
    velocity_unit='m/s',settling_time_unit='s;50Hz end-state resolution; remains within.03m/s through end of that8s stop segment',
    event_file_sha256=files,scalar_counts=scalar_counts,previous_training_and_evaluation_files_unchanged=True,
    local_tensorboard_server_started=False,native_environment_steps=0,optimizer_updates_performed=0,
    behavior_qualified=False,hardware_motion_started=False)
if not receipt['passed']:
    print(json.dumps(receipt,indent=2));raise SystemExit('Braking metrics did not round-trip through TensorBoard events')
receipt_path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps(receipt,indent=2))
