"""CPU review of existing native traces; no threshold or trajectory change."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from rsl_rl.utils.logger import Logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL=Path('reports/robust_policy_repair_20261024')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    output=BASE/REL/'passive_startup_review001'
    output.mkdir(exist_ok=False)
    run=BASE/'runs/fresh_unified_20261027/pilot016_scoped_failure_requested_mixed1536'
    logger=Logger(log_dir=str(output),cfg=read(run/'agent_config.json'),env_cfg={},num_envs=198,
        is_distributed=False,gpu_world_size=1,gpu_global_rank=0,device='cpu')
    logger.init_logging_writer();groups=[];expected={};inputs={}
    try:
        for profile in ('nominal','requested'):
            folder=BASE/REL/f'fresh016_peak_native_{profile}001'
            receipt=read(folder/'receipt.json');result=read(folder/'result.json');cases=read(folder/'cases.json')
            assert receipt['status']=='completed_diagnostic_unqualified'
            assert sha(folder/'native_peak_trace.npz')==result['trace_sha256']==receipt['trace_sha256']
            inputs[profile]=dict(receipt_sha256=sha(folder/'receipt.json'),trace_sha256=result['trace_sha256'],
                cases_sha256=sha(folder/'cases.json'),checkpoint_sha256=receipt['checkpoint_sha256'])
            events=EventAccumulator(str(folder),size_guidance={'scalars':0}).Reload()
            with np.load(folder/'native_peak_trace.npz',allow_pickle=False) as data:
                dq=data['dq'];valid=data['valid'];passive=data['passive'];recovery=data['recovery']
                phases=dict(passive_placement=passive,active_recovery=~passive&recovery,
                    ordinary_active_control=~passive&~recovery)
                for name,phase in phases.items():
                    peaks=np.where((valid&phase)[:,None,:,None],np.abs(dq),0.).max(axis=(0,1,3))
                    ids=np.flatnonzero(peaks>15.)
                    assert np.isclose(peaks.max(),events.Scalars(f'PeakWindow/all/{name}/peak_leg_speed_rad_s')[0].value)
                    row=dict(profile=profile,phase=name,cases_above_15_rad_s=len(ids),
                        case_peak_witnesses=[dict(lane=int(i),case=cases[i],peak_rad_s=float(peaks[i])) for i in ids])
                    groups.append(row)
                    tag=f'PeakStartupReview/{profile}/{name}/cases_above_15_rad_s'
                    expected[tag]=len(ids);logger.writer.add_scalar(tag,len(ids),330)
        logger.writer.flush()
    finally:
        logger.writer.close()
    events=EventAccumulator(str(output),size_guidance={'scalars':0}).Reload()
    assert all(len(events.Scalars(k))==1 and events.Scalars(k)[0].step==330
        and events.Scalars(k)[0].value==v for k,v in expected.items())
    result=dict(schema='passive-startup-trace-review-v1',observed_utc=datetime.now(timezone.utc).isoformat(),
        groups=groups,input_hashes=inputs,RSL_values_read_back=len(expected),
        existing_engineering_alert_rad_s=15.,threshold_changed=False,existing_native_steps_only=True,
        new_native_steps=0,new_PPO=0,optimizer_constructed=False,qualified=False,hardware_ready=False,
        interpretation='The original full-suite alert includes passive samples. In this recorded independent2s prefix,42/43 initial-recovery cases already exceeded15rad/s before RL could apply motor torque. Active recovery also exceeded15 in all48 recovery cases. This is an initialization/control investigation, not permission to omit samples, lower the alert, or infer future results. Shorter zero-effort placement is a candidate requiring a new explicit native comparison after the active learner exits.',
        producer_sha256=sha(Path(__file__)))
    raw=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    for base in (ROOT,BASE):(base/REL/'passive_startup_review001.json').write_text(raw)
    (output/'producer.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(RSL_values_read_back=len(expected),groups=[{k:v for k,v in r.items() if k!='case_peak_witnesses'} for r in groups])))


if __name__=='__main__':main()
