"""Rescore immutable017 native traces after the float32 clock validation fix.

The original run did not save its initial roots. A same-seed native reset is
accepted as an explicit reconstructed reference only if its complete2s prefix
matches all shared measurements byte for byte. No simulation or PPO is run.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL=Path('reports/robust_policy_repair_20261024')
FULL=BASE/REL/'fresh017_full_native_nominal001'
ANCHOR=BASE/REL/'fresh017_peak_native_nominal002'
OUT=BASE/REL/'fresh017_full_native_nominal001_scoring_repair001'
sys.path.insert(0,str(ROOT/'scripts'))
from wheel_leg_full_suite import analyze_case


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    full,anchor=read(FULL/'receipt.json'),read(ANCHOR/'receipt.json')
    assert full['native_policy_steps']==6000 and not Path(f"/proc/{full['pid']}").exists()
    assert anchor['status']=='completed_diagnostic_unqualified' and not Path(f"/proc/{anchor['pid']}").exists()
    assert full['checkpoint_sha256']==anchor['checkpoint_sha256'] and full['seed']==anchor['seed']==271301
    assert anchor['profile']=='nominal' and anchor['passive_clock_intervention'] is None
    assert not OUT.exists()
    inputs={str(p):sha(p) for p in (FULL/'receipt.json',FULL/'native_trace.npz',
        FULL/'cases.json',FULL/'spec.json',FULL/'requirements.json',ANCHOR/'receipt.json',ANCHOR/'native_peak_trace.npz')}
    assert all(sha(FULL/'diagnostic_sources'/n)==h for n,h in full['diagnostic_source_hashes'].items())
    a=dict(np.load(FULL/'native_trace.npz'))
    b=dict(np.load(ANCHOR/'native_peak_trace.npz'))
    fields=dict(valid='valid',leg_dq='dq',leg_torque='torque',gravity='gravity',
        velocity='velocity',omega='omega',position='local_position',
        passive_at_interval_start='passive',recovery_at_interval_start='recovery')
    prefix={k:np.array_equal(a[k][:400],b[v]) for k,v in fields.items()}
    prefix['height']=np.array_equal(a['position'][:400,:,2],b['height'])
    assert all(prefix.values()),prefix
    initial=np.asarray(anchor['initial_root_local_xyzw'])
    cases,spec,requirements=[read(FULL/n) for n in ('cases.json','spec.json','requirements.json')]
    assert initial.shape==(len(cases),7) and np.isfinite(initial).all()
    rows=[analyze_case(c,a,i,initial[i],spec,requirements) for i,c in enumerate(cases)]
    groups=[]
    for cap in requirements['capabilities']:
        selected=[r for r in rows if cap['id'] in r['case']['capabilities']]
        groups.append(dict(capability=cap['id'],case_count=len(selected),
            engineering_pass_count=sum(r['engineering_pass'] for r in selected),
            full_horizon_count=sum(r['full_horizon'] for r in selected),
            native_guard_count=sum(r['native_guard_triggered'] for r in selected),
            robustness_across_domains_confirmed=False))
    OUT.mkdir()
    from rsl_rl.utils.logger import Logger
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run=Path(full['checkpoint']).parent
    logger=Logger(log_dir=str(OUT),cfg=read(run/'agent_config.json'),env_cfg={},num_envs=len(cases),
        is_distributed=False,gpu_world_size=1,gpu_global_rank=0,device='cpu')
    logger.init_logging_writer()
    metrics={}
    for group in groups:
        for k,v in group.items():
            if isinstance(v,(int,float)):
                metrics[f"FullSuite/{group['capability']}/{k}"]=v
    paths=('tilt_peak_deg','closure_gap_peak_m','path/max_xy_excursion_m',
        'tracking/raw_speed_mae_m_s','tracking/raw_yaw_mae_rad_s','tracking/height_mae_m',
        'spin/actual_body_signed_rpm_mean','smoothness/leg_speed_physical_peak_rad_s',
        'smoothness/leg_angle_3_95hz_rms_mrad')
    for cap in groups:
        selected=[r for r in rows if cap['capability'] in r['case']['capabilities']]
        for path in paths:
            values=[]
            for row in selected:
                value=row
                for key in path.split('/'):
                    value=value.get(key) if isinstance(value,dict) else None
                if isinstance(value,(int,float)):values.append(value)
            if values:
                metrics[f"FullSuite/{cap['capability']}/{path}/mean"]=sum(values)/len(values)
                metrics[f"FullSuite/{cap['capability']}/{path}/max"]=max(values)
    step=full['completed_updates']
    for tag,value in metrics.items():logger.writer.add_scalar(tag,value,step)
    logger.writer.flush();logger.writer.close()
    events=EventAccumulator(str(OUT),size_guidance={'scalars':0}).Reload()
    assert all(len(events.Scalars(t))==1 and events.Scalars(t)[0].step==step
        and np.isclose(events.Scalars(t)[0].value,v,rtol=2e-6,atol=2e-8) for t,v in metrics.items())
    assert all(sha(Path(p))==v for p,v in inputs.items())
    result=dict(schema='fresh-full-suite-scoring-repair-v1',observed_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint_sha256=full['checkpoint_sha256'],completed_updates=step,
        native_policy_steps=6000,cases=rows,capabilities=groups,input_sha256=inputs,
        repaired_scoring_source_sha256=sha(ROOT/'scripts/wheel_leg_full_suite.py'),producer_sha256=sha(Path(__file__)),
        initial_reference='Reconstructed same-seed native reset, validated against all10 common2s trace fields; original initial roots were not persisted.',
        matched_native_prefix_fields=prefix,initial_root_local_xyzw=initial.tolist(),
        original_receipt_not_overwritten=True,original_metrics_not_overwritten=True,
        new_native_steps=0,new_PPO=0,optimizer_constructed=False,RSL_values_read_back=len(metrics),
        overall_policy_qualified=False,hardware_ready=False,
        note='Stopped456-update diagnostic in one nominal seed; no full1536 endpoint or cross-domain qualification.')
    write(OUT/'result.json',result)
    for base in (ROOT,BASE):
        (base/REL/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        write(base/REL/'fresh017_full_native_nominal001_scoring_repair001.json',result)
    print(json.dumps(dict(capabilities=groups,RSL_values_read_back=len(metrics)),ensure_ascii=False))


if __name__=='__main__':main()
