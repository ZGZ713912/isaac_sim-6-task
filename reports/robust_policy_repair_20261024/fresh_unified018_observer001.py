"""Read one immutable RSL prefix of the exact original018 learner; no PPO."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL=Path('reports/robust_policy_repair_20261024')
RUN=BASE/'runs/fresh_unified_20261027/pilot018_velocity_quartic_physical_speed_mixed1536'
sys.path.insert(0,str(ROOT/'scripts'))
from read_domain_admission_progress import observe
from seal_scratch_standing import original_process_live
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-name',required=True)
    parser.add_argument('--window',type=int,default=32)
    args=parser.parse_args()
    target=ROOT/REL/(args.output_name+'.json')
    if target.exists():raise ValueError('Preserve the original observation')
    population=observe(RUN,args.window)
    progress=read(RUN/'progress.json');identity=read(BASE/REL/'fresh_unified018_launch_identity001.json')
    sources=read(RUN/'source_hashes.json');ledger_path=BASE/REL/'scratch_campaign001.json'
    ledger_before=ledger_path.read_bytes();ledger=json.loads(ledger_before)
    event,=RUN.glob('events.out.tfevents.*');raw=event.read_bytes()
    with tempfile.TemporaryDirectory() as temporary:
        frozen=Path(temporary)/event.name;frozen.write_bytes(raw)
        events=EventAccumulator(str(frozen),size_guidance={'scalars':0}).Reload()
        tags=[k for k in events.Tags()['scalars'] if k.startswith(
            ('FreshTasks/','FreshObjective/','FreshStandingFailure/','ScratchStandingExtrema/','PhysicalPeakWitness/'))]
        values={tag:{p.step:p.value for p in events.Scalars(tag)} for tag in tags}
    count=population['common_complete_RSL_updates']
    if any(any(i not in row for i in range(count)) for row in values.values()):
        raise ValueError('Wait for aligned real reward, phase and extrema tags')
    if any(not math.isfinite(row[i]) for row in values.values() for i in range(count)):
        raise ValueError('Nonfinite scalar')
    checks=dict(exact_live_training_identity=original_process_live(identity)
        and (progress['pid'],progress['start_ticks'])==(identity['pid'],identity['start_ticks']),
        all274_current_and_archive_sources_kept=len(sources)==274 and all(
            sha(ROOT/n)==v==sha(RUN/'source_snapshot'/n) for n,v in sources.items()),
        exact_GPU_owner=str(identity['pid']) in subprocess.check_output(
            ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines(),
        no_prior_checkpoint_read=progress['checkpoint_reads']==[],
        same_campaign_owner=(ledger['active_pid'],ledger['active_start_ticks'])==(identity['pid'],identity['start_ticks']))
    queue=None
    if ledger.get('active_full_suite_queue'):
        q=read(Path(ledger['active_full_suite_queue']));directory=Path(ledger['active_full_suite_queue']).parent
        checks['queue_sources_kept']=len(q['source_sha256'])==12 and all(
            sha(ROOT/n)==v==sha(directory/'source_snapshot'/n) for n,v in q['source_sha256'].items())
        checks['queue_same_owner']=(q['training_identity']['pid'],q['training_identity']['start_ticks'])==(identity['pid'],identity['start_ticks'])
        queue={k:q[k] for k in ('pid','start_ticks','status','active_child')}
    assert all(checks.values()),checks
    windows=[]
    selected=[k for k in tags if k.startswith(('FreshTasks/standing/','FreshObjective/'))]
    for first in range(0,count//args.window*args.window,args.window):
        windows.append(dict(updates=[first+1,first+args.window],pure_standing_and_objective={tag:
            sum(values[tag][i] for i in range(first,first+args.window))/args.window for tag in selected}))
    peak=values['ScratchStandingExtrema/peak_leg_speed_rad_s']
    witness={k.split('/',1)[1]:v for k,v in values.items() if k.startswith('PhysicalPeakWitness/')}
    assert len(witness)==11
    for i in range(count):
        assert witness['speed_rad_s'][i]==peak[i] and abs(witness['signed_speed_rad_s'][i])==peak[i]
        assert witness['phase_code'][i] in (0.,1.,2.,3.)
        assert math.isclose(witness['motor_power_w'][i],witness['signed_speed_rad_s'][i]*witness['applied_torque_nm'][i],rel_tol=2e-6,abs_tol=2e-6)
    peak_update=max(range(count),key=lambda i:peak[i])
    high=values['ScratchStandingExtrema/leg_motor_samples_at_or_above_50_rad_s']
    result=dict(schema='fresh-unified018-live-observation-v1',observed_utc=datetime.now(timezone.utc).isoformat(),
        checks=checks,all_checks_pass=True,progress_completed_updates=progress['completed_updates'],population=population,
        pure_standing_and_objective_windows=windows,pure_standing_and_objective_latest={k:values[k][count-1] for k in selected},
        all_task_latest={k:v[count-1] for k,v in values.items() if k.startswith('FreshTasks/')},queue=queue,
        extrema=dict(complete_RSL_updates=count,peak_leg_speed_rad_s=max(peak[i] for i in range(count)),
            peak_update_one_based=peak_update+1,global_peak_witness={k:v[peak_update] for k,v in witness.items()},
            phase_counts_at_rollout_maxima={str(code):sum(v==code for v in witness['phase_code'].values()) for code in range(4)},
            leg_motor_samples_at_or_above_50_rad_s=sum(high[i] for i in range(count)),samples_per_update=655360),
        event_prefix=dict(path=str(event),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),
        prior_actual_consumed_updates=5068,total_actual_consumed_updates_observed=5068+progress['completed_updates'],
        new_PPO_by_reader=0,native_steps_by_reader=0,optimizer_constructed=False,qualified=False,hardware_ready=False,
        scope='Training means contain real failure resets; pure-standing tags contain all native active standing lanes, including not-yet-PPO-admitted random lanes. DomainAdmission means mix task roles after192. No cold behavior pass.',
        producer_sha256=sha(Path(__file__)))
    for base in (ROOT,BASE):write(base/REL/target.name,result)
    (BASE/REL/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    assert ledger_path.read_bytes()==ledger_before
    observed=max(progress['completed_updates'],ledger['active_completed_updates_observed'])
    ledger.update(active_completed_updates_observed=observed,completed_updates_observed=observed,
        total_actual_consumed_training_updates=5068+observed,latest_observed_utc=result['observed_utc'],
        latest_observation_file=str(BASE/REL/target.name))
    tmp=ledger_path.with_suffix('.observer018.tmp');write(tmp,ledger);tmp.replace(ledger_path)
    print(json.dumps(dict(completed_updates=progress['completed_updates'],all_checks_pass=True,
        latest_window=windows[-1],latest_nominal_window=population['windows'][-1]['groups']['nominal'],extrema=result['extrema'])))


if __name__=='__main__':main()
