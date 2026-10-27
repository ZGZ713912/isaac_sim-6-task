"""Close the exited016 attempt and account every actual update; no simulator/PPO."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL=Path('reports/robust_policy_repair_20261024')
RUN=BASE/'runs/fresh_unified_20261027/pilot016_scoped_failure_requested_mixed1536'
sys.path.insert(0,str(ROOT/'scripts'))
from seal_scratch_standing import finite,original_process_live,trainable_parameters
from read_domain_admission_progress import observe
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    target=ROOT/REL/'fresh_unified016_stopped_closure001.json'
    if target.exists():raise ValueError('Preserve the first completed closure')
    identity=read(BASE/REL/'fresh_unified016_launch_identity001.json')
    progress=read(RUN/'progress.json');count=progress['completed_updates']
    queue_path=BASE/REL/'fresh_full_suite_queue016b'
    queue=read(queue_path/'receipt.json');sources=read(RUN/'source_hashes.json')
    torch.set_num_threads(2)
    initial,final=[torch.load(RUN/name,map_location='cpu',weights_only=False)
                   for name in ('model_initial.pt','model_stopped.pt')]
    optimizer=final['optimizer_state_dict'];params=trainable_parameters(final)
    adam_steps=sorted({int(v['step']) for v in optimizer['state'].values()})
    event,=RUN.glob('events.out.tfevents.*')
    events=EventAccumulator(str(event),size_guidance={'scalars':0}).Reload()
    selected=[tag for tag in events.Tags()['scalars'] if tag.startswith(
        ('ScratchStandingExtrema/','FreshStandingFailure/','FreshTasks/standing/'))]
    selected+=['DomainAdmission/nominal/tilt_mean_deg']
    rows={tag:events.Scalars(tag) for tag in selected}
    regular=len(rows['DomainAdmission/nominal/tilt_mean_deg'])
    extrema={k:v for k,v in rows.items() if k.startswith('ScratchStandingExtrema/')}
    checks=dict(
        exact_original_process_and_GPU_exited=not original_process_live(identity) and str(identity['pid']) not in
            subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines(),
        queue_exited_without_child=not original_process_live(queue) and queue['status']=='stopped' and queue['active_child'] is None,
        all273_current_and_archived_sources_match=len(sources)==273 and all(sha(ROOT/n)==v==sha(RUN/'source_snapshot'/n) for n,v in sources.items()),
        queue12_current_and_archived_sources_match=len(queue['source_sha256'])==12 and all(
            sha(ROOT/n)==v==sha(queue_path/'source_snapshot'/n) for n,v in queue['source_sha256'].items()),
        committed_archive_kept=sha(RUN/'committed_source.tar')==identity['source_archive_sha256'],
        stopped_at_completed_update=progress['status']=='stopped' and count==progress['optimizer_attempts']==final['iter']==final['infos']['completed_updates'],
        no_optimizer_in_flight=progress['optimizer_update_in_flight'] is False,
        no_old_checkpoint_teacher=progress['checkpoint_reads']==[] and all(initial['infos']['initialization'][k] is False for k in ('checkpoint_loaded','teacher_loaded')),
        initial_zero_empty_Adam=initial['iter']==initial['infos']['completed_updates']==0 and not initial['optimizer_state_dict']['state'],
        all18_Adam_parameters=len(optimizer['state'])==sum(len(v) for v in params.values())==18,
        every_Adam_step_accounted=adam_steps==[12*count],
        full_state_and_RNG_finite=finite(initial) and finite(final) and all(k in final['infos'] for k in ('torch_rng','cuda_rng','env_rng','domain_rng')),
        stopped_checkpoint_receipt=sha(RUN/'model_stopped.pt')==progress['last_checkpoint']['sha256'] and progress['last_checkpoint']['updates']==count,
        frozen_manifest_matches=sources==initial['infos']['source_hashes']==final['infos']['source_hashes'],
        every_true_physical_extrema_update_recorded=all(len(v)==count and [r.step for r in v]==list(range(count)) for v in extrema.values()),
        regular_RSL_stops_before_final_exception=regular==count-1 and all(len(v)==regular and [r.step for r in v]==list(range(regular)) for k,v in rows.items() if k not in extrema),
        every_selected_scalar_finite=all(finite(r.value) for v in rows.values() for r in v))
    if not all(checks.values()):raise ValueError(checks)
    task_tags=[k for k in selected if k.startswith('FreshTasks/standing/')]
    windows=[]
    for first in range(0,regular//32*32,32):
        windows.append(dict(updates=[first+1,first+32],pure_standing_metrics={k.rsplit('/',1)[-1]:
            sum(r.value for r in rows[k][first:first+32])/32 for k in task_tags}))
    result=dict(schema='fresh-stopped-behavior-attempt-closure-v2',observed_utc=datetime.now(timezone.utc).isoformat(),
        run=str(RUN),pid=identity['pid'],start_ticks=identity['start_ticks'],status='stopped',requested_updates=1536,
        actual_consumed_updates=count,checks=checks,all_checks_pass=True,checkpoint=str(RUN/'model_stopped.pt'),
        checkpoint_sha256=sha(RUN/'model_stopped.pt'),source_count=len(sources),source_commit=identity['commit'],
        Adam_states=18,Adam_step_values=adam_steps,RSL_regular_updates=regular,RSL_true_extrema_updates=count,
        peak_leg_speed_rad_s=max(r.value for r in extrema['ScratchStandingExtrema/peak_leg_speed_rad_s']),
        leg_motor_samples_at_or_above_50_rad_s=sum(r.value for r in extrema['ScratchStandingExtrema/leg_motor_samples_at_or_above_50_rad_s']),
        pure_standing_windows=windows,pure_standing_latest={k:rows[k][-1].value for k in task_tags},
        population=observe(RUN,32),RSL_event=dict(path=str(event),bytes=event.stat().st_size,sha256=sha(event)),
        previous_actual_consumed_updates=4282,total_actual_consumed_training_updates=4282+count,
        stopped_attempt_consumed_updates=56+count,fully_completed_updates=4225,failed_attempt_consumed_updates=1,
        log_boundary='Final completed update330 has true extrema and full checkpoint; native logger ends329 because the stop exception occurs before logger.log. No consumption is removed.',
        qualification=False,hardware_ready=False,new_PPO_by_reader=0,native_steps_by_reader=0,
        optimizer_constructed=False,continuation_forbidden=True,producer_sha256=sha(Path(__file__)))
    for base in (ROOT,BASE):write(base/REL/target.name,result)
    (BASE/REL/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    ledger_path=BASE/REL/'scratch_campaign001.json';before=ledger_path.read_bytes();ledger=json.loads(before)
    assert (ledger['active_pid'],ledger['active_start_ticks'])==(identity['pid'],identity['start_ticks'])
    ledger.update(active_run=None,active_pid=None,active_start_ticks=None,active_completed_updates_observed=0,
        completed_updates_observed=count,total_actual_consumed_training_updates=4282+count,
        stopped_attempt_consumed_updates=56+count,latest_stopped_attempt_closure=str(BASE/REL/target.name),
        active_full_suite_queue=None,active_full_suite_queue_pid=None,active_full_suite_queue_start_ticks=None,
        full_suite_queue_status='stopped',latest_queue_closure=str(BASE/REL/'fresh_full_suite_queue016b_stopped_closure001.json'),
        do_not_modify_live_src=False,do_not_modify_training_entrypoint=False,do_not_modify_domain_admission_helper=False,
        latest_observed_utc=result['observed_utc'],latest_observation='016 stopped after330 updates; pure standing failed and actual peak50.97rad/s; no behavior or deployment qualification.')
    assert ledger_path.read_bytes()==before
    tmp=ledger_path.with_suffix('.close016.tmp');write(tmp,ledger);tmp.replace(ledger_path)
    print(json.dumps({k:result[k] for k in ('all_checks_pass','actual_consumed_updates','RSL_regular_updates',
        'peak_leg_speed_rad_s','leg_motor_samples_at_or_above_50_rad_s','total_actual_consumed_training_updates','pure_standing_windows')}))


if __name__=='__main__':main()
