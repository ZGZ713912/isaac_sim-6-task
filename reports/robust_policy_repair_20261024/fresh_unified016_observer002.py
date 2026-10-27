"""Read the original learner and queue without advancing or changing either."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
RUN = BASE/'runs/fresh_unified_20261027/pilot016_scoped_failure_requested_mixed1536'
QUEUE = BASE/REL/'fresh_full_suite_queue016b'
sys.path.insert(0, str(ROOT/'scripts'))
from read_domain_admission_progress import observe
from seal_scratch_standing import original_process_live
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-name', required=True)
    parser.add_argument('--window', type=int, default=32)
    args = parser.parse_args()
    target = ROOT/REL/(args.output_name+'.json')
    if target.exists():
        raise ValueError('Preserve earlier native observations')
    population = observe(RUN, args.window)
    progress, queue = read(RUN/'progress.json'), read(QUEUE/'receipt.json')
    sources = read(RUN/'source_hashes.json')
    identity = read(BASE/REL/'fresh_unified016_launch_identity001.json')
    event, = RUN.glob('events.out.tfevents.*')
    raw = event.read_bytes()
    with tempfile.TemporaryDirectory() as temporary:
        frozen = Path(temporary)/event.name
        frozen.write_bytes(raw)
        events = EventAccumulator(str(frozen), size_guidance={'scalars': 0}).Reload()
        tags = [tag for tag in events.Tags()['scalars']
            if tag.startswith(('ScratchStandingExtrema/', 'FreshStandingFailure/', 'FreshTasks/'))]
        tags += ['Loss/value_function', 'Loss/surrogate', 'Loss/entropy']
        values = {tag: {row.step: row.value for row in events.Scalars(tag)} for tag in tags
            if tag in events.Tags()['scalars']}
    count = population['common_complete_RSL_updates']
    peak = values['ScratchStandingExtrema/peak_leg_speed_rad_s']
    high = values['ScratchStandingExtrema/leg_motor_samples_at_or_above_50_rad_s']
    failure = values['FreshStandingFailure/episode_events']
    if not all(i in peak and i in high and i in failure for i in range(count)):
        raise ValueError('Wait for aligned physical extrema and failure event records')
    live = original_process_live(identity)
    checks = dict(exact_training_identity=(progress['pid'], progress['start_ticks']) == (identity['pid'], identity['start_ticks']),
        original_training_still_live=live, source273_current_and_archived=len(sources) == 273 and all(
            sha(ROOT/name) == value == sha(RUN/'source_snapshot'/name) for name, value in sources.items()),
        queue12_current_and_archived=len(queue['source_sha256']) == 12 and all(
            sha(ROOT/name) == value == sha(QUEUE/'source_snapshot'/name) for name, value in queue['source_sha256'].items()),
        queue_pins_same_training_identity=queue['training_identity']['pid'] == identity['pid']
            and queue['training_identity']['start_ticks'] == identity['start_ticks'],
        no_old_checkpoint_read=progress['checkpoint_reads'] == [],
        actual_GPU_owner=str(identity['pid']) in subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines())
    assert all(checks.values()), checks
    result = dict(schema='fresh-unified016-live-observation-v1', observed_utc=datetime.now(timezone.utc).isoformat(),
        checks=checks, all_checks_pass=True, progress_completed_updates=progress['completed_updates'],
        population=population, exact_process=dict(pid=identity['pid'], start_ticks=identity['start_ticks']),
        queue=dict(pid=queue['pid'], start_ticks=queue['start_ticks'], status=queue['status'], active_child=queue['active_child']),
        extrema=dict(complete_RSL_updates=count, peak_leg_speed_rad_s=max(peak[i] for i in range(count)),
            leg_motor_samples_at_or_above_50_rad_s=sum(high[i] for i in range(count)),
            samples_per_update=128*5*256*4),
        failure_events=dict(all_native_active_lanes_first_updates=sum(failure[i]*128 for i in range(count)),
            last8_mean_per_policy_interval=sum(failure[i] for i in range(max(0, count-8), count))/min(8, count),
            includes_not_yet_PPO_admitted_randomized_lanes=True),
        task_latest={tag: value[count-1] for tag, value in values.items()
            if tag.startswith('FreshTasks/') and count-1 in value},
        supplemental_event_prefix=dict(path=str(event), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
        prior_fully_completed_fresh_updates=4225, prior_failed_updates=1, prior_stopped_updates=56,
        prior_actual_consumed_updates=4282, total_actual_consumed_updates_observed=4282+progress['completed_updates'],
        new_PPO_by_reader=0, native_steps_by_reader=0, optimizer_constructed=False,
        qualified=False, hardware_ready=False, producer_sha256=sha(Path(__file__)))
    for directory in (ROOT/REL, BASE/REL):
        write(directory/target.name, result)
    (BASE/REL/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    ledger_file = BASE/REL/'scratch_campaign001.json'
    before = ledger_file.read_bytes(); ledger = json.loads(before)
    assert (ledger['active_pid'], ledger['active_start_ticks']) == (identity['pid'], identity['start_ticks'])
    observed = max(progress['completed_updates'], ledger['active_completed_updates_observed'])
    ledger.update(active_completed_updates_observed=observed, completed_updates_observed=observed,
        total_actual_consumed_training_updates=4282+observed, latest_observation_file=str(BASE/REL/target.name),
        latest_observed_utc=result['observed_utc'], full_suite_queue_status=queue['status'])
    assert ledger_file.read_bytes() == before
    temporary = ledger_file.with_suffix('.observer016.tmp'); write(temporary, ledger); temporary.replace(ledger_file)
    print(json.dumps(dict(completed_updates=progress['completed_updates'], checks_pass=True,
        latest_window=population['windows'][-1], extrema=result['extrema'], queue_status=queue['status'])))


if __name__ == '__main__':
    main()
