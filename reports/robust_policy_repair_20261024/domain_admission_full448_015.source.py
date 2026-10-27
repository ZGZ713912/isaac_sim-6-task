#!/usr/bin/env python3
"""Read separate native population metrics from an immutable RSL event prefix.

This observer never changes or starts a learner, simulator or TensorBoard
server. It verifies the admission clock against the frozen plan and the
actual startup population, then reports complete, non-overlapping windows.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('nominal', 'randomized', 'admitted')
METRICS = ('sample_count', 'episode_end_events', 'planar_speed_rms_m_s', 'tilt_mean_deg', 'height_mae_m')
COUNTS = ('admitted_nominal_lanes', 'admitted_randomized_lanes', 'population_fraction', 'completed_actor_updates')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observe(run, window):
    if window < 1:
        raise ValueError('Window must contain at least one complete fresh update')
    progress = read(run / 'progress.json')
    task = read(run / 'task_contract.json')['settings']
    plan = task['domain_learning_admission']
    if not plan or progress['checkpoint_reads'] or progress['domain_learning_admission']['training_adopted'] is not True:
        raise ValueError('Need a fresh run using the declared admission curriculum')
    samples = read(run / 'randomization_study.initial_samples.json')['samples']['enabled']
    randomized, count = sum(samples), len(samples)
    nominal = count - randomized
    manifest = read(run / 'source_hashes.json')
    for name, digest in manifest.items():
        if sha(run / 'source_snapshot' / name) != digest:
            raise ValueError('Frozen training source changed: ' + name)
    current_mismatches = [name for name, digest in manifest.items() if sha(ROOT / name) != digest]
    try:
        live = int(Path(f"/proc/{progress['pid']}/stat").read_text().rsplit(')', 1)[1].split()[19]) == progress['start_ticks']
    except FileNotFoundError:
        live = False
    if live and current_mismatches:
        raise ValueError('Active learner checkout source changed')
    event, = list(run.glob('events.out.tfevents.*'))
    raw = event.read_bytes()
    tags = ['DomainAdmission/' + group + '/' + metric for group in GROUPS for metric in METRICS]
    tags += ['DomainAdmission/' + metric for metric in COUNTS]
    with tempfile.TemporaryDirectory() as temporary:
        snapshot = Path(temporary) / event.name
        snapshot.write_bytes(raw)
        accumulator = EventAccumulator(str(snapshot), size_guidance={'scalars': 0}).Reload()
        data = {}
        for tag in tags:
            rows = accumulator.Scalars(tag)
            values = {row.step: row.value for row in rows}
            if len(values) != len(rows) or not all(math.isfinite(value) for value in values.values()):
                raise ValueError('Nonfinite or duplicate native scalar: ' + tag)
            data[tag] = values
    complete = 0
    while all(complete in data[tag] for tag in tags):
        complete += 1
    if not complete or complete > progress['completed_updates']:
        raise ValueError('No complete native update, or event clock exceeds committed learning')
    for step in range(complete):
        elapsed = min(plan['ramp_updates'], max(0, step - plan['warmup_updates']))
        random_count = randomized * elapsed // plan['ramp_updates']
        expected = dict(admitted_nominal_lanes=nominal, admitted_randomized_lanes=random_count,
            population_fraction=(nominal + random_count) / count, completed_actor_updates=step)
        if not all(math.isclose(data['DomainAdmission/' + name][step], value, rel_tol=1e-6, abs_tol=1e-7)
                   for name, value in expected.items()):
            raise ValueError('Actual RSL admission schedule differs at update ' + str(step + 1))
    windows = []
    for first in range(0, complete // window * window, window):
        stop = first + window
        groups = {group: {metric: sum(data['DomainAdmission/' + group + '/' + metric][i]
            for i in range(first, stop)) / window for metric in METRICS} for group in GROUPS}
        populations = {metric: [data['DomainAdmission/' + metric][first], data['DomainAdmission/' + metric][stop - 1]]
            for metric in ('admitted_nominal_lanes', 'admitted_randomized_lanes')}
        windows.append(dict(fresh_PPO_update_numbers_inclusive=[first + 1, stop], groups=groups,
            admitted_lane_count_at_first_and_last_update=populations))
    if not windows:
        raise ValueError('Wait for a complete population window in the same original run')
    return dict(schema='domain-admission-RSL-native-progress-v1', observed_utc=datetime.now(timezone.utc).isoformat(),
        run=str(run), pid=progress['pid'], start_ticks=progress['start_ticks'], original_process_live=live,
        progress_status=progress['status'], progress_completed_updates=progress['completed_updates'],
        common_complete_RSL_updates=complete, requested_updates=progress['requested_updates'],
        actual_startup_nominal_lanes=nominal, actual_startup_randomized_lanes=randomized,
        frozen_source_count=len(manifest), current_source_mismatches=current_mismatches,
        RSL_schedule_matches_every_complete_update=True, plan=plan, window_updates=window, windows=windows,
        latest_complete_native_population=dict(fresh_PPO_update_number=complete,
            **{metric: data['DomainAdmission/'+metric][complete-1] for metric in COUNTS}),
        event_prefix=dict(path=str(event), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
        new_PPO_by_reader=0, native_steps_by_reader=0, optimizer_constructed=False,
        behavior_qualified=False, hardware_ready=False,
        scope='Pre-reset true states under stochastic training and failure autoresets. These means are not fixed standing, recovery or hardware passes; nominal lanes retain requested ground friction.',
        reader_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--window', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.source.py').exists():
        raise ValueError('Preserve earlier event observations and their readers')
    result = observe(args.run.resolve(), args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.source.py').write_bytes(Path(__file__).read_bytes())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('progress_completed_updates', 'common_complete_RSL_updates', 'latest_complete_native_population', 'windows')}, ensure_ascii=False))
