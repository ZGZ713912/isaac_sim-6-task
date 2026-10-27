#!/usr/bin/env python3
"""Seal one exact completed fresh learner, then serially measure its full suite.

No training, continuation, optimizer, action override or robot motion is started.
Requested evaluations use every lane randomized at the unchanged requested
widths; training keeps its original25 percent nominal rehearsal population.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    'scripts/queue_fresh_full_suite.py', 'scripts/evaluate_fresh_full_suite.py',
    'scripts/wheel_leg_full_suite.py', 'scripts/review_standing_acceptance.py',
    'scripts/record_scratch_physics.py', 'scripts/scratch_domain_evaluation.py',
    'scripts/seal_scratch_standing.py', 'src/wheeled_tasks/chassis/randomization_study.py',
    'src/wheeled_algo/scratch_standing.py', 'contracts/wheel_leg_full_suite_v1.json',
    'contracts/wheel_leg_randomization_study_v2.json', 'contracts/wheel_leg_acceptance_requirements_v2.json')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def ticks(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    except FileNotFoundError:
        return None


def terminal_checkpoint(run, identity, updates):
    if ticks(identity['pid']) == identity['start_ticks']:
        raise ValueError('The exact learner has not exited')
    p = read(run/'progress.json')
    pt = run/'model_final.pt'
    if (p.get('training_entry') != 'fresh-unified-v1' or p['pid'] != identity['pid']
            or p['start_ticks'] != identity['start_ticks'] or p['status'] != 'completed_unqualified'
            or p['requested_updates'] != updates or p['completed_updates'] != updates
            or p['optimizer_attempts'] != updates or p['optimizer_update_in_flight']
            or p['checkpoint_reads'] != [] or not pt.is_file()
            or Path(p['last_checkpoint']['path']).resolve() != pt.resolve()
            or p['last_checkpoint']['updates'] != updates or p['last_checkpoint']['sha256'] != sha(pt)
            or p.get('evaluation',{}).get('status') != 'not_performed'):
        raise ValueError('Need the complete fresh terminal checkpoint; never restart failed/stopped training here')
    return pt, p


def all_randomized_study(original):
    if original.get('schema') != 'wheel-leg-randomization-study-v2' or original.get('nominal_fraction') != .25:
        raise ValueError('Need the unchanged requested training study with25 percent nominal rehearsal')
    result = deepcopy(original)
    result['nominal_fraction'] = 0.
    return result


def freeze(output):
    data = {name:(ROOT/name).read_bytes() for name in SOURCE_FILES}
    for name, value in data.items():
        target = output/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    return {name:hashlib.sha256(value).hexdigest() for name,value in data.items()}


def verify_native(folder, checkpoint_sha, profile, seed, capability_ids):
    receipt, result, cases = [read(folder/name) for name in ('receipt.json','result.json','cases.json')]
    samples = receipt['actual_domain_samples']['samples']
    if (receipt['status'] != 'completed_unqualified' or receipt['checkpoint_sha256'] != checkpoint_sha
            or result['checkpoint_sha256'] != checkpoint_sha or receipt['seed'] != seed
            or receipt['native_policy_steps'] != 6000 or receipt['new_PPO'] != 0
            or receipt['optimizer_constructed'] or result['new_PPO'] != 0 or result['optimizer_constructed']
            or receipt['case_count'] != 198 or len(result['cases']) != 198
            or len(samples['enabled']) != 198
            or any(value is not (profile=='requested') for value in samples['enabled'])
            or set(row['capability'] for row in result['capabilities']) != capability_ids
            or sha(folder/'result.json') != receipt['result_sha256']
            or sha(folder/'native_trace.npz') != result['native_trace_sha256']):
        raise ValueError('Full suite policy/domain/source/protocol evidence differs from its declared run')
    return receipt, result, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--training-identity', type=Path, required=True)
    parser.add_argument('--expected-updates', type=int, default=1536)
    parser.add_argument('--seeds', type=int, nargs=2, default=[271301,271401])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.training_run.resolve(), args.output.resolve()
    identity = read(args.training_identity)
    if (Path(identity['run']).resolve() != run or identity['requested_updates'] != args.expected_updates
            or len(set(args.seeds)) != 2):
        raise ValueError('Pin the exact fresh learner and two distinct independent evaluation seeds')
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema='fresh-full-suite-serial-queue-v1', status='starting', pid=os.getpid(),
        start_ticks=ticks(os.getpid()), training_run=str(run), training_identity=identity,
        expected_updates=args.expected_updates, seeds=args.seeds, active_child=None, evaluations=[],
        new_PPO=0, optimizer_constructed=False, training_restart_allowed=False,
        overall_policy_qualified=False, hardware_ready=False)
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save():
        receipt['observed_utc'] = datetime.now(timezone.utc).isoformat()
        write(output/'receipt.json', receipt)

    def check_sources():
        for name, value in receipt['source_sha256'].items():
            if sha(ROOT/name) != value or sha(output/'source_snapshot'/name) != value:
                raise ValueError('Frozen full-suite queue/evaluator source changed: '+name)

    def execute(argv, label):
        check_sources()
        with (output/(label+'.log')).open('xb') as stream:
            child = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            receipt['active_child'] = dict(pid=child.pid,start_ticks=ticks(child.pid),label=label,argv=argv)
            save()
            while child.poll() is None:
                if stop:
                    if ticks(child.pid) == receipt['active_child']['start_ticks']:
                        child.terminate()
                    child.wait()
                    raise InterruptedError('Only the queue/evaluation child stopped; learner was never signaled')
                time.sleep(1.)
            receipt.setdefault('closed_children',[]).append(dict(receipt['active_child'],exit_code=child.returncode))
            receipt['active_child'] = None
            save()
            if child.returncode:
                raise RuntimeError('Read-only child failed: '+label)
        check_sources()

    try:
        receipt['source_sha256'] = freeze(output)
        study = ROOT/'contracts/wheel_leg_randomization_study_v2.json'
        derived = output/'requested_all_randomized_study.json'
        write(derived, all_randomized_study(read(study)))
        receipt['evaluation_sampling'] = dict(parent_file=str(study),parent_sha256=sha(study),
            derived_file=str(derived),derived_sha256=sha(derived),only_changed_field='nominal_fraction:0.25 to0',
            requested_widths_unchanged=True,training_study_unchanged=True,
            meaning='All198 requested evaluation lanes randomized; nominal runs separately disable domains')
        if ticks(identity['pid']) == identity['start_ticks']:
            cmd = Path(f"/proc/{identity['pid']}/cmdline").read_bytes()
            if b'scripts/train_fresh_unified.py' not in cmd or str(run).encode() not in cmd:
                raise ValueError('The pinned process belongs to another task')
        receipt['status'] = 'waiting_for_exact_training_process_exit'
        save()
        while ticks(identity['pid']) == identity['start_ticks']:
            if stop:
                raise InterruptedError('Queue stopped; learner was never signaled')
            p = read(run/'progress.json')
            receipt.update(observed_training_updates=p['completed_updates'],observed_training_status=p['status'])
            save()
            time.sleep(5.)
        checkpoint, progress = terminal_checkpoint(run, identity, args.expected_updates)
        gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        if str(identity['pid']) in gpu.splitlines():
            raise ValueError('Original learner still occupies its GPU process slot')
        receipt.update(status='sealing_exact_terminal_checkpoint',checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint),training_process_and_GPU_exited=True)
        save()
        seal_file = output/'training_closed_seal.json'
        execute([sys.executable,str(ROOT/'scripts/seal_scratch_standing.py'),'--run',str(run),
            '--identity',str(args.training_identity.resolve()),'--output',str(seal_file),'--require-current-source'],'seal')
        if read(seal_file)['all_checks_pass'] is not True:
            raise ValueError('Fresh terminal integrity seal failed')
        receipt['status'] = 'serial_cold_full_suite_evaluation'
        capability_ids = {row['id'] for row in read(ROOT/'contracts/wheel_leg_acceptance_requirements_v2.json')['capabilities']}
        physics = None
        seed_cases = {}
        requested_draws = []
        for seed in args.seeds:
            for profile in ('nominal','requested'):
                label = f'{profile}_seed{seed}'
                folder = output/label
                if sha(derived) != receipt['evaluation_sampling']['derived_sha256']:
                    raise ValueError('The declared evaluation-only sampling population changed')
                execute([sys.executable,'-u',str(ROOT/'scripts/evaluate_fresh_full_suite.py'),
                    '--checkpoint',str(checkpoint),'--output',str(folder),'--seed',str(seed),
                    '--randomization-study',str(derived),'--randomization-profile',profile],label)
                native, result, cases = verify_native(folder,receipt['checkpoint_sha256'],profile,seed,capability_ids)
                if physics is not None and native['physics_runtime'] != physics:
                    raise ValueError('Actual physics clocks, controllers or solver changed between cold tests')
                physics = native['physics_runtime']
                if seed in seed_cases and cases != seed_cases[seed]:
                    raise ValueError('Same-seed nominal/requested cases are not paired')
                seed_cases[seed] = cases
                if profile == 'requested':
                    requested_draws.append(native['actual_domain_samples'])
                receipt['evaluations'].append(dict(profile=profile,seed=seed,folder=str(folder),
                    receipt_sha256=sha(folder/'receipt.json'),result_sha256=sha(folder/'result.json'),
                    capabilities=result['capabilities'],same_exact_checkpoint=True,case_count=198,
                    actual_randomized_lanes=sum(native['actual_domain_samples']['samples']['enabled'])))
                save()
        if len(requested_draws) != 2 or requested_draws[0] == requested_draws[1]:
            raise ValueError('Distinct seed labels did not produce distinct actual requested-domain draws')
        receipt.update(status='completed_unqualified',recorded_case_count=792,
            independent_requested_draws_verified=True,
            note='All native outcomes retained. Full13 same-policy evidence still requires review; no long PPO or hardware motion follows automatically.')
    except InterruptedError as error:
        receipt.update(status='stopped',error=str(error))
    except BaseException:
        receipt.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
