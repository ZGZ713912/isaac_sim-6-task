"""Reconstruct frozen PPO eligibility for the actual018 peak witnesses."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
RUN = BASE / 'runs/fresh_unified_20261027/pilot018_velocity_quartic_physical_speed_mixed1536'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = REL / 'fresh018_peak_admission_read001.json'
    if (ROOT / output).exists() or (BASE / output).exists():
        raise ValueError('Preserve the original peak admission audit')
    source_path = ROOT / REL / 'fresh_unified018_domain_entry208_007.json'
    source = json.loads(source_path.read_text())
    initial_path = RUN / 'randomization_study.initial_samples.json'
    initial = json.loads(initial_path.read_text())['samples']
    task_path = RUN / 'task_contract.json'
    plan = json.loads(task_path.read_text())['settings']['domain_learning_admission']
    relative = 'scripts/domain_learning_admission_candidate.py'
    frozen = RUN / 'source_snapshot' / relative
    manifest = json.loads((RUN / 'source_hashes.json').read_text())
    assert sha(frozen) == manifest[relative] == sha(ROOT / relative)
    specification = importlib.util.spec_from_file_location('frozen018_admission_probe', frozen)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    admission = module.Admission(torch.tensor(initial['enabled'], dtype=torch.bool),
        warmup_updates=plan['warmup_updates'], ramp_updates=plan['ramp_updates'], seed=plan['priority_seed'])
    prefix = source['event_prefix']
    with open(prefix['path'], 'rb') as handle:
        raw = handle.read(prefix['bytes'])
    assert hashlib.sha256(raw).hexdigest() == prefix['sha256']
    with tempfile.TemporaryDirectory() as directory:
        event = Path(directory) / Path(prefix['path']).name
        event.write_bytes(raw)
        events = EventAccumulator(str(event), size_guidance={'scalars': 0}).Reload()
        values = {tag: {point.step: point.value for point in events.Scalars(tag)}
                  for tag in events.Tags()['scalars'] if tag.startswith(
                      ('PhysicalPeakWitness/', 'DomainAdmission/', 'FreshStandingFailure/'))}
    count = source['extrema']['complete_RSL_updates']
    rows = []
    for step in range(count):
        mask = admission.mask(step)
        assert float((mask & admission.enabled).sum()) == values['DomainAdmission/admitted_randomized_lanes'][step]
        phase = int(values['PhysicalPeakWitness/phase_code'][step])
        if phase not in (0, 1):
            continue
        witness = {tag.split('/', 1)[1]: data[step] for tag, data in values.items()
                   if tag.startswith('PhysicalPeakWitness/')}
        lane = int(witness['lane'])
        first_admitted = next(update + 1 for update in range(plan['warmup_updates'] + plan['ramp_updates'] + 1)
                              if admission.mask(update)[lane])
        rows.append(dict(update_one_based=step + 1, witness=witness,
            sampled_randomized_domain=bool(admission.enabled[lane]),
            admitted_for_PPO_and_normalization=bool(mask[lane]),
            admitted_randomized_count=int((mask & admission.enabled).sum()),
            first_update_for_this_lane_admission=first_admitted,
            sampled_motor_strength_policy_order=initial['motor_strength'][lane],
            phase_scope='Maximum of this rollout, not a count of all per-phase high-speed samples'))
    peak = source['extrema']['global_peak_witness']
    matched, = [row for row in rows if row['update_one_based'] == source['extrema']['peak_update_one_based']]
    assert matched['witness'] == peak
    assert matched['update_one_based'] == 206 and not matched['admitted_for_PPO_and_normalization']
    assert matched['first_update_for_this_lane_admission'] == 307
    failures = []
    for start, stop in ((0, 192), (160, 192)):
        metrics = {tag.split('/', 1)[1]: sum(data[step] for step in range(start, stop)) / (stop - start)
                   for tag, data in values.items() if tag.startswith('FreshStandingFailure/')}
        failures.append(dict(updates_one_based=[start + 1, stop], mean_events_per_policy_step=metrics,
            scope='All256 native upright-standing lanes, includes random lanes not admitted to PPO; tilt and height events can overlap'))
    result = dict(schema='fresh018-peak-admission-read-v1', observed_utc=datetime.now(timezone.utc).isoformat(),
        source_observation=str(source_path), source_sha256=sha(source_path), event_prefix=prefix,
        actual_initial_samples_sha256=sha(initial_path), frozen_task_sha256=sha(task_path),
        admission_source_sha256=manifest[relative], verified_RSL_updates=count,
        native_nonordinary_rollout_peak_witnesses=rows, foundation_failure_events=failures,
        global_peak_admitted=False, global_peak_witness=peak,
        interpretation='The206 active peak is real, but its lane was not yet used for PPO or normalization; it does not establish that quartic cost failed after learning this domain. Passive zero-torque peaks and active positive-work peaks both occur.',
        new_PPO=0, new_native_steps=0, optimizer_constructed=False, policy_changed=False,
        hardware_ready=False, producer_sha256=sha(Path(__file__)))
    for root in (ROOT, BASE):
        (root / output).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (BASE / REL / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(global_peak=matched, all_nonordinary_rollout_peaks=rows,
                          foundation_failure_events=failures)))


if __name__ == '__main__':
    main()
