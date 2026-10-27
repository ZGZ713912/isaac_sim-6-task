"""Keep the fresh013 short standing evidence separate from full qualification."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

from rsl_rl.utils.logger import Logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
QUEUE = BASE/REL/'requested_v2_evaluation_queue005'
NAME = 'fresh_capability_scope_read001'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = BASE/REL/(NAME+'.json')
    event_dir = BASE/REL/(NAME+'_events')
    if output.exists() or event_dir.exists() or (ROOT/REL/output.name).exists():
        raise ValueError('Preserve earlier scope measurements')
    comparison_path = ROOT/REL/'domain_admission_cold_comparison001.json'
    comparison = read(comparison_path)
    queue = read(QUEUE/'receipt.json')
    requirements = ROOT/'contracts/wheel_leg_acceptance_requirements_v2.json'
    if not comparison['all_checks_pass'] or sha(QUEUE/'receipt.json') != comparison['queue_sha256']:
        raise ValueError('Need the original completed, authenticated cold comparison')
    profiles = []
    values = {}
    for saved in queue['profiles']:
        if saved['profile'] not in ('nominal', 'requested'):
            continue
        run = Path(saved['run'])
        receipt = read(run/'receipt.json')
        review = read(saved['review'])
        result = read(run/'evaluation/result.json')
        if not (sha(run/'receipt.json') == saved['receipt_sha256']
                and sha(saved['review']) == saved['review_sha256']
                and sha(run/'evaluation/result.json') == review['original_result_sha256']
                and sha(run/'evaluation/native_trace.npz') == review['native_trace_sha256'] == result['trace_sha256']
                and sha(requirements) == review['requirements_sha256']
                and receipt['checkpoint_sha256'] == comparison['checkpoint_sha256']
                and receipt['completed_updates'] == review['completed_updates'] == 512
                and receipt['new_PPO'] == 0 and receipt['optimizer_constructed'] is False
                and receipt['evaluation_duration_s'] == 8.):
            raise ValueError('The original standing evidence changed')
        enabled = result['actual_domain_samples']['enabled']
        if len(enabled) != len(result['cases']) or len(review['cases']) != len(enabled):
            raise ValueError('Domain labels must describe the actual original cars')
        rows = []
        for i, (case, old, randomized) in enumerate(zip(result['cases'], review['cases'], enabled)):
            if type(randomized) is not bool or old['case_index'] != i or old['height_m'] != case['height_m']:
                raise ValueError('Original case order and actual domain labels differ')
            xy_limit = .1 if old['height_m'] == .42 else .05
            rows.append(dict(case_index=i, height_m=old['height_m'], pitch_deg=old['pitch_deg'],
                actual_randomized=randomized, basic8s_pass=old['standing_pass_v2'],
                max_xy_excursion_m=old['max_xy_excursion_m'], inherited_xy_limit_m=xy_limit,
                basic8s_and_inherited_xy_pass=bool(old['standing_pass_v2'] and old['max_xy_excursion_m'] <= xy_limit),
                retained_loss_ms=old['support']['retained_loss_ms'],
                longest_raw_loss_ms=old['support']['longest_raw_loss_ms']))
        groups = []
        for height in (.23, .305, .42):
            for subset in ('all', 'actual_randomized', 'nominal_mask'):
                selected = [r for r in rows if r['height_m'] == height
                    and (subset == 'all' or r['actual_randomized'] == (subset == 'actual_randomized'))]
                group = dict(height_m=height, subset=subset, case_count=len(selected),
                    basic8s_pass_count=sum(r['basic8s_pass'] for r in selected),
                    basic8s_and_inherited_xy_pass_count=sum(r['basic8s_and_inherited_xy_pass'] for r in selected),
                    retained_loss_ms=sum(r['retained_loss_ms'] for r in selected),
                    longest_raw_loss_ms=max((r['longest_raw_loss_ms'] for r in selected), default=None),
                    max_xy_excursion_m=max((r['max_xy_excursion_m'] for r in selected), default=None),
                    tested_seconds=8., full30s_qualification_measured=False)
                groups.append(group)
                for key, value in group.items():
                    if key not in ('height_m', 'subset') and value is not None:
                        values[f'CapabilityScope/Fresh013/{saved["profile"]}/{height:.3f}m/{subset}/{key}'] = float(value)
        profiles.append(dict(profile=saved['profile'], run=str(run),
            receipt_sha256=sha(run/'receipt.json'), review_sha256=sha(saved['review']),
            result_sha256=sha(run/'evaluation/result.json'), native_trace_sha256=result['trace_sha256'],
            actual_domain_samples_source='authenticated original result.actual_domain_samples.enabled',
            groups=groups, cases=rows))
    if [r['profile'] for r in profiles] != ['nominal', 'requested']:
        raise ValueError('Need both completed cold profiles of the same checkpoint')
    nominal = Path(profiles[0]['run'])
    checkpoint = Path(read(nominal/'receipt.json')['checkpoint'])
    if sha(checkpoint) != comparison['checkpoint_sha256']:
        raise ValueError('Original final policy changed')
    logger = Logger(log_dir=str(event_dir), cfg=read(checkpoint.parent/'agent_config.json'),
        env_cfg=read(nominal/'config/contract.json'), num_envs=1, is_distributed=False,
        gpu_world_size=1, gpu_global_rank=0, device='cpu')
    logger.init_logging_writer()
    try:
        for tag, value in values.items():
            logger.writer.add_scalar(tag, value, 512)
        logger.writer.flush()
    finally:
        logger.writer.close()
    events = EventAccumulator(str(event_dir), size_guidance={'scalars': 0}).Reload()
    if set(events.Tags()['scalars']) != set(values):
        raise ValueError('RSL tags differ from the authenticated scope measurements')
    for tag, value in values.items():
        rows = events.Scalars(tag)
        if len(rows) != 1 or rows[0].step != 512 or not math.isclose(rows[0].value, value, rel_tol=2e-6, abs_tol=2e-8):
            raise ValueError('RSL readback failed: '+tag)
    result = dict(schema='fresh013-standing-capability-scope-v1',
        observed_utc=datetime.now(timezone.utc).isoformat(), producer_sha256=sha(__file__),
        checkpoint_sha256=comparison['checkpoint_sha256'], completed_fresh_updates=512,
        comparison_sha256=sha(comparison_path), requirements_sha256=sha(requirements), profiles=profiles,
        step_definition='original fresh013 completed PPO update512; no learning by this reader',
        RSL_events=dict(directory=str(event_dir), verified_scalar_count=len(values), all_values_read_back=True,
            files_sha256={p.name:sha(p) for p in event_dir.glob('events.out.tfevents.*')}),
        new_PPO=0, new_native_steps=0, optimizer_constructed=False, original_artifacts_modified=False,
        full13_qualified=False, hardware_ready=False,
        interpretation='Legacy basic8s pass excludes inherited XY limits and is not a full30s standing qualification. The requested middle group contains six actually randomized cars and one nominal-mask car; ground friction is independently randomized. No different generations or domains are combined into one qualifying policy.')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    shutil.copyfile(output, ROOT/REL/output.name)
    shutil.copyfile(Path(__file__), BASE/REL/Path(__file__).name)
    print(json.dumps(dict(RSL_scalars=len(values), middle=[dict(profile=r['profile'], groups=[g for g in r['groups'] if g['height_m']==.305]) for r in profiles])))


if __name__ == '__main__':
    main()
