"""Expose full-speed braking states after the first native response diagnosis."""
from pathlib import Path

from .transient_campaign_plan import make_campaign_plan, validate_campaign_plan

SCHEMA = 'flat-base-finetune-plan-v22'
EVIDENCE = 'reports/flat_transient_response_20261021/paired_native001/report.json'


def make_dwell_plan(base, campaign, evidence_sha256):
    plan = make_campaign_plan(base, campaign)
    plan['schema'] = SCHEMA
    plan['transient_dwell'] = dict(version='full-speed-braking-exposure-v1',
        phase_seconds=8., complete_cycle_seconds=32.,
        diagnosis_report=EVIDENCE, diagnosis_report_sha256=evidence_sha256,
        rationale='The native high-height baseline needs about 6s to reach 3m/s; 2.5s training phases omit full-speed braking')
    plan['episode_seconds'] = 32.
    # Retain the old horizons for every ordinary retention group.
    plan['group_episode_seconds'] = {g['name']: base.get('group_episode_seconds', {}).get(
        g['name'], base['episode_seconds']) for g in plan['train_groups']}
    for group in plan['train_groups']:
        skill = group['skill']
        if skill['kind'] == 'start_stop':
            skill['segment_seconds'] = 8.
            plan['group_episode_seconds'][group['name']] = 32.
        elif 'yaw_start_stop_segment_seconds' in skill:
            skill['yaw_start_stop_segment_seconds'] = 8.
            plan['group_episode_seconds'][group['name']] = 32.
    return plan


def validate_dwell_plan(root, plan):
    from .flat_base_continuation import digest, read_plan
    root = Path(root)
    evidence = root/EVIDENCE
    if digest(evidence) != plan['transient_dwell']['diagnosis_report_sha256']:
        raise ValueError('Full-speed dwell diagnosis changed')
    campaign = plan['transient_campaign']
    base = read_plan(root, campaign['base_plan_file'])
    # Validate campaign lineage, exact population, consumption and admission
    # independently before checking the sole additional course modification.
    validate_campaign_plan(root, make_campaign_plan(base, campaign))
    if plan != make_dwell_plan(base, campaign, digest(evidence)):
        raise ValueError('Full-speed braking exposure changed an undeclared setting')
    return dict(phase_seconds=8., cycle_seconds=32., other_group_horizons_preserved=True)
