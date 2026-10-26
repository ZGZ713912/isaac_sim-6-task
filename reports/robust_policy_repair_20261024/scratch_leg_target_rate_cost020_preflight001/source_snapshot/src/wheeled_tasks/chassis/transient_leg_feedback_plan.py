"""One causal leg-recurrence revision, retaining rapid learned corrections."""
from copy import deepcopy
from pathlib import Path

from wheeled_algo.transient_leg_feedback import SETTINGS
from .transient_dwell_plan import make_dwell_plan, validate_dwell_plan, EVIDENCE as DWELL_EVIDENCE

SCHEMA = 'flat-base-finetune-plan-v23'
EVIDENCE = 'reports/flat_transient_response_20261021/action_relaxation_native001/report.json'
DECISION = 'reports/flat_transient_response_20261021/action_relaxation_decision001.json'


def make_leg_feedback_plan(base, campaign, dwell_sha, evidence_sha, decision_sha):
    if campaign['num_envs'] != 512:
        raise ValueError('Leg feedback iteration retains its diagnosed 512 population')
    plan = make_dwell_plan(base, campaign, dwell_sha)
    plan.update(schema=SCHEMA, transient_leg_feedback=dict(actor=deepcopy(SETTINGS),
        diagnosis_report=EVIDENCE, diagnosis_report_sha256=evidence_sha,
        decision_file=DECISION, decision_sha256=decision_sha))
    return plan


def validate_leg_feedback_plan(root, plan):
    from .flat_base_continuation import digest, read_plan
    root = Path(root)
    settings = plan['transient_leg_feedback']
    if (settings['diagnosis_report_sha256'] != digest(root/EVIDENCE)
            or settings['decision_sha256'] != digest(root/DECISION)):
        raise ValueError('Parent-leg causal evidence changed')
    spec = plan['transient_campaign']
    base = read_plan(root, spec['base_plan_file'])
    validate_dwell_plan(root, make_dwell_plan(base, spec, digest(root/DWELL_EVIDENCE)))
    expected = make_leg_feedback_plan(base, spec, digest(root/DWELL_EVIDENCE),
                                      digest(root/EVIDENCE), digest(root/DECISION))
    if plan != expected:
        raise ValueError('Parent-leg feedback plan changed an undeclared setting')
    return dict(parent_leg_alpha=.25, learned_leg_and_wheel_corrections_direct=True,
                rewards_physics_clock_and_budget_unchanged=True)


def validate_leg_feedback_migration(info, old, new, sources):
    """Reject unrelated source mutations during this full-state continuation."""
    changed_fields = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    allowed_fields = {'schema', 'transient_campaign', 'transient_leg_feedback',
                      'first_block_updates', 'checkpoint_interval'}
    allowed_sources = {
        'scripts/train_flat_base.py', 'src/wheeled_algo/directional_motion.py',
        'src/wheeled_algo/directional_runtime.py', 'src/wheeled_algo/command_residual.py',
        'src/wheeled_algo/transient_leg_feedback.py',
        'src/wheeled_tasks/chassis/transient_leg_feedback_plan.py',
        'src/wheeled_tasks/chassis/transient_campaign_plan.py',
        'src/wheeled_tasks/chassis/flat_base_continuation.py',
        'src/wheeled_tasks/chassis/flat_clock_transition.py'}
    changed_sources = {name for name in info['source_sha256'].keys() | sources.keys()
                       if info['source_sha256'].get(name) != sources.get(name)}
    if (info['num_envs'] != 512 or changed_fields - allowed_fields
            or changed_sources - allowed_sources
            or old.get('transient_leg_feedback') not in (None, new['transient_leg_feedback'])):
        raise ValueError('Leg feedback migration changed unrelated sources, settings or population')
    return dict(changed_sources=sorted(changed_sources), changed_fields=sorted(changed_fields),
                actor_critic_Adam_RNG_counters='must restore exactly; declared mean function change',
                additional_critic_warmup=0)
