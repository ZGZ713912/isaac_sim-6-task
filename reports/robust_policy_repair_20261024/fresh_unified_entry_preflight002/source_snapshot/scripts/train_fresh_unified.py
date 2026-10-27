#!/usr/bin/env python3
"""Fresh mixed-task PPO. No checkpoint, teacher or continuation is accepted.

This entry is inactive until the declared native runtime and reward patches
have been applied and verified after the previous learner's integrity seal.
Its preflight writes configuration only; behavior requires native evaluation.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import train_scratch_standing as base
from train_scratch_standing import digest, write_json
from unified_training_cases_candidate import TrainingCases

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/'reports/robust_policy_repair_20261024/three_height_static_prior002.result.json'
NATIVE_RECEIPT = ROOT/'reports/robust_policy_repair_20261024/unified_training_runtime_candidate001/receipt.json'
PHASE_RECEIPT = ROOT/'reports/robust_policy_repair_20261024/standing_phase_candidate001/receipt.json'
EXTRA_SOURCES = ('scripts/train_fresh_unified.py', 'scripts/unified_training_cases_candidate.py',
    'scripts/unified_training_runtime_candidate.py', 'scripts/height_initialization_candidate.py',
    'scripts/record_scratch_physics.py')


def scratch_arguments(args):
    result = deepcopy(args)
    result.all_heights_after = args.foundation_updates
    result.precise_objective = True
    result.standing_reward_shape = 'gaussian'
    result.standing_failures = False
    result.wheel_velocity_increment = True
    result.separate_leg_velocity_feedback = True
    result.static_load_initialization = True
    result.contact_break_cost = None
    result.leg_target_rate_cost = None
    result.normalize_observations = True
    result.desync_episodes = False
    return result


def height_initialization(cfg, task_contract):
    prior = json.loads(PRIOR.read_text())
    if (prior['bank_sha256'] != task_contract['bank_sha256']
            or prior['current_manifest_sha256'] != cfg['asset_manifest_sha256']
            or not prior['all_three_reset_geometries_match']):
        raise ValueError('Three-height mechanical initialization differs from this bank/asset')
    coefficients = prior['candidate']['coefficient_rows_by_basis_policy_legs']
    middle = task_contract['static_load_initialization']['biased_policy_raw']
    if len(coefficients) != 3 or any(len(row) != 4 for row in coefficients):
        raise ValueError('Need the declared trainable constant/linear/quadratic four-leg coefficients')
    if any(abs(a-b) > 2e-6 for a, b in zip(coefficients[0], middle[:4], strict=True)):
        raise ValueError('Height constant and authenticated nominal load bias disagree')
    import math
    if not all(math.isfinite(value) for row in coefficients for value in row):
        raise ValueError('Nonfinite height initialization')
    return dict(schema='fresh-trainable-height-initialization-v1', prior_file=str(PRIOR),
        prior_sha256=digest(PRIOR), height_coefficients=coefficients[1:],
        constant_leg_bias=coefficients[0], source='declared mechanism static load and validated reset geometry; no policy',
        dynamic_or_randomized_validation_claimed=False, all_coefficients_trainable=True)


def prepare(args):
    common = scratch_arguments(args)
    cfg, task = base.prepare(common)
    cfg.update(episode_seconds=30., record_diagnostics=True,
        flat_half_length_m=64., flat_floor_width_m=128., curriculum='fresh_unified_mixed_tasks')
    task.update(version=task['version']+'-fresh-unified-mixed-tasks-v1',
        episode_seconds=30., quiet_training=False, boundary_xy_m=[60., 56.],
        hold_motion_during_recovery=True, origin_updates=0)
    bank = json.loads(args.bank.read_text())
    catalogue = TrainingCases(bank, args.num_envs, args.training_case_seed,
        foundation_updates=args.foundation_updates, full_tasks_at=args.full_tasks_at)
    task['fresh_unified_training'] = dict(schema='fresh-unified-mixed-task-training-v1',
        **{key: value for key, value in catalogue.summary(0).items() if key != 'schema'},
        phase_reward_required='pre-interval-and-pre-reset-recovery-v1',
        native_runtime_required='case-sequences-and-timeouts-v1',
        final_stage=catalogue.summary(args.full_tasks_at),
        no_mid_episode_physical_state_writes=True, actual_episode_clock=True)
    design = json.loads((args.output/'config/design.json').read_text())
    row = design['stages']['speed']['roles']['train']
    for key in ('episode_seconds', 'record_diagnostics', 'flat_half_length_m', 'flat_floor_width_m', 'curriculum'):
        if key in row['contract']:
            row['contract'][key] = deepcopy(cfg[key])
    design['scope'] = 'fresh_unified_mixed_task_mechanism_construction'
    design['review_basis']['fresh_task_catalogue'] = task['fresh_unified_training']
    write_json(args.output/'config/design.json', design)
    cfg['design_preflight']['sha256'] = digest(args.output/'config/design.json')
    write_json(args.output/'config/contract.json', cfg)
    path = args.output/'task_contract.json'
    contract = json.loads(path.read_text())
    contract.update(settings=task, mechanism_config_sha256=digest(args.output/'config/contract.json'),
        curriculum=deepcopy(task['fresh_unified_training']),
        current_scope='fresh full-task training; native full13 capabilities and hardware remain unqualified',
        standing_failure_resets=False, desynchronized_initial_episode_lengths=False,
        terminal_evaluation='separate cold full suite; no terminal standing-only qualification')
    contract['trainable_height_initialization'] = height_initialization(cfg, contract)
    (args.output/'height_initialization_reference.json').write_bytes(PRIOR.read_bytes())
    write_json(path, contract)
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.isolated_origins import validate_isolated_origins
    from wheeled_tasks.chassis.getup_ground import flat_patch
    control = load_chassis_control(ROOT/cfg['control_math_source'])
    check = design_preflight(cfg, control, ROOT, stage='speed', role='train', seed=args.seed,
        num_envs=args.num_envs, block_index=0)
    validate_isolated_origins(cfg, args.num_envs)
    points, indices = flat_patch(cfg['flat_half_length_m'], cfg['flat_floor_width_m'])
    write_json(args.output/'mechanism_preflight.json', check)
    write_json(args.output/'floor_geometry.json', dict(points=len(points), triangles=len(indices)//3,
        half_length_m=64., width_m=128., native_simulation=False))
    return cfg, task, contract


def require_adopted_native_sources():
    for file in (NATIVE_RECEIPT, PHASE_RECEIPT):
        for name, hashes in json.loads(file.read_text())['source'].items():
            if digest(ROOT/name) != hashes['candidate_sha256']:
                raise ValueError('Full native training requires the declared adopted source: '+name)


def sources():
    result = base.sources()
    result.update({name: digest(ROOT/name) for name in EXTRA_SOURCES})
    return dict(sorted(result.items()))


class TrainingStopped(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=1536)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--rollout-steps', type=int, default=128)
    parser.add_argument('--seed', type=int, default=271001)
    parser.add_argument('--training-case-seed', type=int, default=271701)
    parser.add_argument('--foundation-updates', type=int, default=192)
    parser.add_argument('--full-tasks-at', type=int, default=768)
    parser.add_argument('--domain-admission-warmup-updates', type=int, default=192)
    parser.add_argument('--domain-admission-ramp-updates', type=int, default=128)
    parser.add_argument('--domain-admission-seed', type=int, default=271501)
    parser.add_argument('--randomization-study', type=Path,
        default=ROOT/'contracts/wheel_leg_randomization_study_v2.json')
    parser.add_argument('--randomization-profile', choices=('requested',), default='requested')
    parser.add_argument('--randomization-strength', type=float, default=1.)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--speed-weight', type=float, default=2.)
    parser.add_argument('--contact-weight', type=float, default=8.)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not (128 <= args.num_envs <= 256 and args.rollout_steps >= 8
            and 0 < args.foundation_updates < args.full_tasks_at < args.updates
            and 0 <= args.domain_admission_warmup_updates
            and args.domain_admission_ramp_updates > 0
            and args.domain_admission_warmup_updates+args.domain_admission_ramp_updates < args.updates
            and args.randomization_strength == 1.):
        parser.error('Need128..256 environments, full requested widths and completed task/domain curricula before the endpoint')
    args.output, args.bank = args.output.resolve(), args.bank.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, task, contract = prepare(args)
    source_hashes = sources()
    revision = None
    if not args.preflight_only:
        require_adopted_native_sources()
        revision = base.archive_committed_sources(source_hashes, args.output)
    for name, expected in source_hashes.items():
        data = (ROOT/name).read_bytes()
        if digest(ROOT/name) != expected:
            raise ValueError('Source changed during snapshot: '+name)
        target = args.output/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    write_json(args.output/'source_hashes.json', source_hashes)
    (args.output/'pose_bank.json').write_bytes(args.bank.read_bytes())
    if args.preflight_only:
        write_json(args.output/'preflight.json', dict(schema='fresh-unified-preflight-v1',
            status='configuration_prepared_unqualified', source_count=len(source_hashes),
            task=task['fresh_unified_training'], native_steps=0, new_PPO=0,
            checkpoint_reads=[], native_patches_verified=False, qualified=False, hardware_ready=False))
        print(json.dumps(dict(status='configuration_prepared_unqualified', source_count=len(source_hashes))))
        return
    start = time.monotonic()
    report = dict(schema='scratch-standing-progress-v1', training_entry='fresh-unified-v1', status='starting',
        pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        started_utc=datetime.now(timezone.utc).isoformat(), seed=args.seed, completed_updates=0,
        optimizer_attempts=0, checkpoint_reads=[], requested_updates=args.updates,
        committed_source=revision, hardware_ready=False, qualified=False,
        args={**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              'precise_objective': True})
    write_json(args.output/'progress.json', report)
    launcher = env = runner = audit = None
    original_load = None
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        report['requested_stop_signal'] = signum

    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(dict(headless=True, device=args.device, enable_cameras=False,
            kit_args='--/exts/omni.kit.telemetry/skipDeferredStartup=true'))
        import torch
        import numpy as np
        from rsl_rl.runners import OnPolicyRunner
        from unified_training_runtime_candidate import make_environment_class
        from wheeled_tasks.chassis.control_contract import load_chassis_control
        from wheeled_tasks.chassis.physical_standing_reward import install as install_physical
        from wheeled_tasks.chassis.precise_standing_reward import install as install_precision
        from wheeled_algo.scratch_standing import agent_config, initialize_fresh, tensor_hash
        from wheeled_algo.recovery_rollout import RecoveryRolloutMask
        from wheeled_algo.native_ppo_audit import NativePPOAudit
        from wheeled_tasks.chassis.randomization_study import runtime_samples, scalar_metrics
        from domain_learning_admission_candidate import Admission, install as install_admission
        from record_scratch_physics import record_physics

        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        base.torch = torch
        original_load = torch.load

        def forbidden_load(*positional, **keywords):
            report['checkpoint_reads'].append(str(positional[0] if positional else keywords.get('f')))
            raise RuntimeError('Fresh unified training forbids every torch.load call')

        torch.load = forbidden_load
        manifest = json.loads((ROOT/cfg['asset_directory']/'manifest.json').read_text())
        environment = make_environment_class()
        env = environment(cfg, manifest, load_chassis_control(ROOT/cfg['control_math_source']), ROOT,
            stage_name='speed', num_envs=args.num_envs, device=args.device, seed=args.seed, level=1.,
            run_role='train', bank=args.bank, settings=task, training_case_seed=args.training_case_seed,
            foundation_updates=args.foundation_updates, full_tasks_at=args.full_tasks_at)
        report['domain_sensor_clock_version'] = env.domain.SENSOR_CLOCK_VERSION
        if report['domain_sensor_clock_version'] != task['domain_sensor_clock_version']:
            raise ValueError('Frozen sensor clock and actual implementation differ')
        report['physical_reward'] = install_physical(env, speed_weight=args.speed_weight, contact_weight=args.contact_weight)
        report['precise_reward'] = install_precision(env)
        if report['precise_reward'].get('phase_eligibility_version') != 'pre-interval-and-pre-reset-recovery-v1':
            raise ValueError('Full training requires phase-consistent recovery rewards')
        report['physics_runtime'] = record_physics(env)
        write_json(args.output/'startup_report.json', env.startup_report)
        configuration = agent_config(args.seed, args.device, args.rollout_steps, args.learning_rate,
            normalize_observations=True, wheel_velocity_increment=True, separate_leg_velocity_feedback=True)
        configuration['actor'].update(class_name='height_initialization_candidate:HeightInitializedModel',
            height_coefficients=contract['trainable_height_initialization']['height_coefficients'])
        configuration['experiment_name'] = 'fresh_unified'
        write_json(args.output/'agent_config.json', configuration)
        runner = OnPolicyRunner(env, configuration, log_dir=str(args.output), device=args.device)
        from wheeled_algo.standing_load_initialization import initial_action
        initial = initial_action(base.nominal_action(env), contract['static_load_initialization'])
        report['initialization'] = initialize_fresh(runner.alg, initial,
            output_bias_source=contract['trainable_height_initialization']['source'])
        report['initialization'].update(actor_class=configuration['actor']['class_name'],
            wheel_velocity_increment=True, separate_leg_velocity_feedback=True,
            trainable_height_initialization=contract['trainable_height_initialization'])
        write_json(args.output/'initialization.json', report['initialization'])
        write_json(args.output/'randomization_study.initial_samples.json', runtime_samples(env))
        initial_states = {name: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            for name, model in (('actor', runner.alg.actor), ('critic', runner.alg.critic))}
        mask = RecoveryRolloutMask(runner.alg, num_envs=args.num_envs)
        admission = Admission(env.domain.enabled, warmup_updates=args.domain_admission_warmup_updates,
            ramp_updates=args.domain_admission_ramp_updates, seed=args.domain_admission_seed)
        report['domain_learning_admission'] = install_admission(env, runner.alg, admission, record_group_metrics=True)
        report['domain_learning_admission'].update(training_adopted=True, total_updates=args.updates)
        native_update = runner.alg.update

        def save(path, infos=None):
            path = Path(path)
            state = runner.alg.save()
            state.update(iter=report['completed_updates'], infos=dict(schema='scratch-standing-checkpoint-v1',
                initialization=report['initialization'], completed_updates=report['completed_updates'],
                training_origin='scratch', continuation_forbidden=True, committed_source=revision,
                settings=task, source_hashes=source_hashes, bank_sha256=digest(args.bank),
                actor_class=configuration['actor']['class_name'], agent_config_sha256=digest(args.output/'agent_config.json'),
                mechanism_config_sha256=digest(args.output/'config/contract.json'),
                domain_learning_admission=report['domain_learning_admission'],
                fresh_unified_training=task['fresh_unified_training'],
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                env_rng=env.generator.get_state(), domain_rng=env.domain.rng.get_state()))
            temporary = path.with_suffix(path.suffix+'.tmp')
            torch.save(state, temporary)
            temporary.replace(path)
            report['last_checkpoint'] = dict(path=str(path), sha256=digest(path), updates=report['completed_updates'])

        def update():
            nonlocal audit
            report['optimizer_attempts'] += 1
            report['optimizer_update_in_flight'] = True
            write_json(args.output/'progress.json', report)
            losses = native_update()
            report['completed_updates'] += 1
            report['optimizer_update_in_flight'] = False
            for model in (runner.alg.actor, runner.alg.critic):
                if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise RuntimeError('Nonfinite fresh PPO parameter')
            if report['completed_updates'] == 1:
                audit.close()
                result = audit.report()
                for name, model in (('actor', runner.alg.actor), ('critic', runner.alg.critic)):
                    result[name+'_changed_parameter_tensors'] = sum(not torch.equal(initial_states[name][key], value.detach().cpu())
                        for key, value in model.named_parameters())
                    result[name+'_sha256'] = tensor_hash(model.state_dict())
                if (not result['all_recorded_values_finite'] or result['committed_optimizer_calls'] != 12
                        or not result['actor_changed_parameter_tensors'] or not result['critic_changed_parameter_tensors']):
                    raise RuntimeError('First fresh native PPO audit rejected')
                write_json(args.output/'first_update_audit.json', result)
                audit = None
                for key, value in scalar_metrics(task['randomization_study']).items():
                    runner.logger.writer.add_scalar('DomainStudyConfig/'+key, value, 0)
                for key in ('foundation_updates', 'full_tasks_at', 'independent_training_seed'):
                    runner.logger.writer.add_scalar('FreshTasksConfig/'+key, task['fresh_unified_training'][key], 0)
                for key in ('warmup_updates', 'ramp_updates', 'total_updates', 'full_population_at_completed_update'):
                    runner.logger.writer.add_scalar('DomainAdmissionConfig/'+key, task['domain_learning_admission'][key], 0)
                runner.logger.writer.flush()
            env.actor_updates = report['completed_updates']
            report.update(status='training', elapsed_s=time.monotonic()-start,
                last_losses={key: float(value) for key, value in losses.items()}, valid_rollout=mask.metrics,
                last_physical_metrics={key: float(value) for key, value in env.physical_standing_last['metrics'].items()})
            write_json(args.output/'progress.json', report)
            if stop_requested:
                save(args.output/'model_stopped.pt')
                raise TrainingStopped()
            return losses

        runner.save, runner.alg.update = save, update
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        save(args.output/'model_initial.pt')
        audit = NativePPOAudit(runner.alg)
        runner.learn(args.updates, init_at_random_ep_len=False)
        save(args.output/'model_final.pt')
        report.update(status='completed_unqualified', elapsed_s=time.monotonic()-start,
            evaluation=dict(status='not_performed', required='cold full13 same-policy held-out suite', qualified=False))
    except TrainingStopped:
        report.update(status='stopped', elapsed_s=time.monotonic()-start)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc(), elapsed_s=time.monotonic()-start)
        raise
    finally:
        write_json(args.output/'progress.json', report)
        if original_load is not None:
            torch.load = original_load
        if audit is not None:
            audit.close()
        if runner is not None and runner.logger.writer is not None:
            runner.logger.writer.flush()
            runner.logger.writer.close()
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


if __name__ == '__main__':
    main()
