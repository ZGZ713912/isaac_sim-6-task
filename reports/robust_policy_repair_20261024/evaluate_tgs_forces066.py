"""Labelled zero-PPO TGS external-force integration diagnostic, native actor unchanged."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import evaluate_gap_robustness as evaluation
from isaaclab.app import AppLauncher

native_launcher = AppLauncher.__init__
construction = []


def launch(self, *args, **kwargs):
    native_launcher(self, *args, **kwargs)
    from isaaclab.sim import SimulationContext
    from isaaclab_physx.physics import PhysxCfg
    native_context = SimulationContext.__init__

    def context(instance, cfg=None):
        if cfg is None:
            raise ValueError('Require explicit SimulationCfg')
        default_resolved = cfg.physics is None
        if default_resolved:
            cfg.physics = PhysxCfg()
        if cfg.physics.solver_type != 1:
            raise ValueError('Require explicit native TGS SimulationCfg')
        if cfg.physics.enable_external_forces_every_iteration:
            raise ValueError('Expected the frozen default false before the single-variable intervention')
        before = cfg.to_dict()
        cfg.physics.enable_external_forces_every_iteration = True
        after = cfg.to_dict()
        check = dict(after)
        check['physics'] = dict(after['physics'])
        check['physics']['enable_external_forces_every_iteration'] = False
        if check != before:
            raise ValueError('Unexpected simulation configuration change')
        construction.append(dict(default_physics_resolved=default_resolved,
            before=repr(before), after=repr(after)))
        return native_context(instance, cfg)

    SimulationContext.__init__ = context


AppLauncher.__init__ = launch
native_evaluate = evaluation.evaluate


def evaluate(env, mean, output, torch, *, seed, seconds=12., cases=None):
    import numpy as np
    from wheeled_tasks.chassis.grounded_standing_reset import install, MODULE, VERSION
    from wheeled_tasks.chassis.standing_contact_monitor import StandingContactMonitor
    if len(construction) != 1:
        raise ValueError('Need exactly one labelled native simulation context')
    cfg = env.sim.cfg
    if not cfg.physics.enable_external_forces_every_iteration or cfg.physics.solver_type != 1:
        raise ValueError('Native TGS override not active')
    prim = env.sim.stage.GetPrimAtPath(cfg.physics_prim_path)
    authored = prim.GetAttribute('physxScene:enableExternalForcesEveryIteration').Get()
    if authored is not True:
        raise ValueError('Native USD scene did not read back the requested true flag')
    evaluation.write_json(Path(output).parent / 'tgs_force_integration_readback.json', dict(
        schema='native-tgs-force-integration-ablation-v1', construction=construction,
        actual_scene_path=str(prim.GetPath()), authored_flag=authored,
        solver_type=prim.GetAttribute('physxScene:solverType').Get(),
        only_change='enable_external_forces_every_iteration false to true',
        policy_changed=False, new_PPO=0, qualification=None))
    placement = install(env)
    env.evaluation_overrides = dict(env.evaluation_overrides,
        grounded_standing_reset=dict(version=VERSION, source_sha256=evaluation.digest(ROOT / MODULE),
            preload_m=placement['preload_m'], scope='standing initialization only'),
        diagnostic_tgs_external_forces_every_iteration=dict(before=False, after=True,
            source_sha256=evaluation.digest(Path(__file__)), native_USD_readback=authored))
    monitor = StandingContactMonitor(env, record=True)
    result = native_evaluate(env, mean, output, torch, seed=seed, seconds=seconds, cases=cases)
    target = Path(output) / 'native_wheel_contact_1ms.npz'
    np.savez_compressed(target, **monitor.arrays())
    evaluation.write_json(Path(output) / 'native_wheel_contact_1ms.json', dict(
        schema='native-1ms-wheel-support-readback-v1', data=str(target), sha256=evaluation.digest(target),
        physical_dt_s=.001, decimation=5, force_threshold_n=2.,
        axis_order='policy_frame,physical_substep,environment,wheel,xyz',
        actions_changed=False, reward_changed=False, reset_changed=True, reset=placement,
        physics_changed='TGS external force integration flag only',
        evaluation_overrides=env.evaluation_overrides, new_PPO=0,
        monitor_source_sha256=evaluation.digest(ROOT / 'src/wheeled_tasks/chassis/standing_contact_monitor.py')))
    return result


evaluation.evaluate = evaluate
evaluation.main()
