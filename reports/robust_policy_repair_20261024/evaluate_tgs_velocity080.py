"""Separate numerical diagnostic: TGS velocity iterations32->1 only, no PPO.

Historical64/32 acceptance remains unchanged. Results under this deliberately
different solver setting cannot qualify the original policy or physics model.
"""
from copy import deepcopy
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
    from wheeled_tasks.chassis.env import ChassisEnv
    native_init = ChassisEnv.__init__

    def initialize(instance, config, *args, **kwargs):
        if config['position_iterations'] != 64 or config['velocity_iterations'] != 32:
            raise ValueError('Require unchanged64/32 baseline before the diagnostic')
        before, after = deepcopy(config), deepcopy(config)
        after['velocity_iterations'] = 1
        check = deepcopy(after)
        check['velocity_iterations'] = 32
        if check != before:
            raise ValueError('Unexpected configuration change')
        construction.append(dict(only_change='velocity_iterations32_to1',
            before=repr(before), after=repr(after)))
        return native_init(instance, after, *args, **kwargs)

    ChassisEnv.__init__ = initialize


AppLauncher.__init__ = launch
native_evaluate = evaluation.evaluate


def evaluate(env, mean, output, torch, *, seed, seconds=12., cases=None):
    from pxr import PhysxSchema
    if len(construction) != 1 or env.cfg['position_iterations'] != 64 or env.cfg['velocity_iterations'] != 1:
        raise ValueError('Diagnostic configuration not active exactly once')
    roots = [prim for prim in env.sim.stage.Traverse() if prim.HasAPI(PhysxSchema.PhysxArticulationAPI)]
    readback = [dict(path=str(p.GetPath()),
        position=p.GetAttribute('physxArticulation:solverPositionIterationCount').Get(),
        velocity=p.GetAttribute('physxArticulation:solverVelocityIterationCount').Get()) for p in roots]
    if len(readback) != env.num_envs or any(r['position'] != 64 or r['velocity'] != 1 for r in readback):
        raise ValueError('Native articulation solver readback differs from64/1')
    cfg = env.sim.cfg.physics
    if cfg.solver_type != 1 or cfg.min_velocity_iteration_count > 1 or cfg.max_velocity_iteration_count < 1:
        raise ValueError('Scene solver clamps do not permit TGS velocity1')
    override = dict(before=32, after=1, position_iterations=64, native_USD_readback=readback,
        scene_min_velocity_iterations=cfg.min_velocity_iteration_count,
        scene_max_velocity_iterations=cfg.max_velocity_iteration_count,
        enable_external_forces_every_iteration=cfg.enable_external_forces_every_iteration,
        source_sha256=evaluation.digest(Path(__file__)), original_acceptance_unchanged=True,
        qualification_eligible=False)
    env.evaluation_overrides = dict(env.evaluation_overrides, diagnostic_tgs_velocity_iterations=override)
    evaluation.write_json(Path(output).parent / 'tgs_velocity_readback080.json', dict(
        schema='native-tgs-velocity-iterations-diagnostic-v1', construction=construction,
        override=override, policy_changed=False, new_PPO=0, qualification=None))
    return native_evaluate(env, mean, output, torch, seed=seed, seconds=seconds, cases=cases)


evaluation.evaluate = evaluate
import runpy
runpy.run_path(str(Path(__file__).with_name('evaluate_height_matched073.py')), run_name='__main__')
