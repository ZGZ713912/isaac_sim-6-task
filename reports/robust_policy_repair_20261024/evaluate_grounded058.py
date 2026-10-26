"""Labelled reset-placement diagnostic with unchanged native policy and reward."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import evaluate_gap_robustness as evaluation

native_evaluate = evaluation.evaluate


def evaluate(env, mean, output, torch, *, seed, seconds=12., cases=None):
    import numpy as np
    from wheeled_tasks.chassis.grounded_standing_reset import install, MODULE, VERSION
    from wheeled_tasks.chassis.standing_contact_monitor import StandingContactMonitor
    placement = install(env)
    env.evaluation_overrides = dict(env.evaluation_overrides,
        grounded_standing_reset=dict(version=VERSION, source_sha256=evaluation.digest(ROOT / MODULE),
            preload_m=placement['preload_m'], scope='standing initialization only'))
    monitor = StandingContactMonitor(env, record=True)
    result = native_evaluate(env, mean, output, torch, seed=seed, seconds=seconds, cases=cases)
    target = Path(output) / 'native_wheel_contact_1ms.npz'
    np.savez_compressed(target, **monitor.arrays())
    evaluation.write_json(Path(output) / 'native_wheel_contact_1ms.json', dict(
        schema='native-1ms-wheel-support-readback-v1', data=str(target), sha256=evaluation.digest(target),
        physical_dt_s=.001, decimation=5, force_threshold_n=2.,
        axis_order='policy_frame,physical_substep,environment,wheel,xyz',
        actions_changed=False, reward_changed=False, reset_changed=True, reset=placement,
        evaluation_overrides=env.evaluation_overrides, new_PPO=0,
        monitor_source_sha256=evaluation.digest(ROOT / 'src/wheeled_tasks/chassis/standing_contact_monitor.py')))
    return result


evaluation.evaluate = evaluate
evaluation.main()
