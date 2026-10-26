"""Native1ms leg-PD damping ablation; policy and contact predicates unchanged."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import evaluate_gap_robustness as evaluation

if '--leg-kd-factor' not in sys.argv:
    raise ValueError('Require an explicit native PD damping diagnostic factor')
index = sys.argv.index('--leg-kd-factor')
kd_factor = float(sys.argv[index + 1]); del sys.argv[index:index + 2]
if not .5 <= kd_factor <= 1.5:
    raise ValueError('Diagnostic KD factor must stay within.5..1.5')
native_evaluate = evaluation.evaluate


def evaluate(env, mean, output, torch, *, seed, seconds=12., cases=None):
    import numpy as np
    from wheeled_tasks.chassis.grounded_standing_reset import install, MODULE, VERSION
    from wheeled_tasks.chassis.standing_contact_monitor import StandingContactMonitor
    placement = install(env)
    env.evaluation_overrides = dict(env.evaluation_overrides,
        grounded_standing_reset=dict(version=VERSION, source_sha256=evaluation.digest(ROOT / MODULE),
            preload_m=placement['preload_m'], scope='standing initialization only'))
    native_reset = env.reset

    def reset(ids):
        result = native_reset(ids)
        env.domain.kd[ids] *= kd_factor
        return result

    env.reset = reset
    env.domain.kd *= kd_factor
    env.evaluation_overrides['diagnostic_leg_kd_factor'] = dict(factor=kd_factor,
        source_sha256=evaluation.digest(Path(__file__)), nominal_kd_nm_s_rad=2.5,
        scope='multiply native leg velocity feedback coefficient only; same force/torque bounds')
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
