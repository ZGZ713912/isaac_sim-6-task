"""Optional zero-PPO quiet-leg target bandwidth diagnostic, explicitly labelled."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import evaluate_gap_robustness as evaluation

if '--quiet-final-leg-alpha' not in sys.argv:
    raise ValueError('Require an explicit diagnostic leg output alpha')
index = sys.argv.index('--quiet-final-leg-alpha')
alpha = float(sys.argv[index + 1]); del sys.argv[index:index + 2]
if not .05 <= alpha <= 1.:
    raise ValueError('Diagnostic alpha must be within.05..1')
native_evaluate = evaluation.evaluate


def evaluate(env, mean, output, torch, *, seed, seconds=12., cases=None):
    import numpy as np
    from wheeled_tasks.chassis.grounded_standing_reset import install, MODULE, VERSION
    from wheeled_tasks.chassis.standing_contact_monitor import StandingContactMonitor
    placement = install(env)
    env.evaluation_overrides = dict(env.evaluation_overrides,
        grounded_standing_reset=dict(version=VERSION, source_sha256=evaluation.digest(ROOT / MODULE),
            preload_m=placement['preload_m'], scope='standing initialization only'))
    env.evaluation_overrides['diagnostic_quiet_final_leg_alpha'] = dict(alpha=alpha,
        source_sha256=evaluation.digest(Path(__file__)), scope='existing quiet gate; causal previous-action35D; circular delta; wheels unchanged')
    class FilteredQuietLegs(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original

        def forward(self, obs):
            action = self.original(obs)
            previous = obs[:, 22:26]
            difference = (action[:, :4] - previous) * .25
            delta = torch.atan2(difference.sin(), difference.cos()) / .25
            filtered = previous + alpha * delta
            gate = self.original.quiet_stance_gate(obs)
            legs = action[:, :4] + gate * (filtered - action[:, :4])
            return torch.cat((legs, action[:, 4:]), -1)
    mean = FilteredQuietLegs(mean)
    monitor = StandingContactMonitor(env, record=True)
    result = native_evaluate(env, mean, output, torch, seed=seed, seconds=seconds, cases=cases)
    target = Path(output) / 'native_wheel_contact_1ms.npz'
    np.savez_compressed(target, **monitor.arrays())
    evaluation.write_json(Path(output) / 'native_wheel_contact_1ms.json', dict(
        schema='native-1ms-wheel-support-readback-v1', data=str(target), sha256=evaluation.digest(target),
        physical_dt_s=.001, decimation=5, force_threshold_n=2.,
        axis_order='policy_frame,physical_substep,environment,wheel,xyz',
        actions_changed=True, reward_changed=False, reset_changed=True, reset=placement,
        evaluation_overrides=env.evaluation_overrides, new_PPO=0,
        monitor_source_sha256=evaluation.digest(ROOT / 'src/wheeled_tasks/chassis/standing_contact_monitor.py')))
    return result


evaluation.evaluate = evaluate
evaluation.main()
