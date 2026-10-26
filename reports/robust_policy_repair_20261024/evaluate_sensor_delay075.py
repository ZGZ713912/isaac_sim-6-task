"""Verify zero transport delay at every actual policy call, before final reset."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import evaluate_gap_robustness as evaluation

native_evaluate = evaluation.evaluate


def evaluate(env, mean, output, torch, *, seed, seconds=12., cases=None):
    env.evaluation_overrides = dict(env.evaluation_overrides, sensor_delay_policy_ticks=0)

    class CheckedMean(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original
            self.calls = 0

        def forward(self, obs):
            if bool(env.domain.sensor_delay.any()):
                raise ValueError('Nonzero sensor delay at an actual evaluation policy call')
            self.calls += 1
            return self.original(obs)

    checked = CheckedMean(mean)
    result = native_evaluate(env, checked, output, torch, seed=seed, seconds=seconds, cases=cases)
    assert checked.calls == round(seconds / env.policy_dt)
    evaluation.write_json(Path(output) / 'sensor_delay_policy_calls.json', dict(
        zero_delay_verified_each_policy_call=True, policy_calls=checked.calls,
        actor_output_changed=False, sensor_noise_changed=False, RNG_changed=False,
        scope='Checked during inference; final native environment reset is outside evaluation.'))
    return result


evaluation.evaluate = evaluate
runpy.run_path(str(Path(__file__).with_name('evaluate_height_matched073.py')), run_name='__main__')
