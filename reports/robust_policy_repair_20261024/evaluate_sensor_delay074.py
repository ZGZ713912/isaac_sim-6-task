"""Only remove sampled sensor transport delay; preserve RNG, noise and gains."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import evaluate_gap_robustness as evaluation

native_evaluate = evaluation.evaluate


def evaluate(env, mean, output, torch, *, seed, seconds=12., cases=None):
    env.evaluation_overrides = dict(env.evaluation_overrides, sensor_delay_policy_ticks=0)
    result = native_evaluate(env, mean, output, torch, seed=seed, seconds=seconds, cases=cases)
    if bool(env.domain.sensor_delay.any()):
        raise ValueError('Native sensor transport delay was not zero')
    return result


evaluation.evaluate = evaluate
runpy.run_path(str(Path(__file__).with_name('evaluate_height_matched073.py')), run_name='__main__')
