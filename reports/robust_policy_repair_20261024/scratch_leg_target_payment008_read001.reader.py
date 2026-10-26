"""Read actual RSL event payments without interacting with the learner."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/scratch_standing_20261026/pilot008_leg_target_rate020'
OUTPUT = Path(__file__).with_name('scratch_leg_target_payment008_read001.json')
assert not OUTPUT.exists(), 'Preserve earlier observations'
contract = json.loads((RUN / 'task_contract.json').read_text())['standing_leg_target_rate_contract']
assert contract['weight_density_per_second'] == .2 and contract['policy_dt_s'] == .005
event, = RUN.glob('events.out.tfevents.*')
raw = event.read_bytes()
tags = [
    'LegTargetRate/target_delta_rms_rad',
    'LegTargetRate/target_delta_peak_rad',
    'LegTargetRate/target_rate_rms_rad_s',
    'LegTargetRate/cost_density_per_second',
    'LegTargetRate/paid_cost_per_policy_step',
    'LegTargetRate/weight_density_per_second',
    'PrecisionStanding/score_density_per_second',
    'PrecisionStanding/reward_per_policy_step',
    'ScratchStanding/paid_speed_cost_per_policy_step',
    'ScratchStanding/paid_support_cost_per_policy_step',
    'Policy/mean_std',
]
with tempfile.TemporaryDirectory() as directory:
    prefix = Path(directory) / event.name
    prefix.write_bytes(raw)
    accumulator = EventAccumulator(str(prefix), size_guidance={'scalars': 0}).Reload()
    data = {}
    for tag in tags:
        rows = accumulator.Scalars(tag)
        values = {row.step: row.value for row in rows}
        assert len(values) == len(rows) and all(math.isfinite(v) for v in values.values()), tag
        data[tag] = values
common = 0
while all(common in values for values in data.values()):
    common += 1
assert common >= 32
weight_error = payment_error = rate_error = 0.
for step in range(common):
    value = {tag: data[tag][step] for tag in tags}
    weight_error = max(weight_error, abs(value[tags[5]] - .2))
    payment_error = max(payment_error, abs(value[tags[3]] * .005 - value[tags[4]]))
    rate_error = max(rate_error, abs(value[tags[0]] / .005 - value[tags[2]]))
    assert value[tags[1]] >= value[tags[0]] >= 0.
    assert value[tags[3]] >= 0. and value[tags[4]] >= 0.
assert weight_error < 1e-7 and payment_error < 2e-8 and rate_error < 2e-6
windows = []
for first in range(0, common // 32 * 32, 32):
    means = {tag: sum(data[tag][step] for step in range(first, first + 32)) / 32 for tag in tags}
    base_reward = means['PrecisionStanding/reward_per_policy_step']
    windows.append(dict(
        PPO_update_numbers_inclusive=[first + 1, first + 32], means=means,
        paid_cost_divided_by_precise_reward=(means[tags[4]] / base_reward if base_reward > 0. else None),
    ))
manifest = json.loads((RUN / 'source_hashes.json').read_text())
source = {
    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    and hashlib.sha256((RUN / 'source_snapshot' / name).read_bytes()).hexdigest() == expected
    for name, expected in manifest.items()
}
assert len(source) == 263 and all(source.values())
report = dict(
    schema='scratch-leg-target-payment-observation-v1', observed_utc=datetime.now(timezone.utc).isoformat(),
    run=str(RUN), complete_updates=common,
    event_prefix=dict(path=str(event), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
    all_live_and_frozen_sources_match=True, source_count=len(source),
    maximum_logged_weight_error=weight_error, maximum_density_times_dt_payment_error=payment_error,
    maximum_delta_divided_by_dt_rate_error=rate_error, all_payment_checks_pass=True, windows=windows,
    reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    new_PPO=0, native_steps=0, behavior_qualification=False, hardware_ready=False,
    scope='Actual RSL scalar payment and unit checks. Target rates are commanded references, not measured motor velocity. '
          'Cost/base-reward ratio is a ratio of logged means; it does not attribute advantages, causality or standing success.',
)
OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('windows', 'scope')}))
print(json.dumps(dict(last_complete_window=windows[-1])))
