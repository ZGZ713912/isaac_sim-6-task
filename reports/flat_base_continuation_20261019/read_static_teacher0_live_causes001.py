"""Read native causes from a completed, flushed prefix of the current trial."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import math
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/flat_base_20261019/pilot026'
events = EventAccumulator(str(RUN),size_guidance={'scalars':0})
events.Reload()
tags = set(events.Tags()['scalars'])
steps = sorted(p.step for p in events.Scalars('FlatTransfer/actor_updates_total'))
last = max(steps)
output = HERE / f'static_teacher0_native_causes_live{last}.json'
if output.exists():
    raise SystemExit('Preserve the completed prefix review')
rows = []
unique = {n:0 for n in steps}
for tag in sorted(tags):
    if not tag.startswith('FlatTermination/whole/') or tag.count('/') != 4:
        continue  # Unsplit groups and disjoint height bins only.
    values = {p.step:p.value for p in events.Scalars(tag) if p.step in steps and p.value}
    if values:
        if any(not math.isfinite(v) or v < 0 or not v.is_integer() for v in values.values()):
            raise ValueError('Native flag counts must be nonnegative integers')
        rows.append(dict(tag=tag,counts_by_completed_update=values))
        if tag.endswith('/terminated_events'):
            for n,v in values.items():
                unique[n] += v
failure = {p.step+1:p.value for p in events.Scalars('/reward/failure_event') if p.step+1 in steps}
checks = dict(all_native_reward_steps_present=sorted(failure) == steps,
    terminated_count_matches_native_reward=all(math.isclose(unique[n],-failure.get(n,float('nan'))*4096*96/200,
                                                         abs_tol=1e-5) for n in steps))
if not all(checks.values()):
    raise SystemExit('Native flags and event payment are not fully flushed; retry later')
result = dict(schema='flat-native-cause-completed-prefix-readout-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(),passed=True,checks=checks,
    completed_update_steps=steps,unique_terminated_flagged_frames=sum(unique.values()),
    nonzero_unsplit_cause_and_outcome_counts=rows,
    event_file_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.glob('events.out.tfevents.*')},
    scope='Completed training prefix only. Time-limit done is separate from terminated. Cause flags can overlap, and this cannot retroactively identify causes in the prior1264 trial.',
    new_native_steps=0,new_optimizer_updates=0,behavior_qualified=False)
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(last_completed_update=last,unique_terminated_frames=result['unique_terminated_flagged_frames'],
    nonzero_failure_causes=[r for r in rows if not r['tag'].endswith(('done_events','success_events'))]),indent=2))
