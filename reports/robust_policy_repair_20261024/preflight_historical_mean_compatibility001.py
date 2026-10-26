"""Check the explicit legacy migration against its exact archived forward code."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import time
import types

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from export_robust_getup import load_mean, digest
from train_robust_getup import write_json

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
source = Path('runs/unified_robust_20261022/offline_recovery096/model_migrated.pt')
archive = source.parent / 'source'
assert digest(source) == 'eb49ecdf871d5d2ca218586b75072c62f61771fc8431171f363c95d8bbb7a50f'
with torch.random.fork_rng(devices=[]):
    compatible, info = load_mean(source, migration_source_root=archive)
changed = {n: dict(expected=s, current=digest(n), archive=digest(archive / n))
           for n, s in info['source_sha256'].items() if n.startswith('src/wheeled_algo/') and digest(n) != s}
assert set(changed) == {'src/wheeled_algo/robust_getup.py', 'src/wheeled_algo/recovery_coordinates.py'}
assert all(row['expected'] == row['archive'] for row in changed.values())
modules = []
for filename in ('recovery_coordinates', 'robust_getup'):
    path = archive / 'src/wheeled_algo' / (filename + '.py')
    name = 'wheeled_algo._archived096_' + filename
    module = types.ModuleType(name)
    module.__file__ = str(path.resolve()); module.__package__ = 'wheeled_algo'
    sys.modules[name] = module
    exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
    modules.append(module)
coordinates, actor_module = modules
actor_module.smooth_recovery_legs = coordinates.smooth_recovery_legs
original = deepcopy(compatible)
original.__class__ = actor_module.RobustGetupActor
replays = []
with torch.inference_mode():
    for path in (Path('runs/unified_robust_20261022/early_roll_push107/robust/trajectory.npz'),
                 B / 'diagnosis026_motion_damping080_nominal_pair/robust/trajectory.npz'):
        with np.load(path) as data:
            raw = data['observations'].reshape(-1, 35)
            raw = raw[np.isfinite(raw).all(-1)]
            observations = torch.from_numpy(raw[np.linspace(0, len(raw)-1, 16384).astype(int)].copy())
        a, b = original(observations), compatible(observations)
        replays.append(dict(path=str(path), inputs=len(observations),
            bitexact=bool(torch.equal(a, b)), max_action_delta=float((a - b).abs().max())))
checks = dict(explicit_archive_checked_migration_load=True,
    exactly_two_reviewed_opt_in_source_differences=len(changed) == 2,
    archived_source_hashes_exact=all(v['archive'] == v['expected'] for v in changed.values()),
    original_forward_executed=True,
    archived_smooth_decoder_executed=actor_module.smooth_recovery_legs is coordinates.smooth_recovery_legs,
    old_default_knee_step=compatible.recovery_knee_step == .15,
    circular_extension_disabled=compatible.recovery_previous_common_version is None,
    no_new_entry_or_target_limiter=all(info['settings'].get(k) is None for k in (
        'gap_entry_handoff_version', 'gap_upright_target_limit_version', 'gap_final_target_winding_version')),
    historical_and_current_recorded_function_bitexact=all(v['bitexact'] for v in replays))
assert all(checks.values()), checks
out = dict(schema='historical-mean-explicit-migration-compatibility-v1', checks=checks,
    checkpoint=str(source), checkpoint_sha256=digest(source), archive=str(archive), changed_sources=changed,
    replays=replays, clock=569, new_PPO=0, simulator_started=False,
    limitation='CPU archived/current function equality at recorded inputs. This is not native trajectory replay, not a current-observer qualification and not an adoption of the historical policy.',
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'historical_mean_compatibility_preflight001.json', out)
print(json.dumps(out, indent=2))
