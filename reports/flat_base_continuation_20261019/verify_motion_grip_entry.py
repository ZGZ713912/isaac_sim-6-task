"""Read the real native entry state without waiting for a new optimizer update."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_tasks.chassis.flat_base_continuation import digest, encode

torch.set_num_threads(1)
load = lambda path: torch.load(path, map_location='cpu', weights_only=True)
parent_path = ROOT / 'runs/flat_base_20261019/pilot029_static_grip1338/model_001338.pt'
entry_path = ROOT / 'runs/flat_base_20261019/pilot030_motion_grip1348/initial_state.pt'
parent, entry = load(parent_path), load(entry_path)
info = entry['infos']
candidate = HERE / 'motion_grip_candidate001'
prep = json.loads((candidate / 'preparation001.json').read_text())
queue = json.loads((ROOT / 'runs/flat_base_20261019/motion_grip1348_queue001.json').read_text())
material = info['startup']['contact_material']


def exact(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


expected_optimizer = deepcopy(parent['optimizer_state_dict'])
expected_optimizer['param_groups'][0]['lr'] = .0001
pid = queue['worker_pid']
stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
checks = dict(
    exact_parent=digest(parent_path) == info['parent_checkpoint_sha256'] == prep['parent_checkpoint_sha256'],
    actor_and_critic_exact=all(exact(parent[key], entry[key]) for key in ('actor_state_dict', 'critic_state_dict')),
    native_full_restore=len(info['full_restore_verification']) == 6 and all(info['full_restore_verification'].values()),
    entire_parent_Adam_exact=exact(expected_optimizer, entry['optimizer_state_dict'])
        and info['optimizer_after_declared_rate_exact'],
    sampling_rng_exact=all(exact(parent[key], entry[key]) for key in ('torch_rng_state', 'cuda_rng_state')),
    inherited_counters_exact=all(info[key] == parent['infos'][key] for key in
        ('successful_updates_total', 'actor_updates_total', 'consumed_optimizer_updates_total', 'training_transitions_total')),
    initial_two_critic_updates=info['critic_warmup_remaining'] == 2 and info['motion_grip_actor_updates'] == 0 and info['static_grip_actor_updates'] == 8
        and info['actor_trainability']['trainable_parameters'] == 0,
    current_sources_exact=info['source_sha256'] == prep['source_sha256']
        and all(digest(ROOT / name) == value for name, value in prep['source_sha256'].items()),
    latest_material_ranges=material['ground_static_range'] == material['ground_dynamic_range'] == [.2, .5]
        and material['wheel_static_friction'] == material['wheel_dynamic_friction'] == .1,
    sampled4096_ground_in_range=len(material['ground_static_friction_per_environment']) == 4096
        and .2 <= min(material['ground_static_friction_per_environment'])
        <= max(material['ground_static_friction_per_environment']) <= .5,
    native_material_readback=material['native_readback']['usd_collision_bindings_verified']
        and material['native_readback']['physx_robot_shape_coefficients_verified'],
    worker_identity_live=stat[0] != 'Z' and stat[19] == queue['worker_start_ticks']
        and 'scripts/train_flat_base.py' in Path(f'/proc/{pid}/cmdline').read_text(),
)
target = candidate / 'native_entry001.json'
if target.exists():
    raise ValueError('Preserve the existing entry receipt')
target.write_bytes(encode(dict(schema='flat-motion-grip-native-entry-audit-v1', passed=all(checks.values()),
    checks=checks, parent_checkpoint_sha256=digest(parent_path), initial_state_sha256=digest(entry_path),
    sampled_ground_min_max=material['sampled_ground_static_min_max'], worker_pid=pid,
    reader_sha256=digest(Path(__file__)), observed_new_optimizer_updates_at_entry=0,
    scope='Native initialization and full-state restoration; subsequent learning and behavior remain pending')))
print(json.dumps(dict(passed=all(checks.values()), checks=checks)))
if not all(checks.values()):
    raise SystemExit(1)
