"""Execute the real legacy wrapper with only the native environment stubbed."""
from pathlib import Path
from copy import deepcopy
import importlib.util
import json
import sys
import time
import types

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
import export_robust_getup as export
import evaluate_robust_getup_batch as batch
from train_robust_getup import write_json

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
wrapper = R / 'evaluate_historical_full_mean001.py'
checkpoint = B / 'motion_damping080002/model_probe.pt'
source = Path('runs/unified_robust_20261022/offline_recovery096/model_migrated.pt')
output = B / 'historical_full_mean_wrapper_cpu001'
assert not output.exists()
with torch.random.fork_rng(devices=[]):
    expected, _ = export.load_mean(source, migration_source_root=source.parent / 'source')
with np.load(Path('runs/unified_robust_20261022/early_roll_push107/robust/trajectory.npz')) as data:
    observations = torch.from_numpy(data['observations'].reshape(-1, 35)[::19].copy())
native_load = export.load_mean
native_main = batch.main
module_name = 'wheeled_tasks.chassis.robust_getup_env'
assert module_name not in sys.modules
stub = types.ModuleType(module_name)
class Env: pass
stub.RobustGetupEnv = Env
sys.modules[module_name] = stub
result = {}


def fake_main():
    actual, info = export.load_mean(checkpoint)
    env = Env(); env.evaluation_overrides = {'native_stub': True}
    label = env.evaluation_overrides['historical_full_mean_probe']
    with torch.inference_mode():
        result.update(original_function_bitexact=bool(torch.equal(expected(observations), actual(observations))),
            current_environment_profile_info_retained=info['completed_updates'] == 1501,
            historical_mean_class=type(actual).__name__ == 'RobustGetupActor',
            metadata_survives_native_assignment=env.evaluation_overrides['native_stub'] is True,
            explicit_current_observer_label=label['current_observer_retained'],
            no_saved_checkpoint_qualification=label['eligible_for_checkpoint_mean_qualification'] is False,
            target_limiter_difference_disclosed=label['current_actor_target_limiter_retained'] is False)


try:
    batch.main = fake_main
    sys.argv = [str(wrapper), '--checkpoint', str(checkpoint), '--output', str(output)]
    spec = importlib.util.spec_from_file_location('historical_full_wrapper_fixture', wrapper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    batch.main = native_main
    export.load_mean = native_load
    del sys.modules[module_name]
result.update(metadata_definition_removed_after_return='evaluation_overrides' not in Env.__dict__,
    source_models_unchanged=export.digest(checkpoint) == '9a3610dc396c97d5bf3a0965f2be1217ac3e225e857604554c84b991d7f9eee9'
        and export.digest(source) == 'eb49ecdf871d5d2ca218586b75072c62f61771fc8431171f363c95d8bbb7a50f')
assert all(result.values()), result
write_json(R / 'historical_full_wrapper_preflight001.json', dict(
    schema='historical-full-mean-wrapper-preflight-v1', checks=result,
    wrapper_sha256=export.digest(wrapper), batch_sha256=export.digest('scripts/evaluate_robust_getup_batch.py'),
    original_proof_sha256=export.digest(R / 'historical_mean_compatibility_preflight001.json'),
    observations_checked=len(observations), new_PPO=0, simulator_started=False,
    qualification=None, created_unix=time.time()))
print(json.dumps(result))
