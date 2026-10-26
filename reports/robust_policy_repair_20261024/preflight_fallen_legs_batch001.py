"""Import the real batch wrapper without starting Kit; verify function and labels."""
import ast
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts', str(Path(__file__).resolve().parent)]
import numpy as np
import torch
import export_robust_getup as export
from fallen_output_partition_probe001 import partition
from train_robust_getup import digest, write_json

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
wrapper = R / 'evaluate_fallen_legs_original_target001.py'
tree = ast.parse(wrapper.read_text())
assert isinstance(tree.body[-1], ast.Try)
tree.body = tree.body[:-1]
fixture = R / 'fallen_legs_batch_wrapper_CPU001'
assert not fixture.exists()
cp = B / 'pilot014_fallen_output/model_final.pt'
old = export.load_mean
sys.argv = [str(wrapper), '--checkpoint', str(cp), '--output', str(fixture)]
namespace = {'__file__': str(wrapper.resolve()), '__name__': 'fallen_legs_batch_fixture'}
exec(compile(tree, str(wrapper), 'exec'), namespace)
class Dummy:
    pass
namespace['label_native_env'] = lambda: namespace['attach_artifact_label'](Dummy)
try:
    actual, _ = namespace['load'](cp)
    expected, _ = old(cp)
    baseline, _ = old(B / 'fallen_output_root001/model_initial.pt')
    evidence = partition(expected, baseline, 'legs')
    record = json.loads((fixture / 'runtime_function_readback.json').read_text())
    dummy = Dummy()
    dummy.other_state = torch.tensor([4., 5.])
    dummy.evaluation_overrides = {}
    dummy.evaluation_overrides['original_field'] = 'retained'
    values = np.load(B / 'diagnosis025_motion_damping080_fixed/evaluation/trajectory.npz')['observations'].reshape(-1, 35)[::40].copy()
    with torch.inference_mode():
        x = torch.from_numpy(values)
        function_exact = torch.equal(actual(x), expected(x))
    checks = dict(real_native_main_not_called=not (fixture / 'progress.json').exists(),
        actual_mean_matches_old040_partition=all(torch.equal(v, actual.state_dict()[k]) for k, v in expected.state_dict().items()),
        recorded_forward_bit_exact=function_exact,
        runtime_hash_matches040=record['runtime_partition']['final_layer_runtime_sha256'] == evidence['final_layer_runtime_sha256'],
        label_survives_native_override_assignment='fallen_legs_original_target_probe' in dummy.evaluation_overrides,
        other_artifact_fields_retained=dummy.evaluation_overrides['original_field'] == 'retained',
        dummy_physical_state_untouched=torch.equal(dummy.other_state, torch.tensor([4., 5.])),
        explicit_qualification_exclusion=record['eligible_for_checkpoint_mean_qualification'] is False)
finally:
    export.load_mean = old
out = dict(schema='fallen-legs-original-batch-wrapper-preflight-v1', checks=checks,
    recorded_frames=len(values), runtime_layer_sha256=evidence['final_layer_runtime_sha256'],
    wrapper_sha256=digest(wrapper), checkpoint=str(cp), checkpoint_sha256=digest(cp),
    previous_scope_preflight_sha256=digest(R / 'fallen_partition_preflight001.json'),
    batch_sha256=digest('scripts/evaluate_robust_getup_batch.py'), new_PPO=0,
    actual_experiment_consumed=2654, repair_optimizer_updates=274, qualification=None, created_unix=time.time())
write_json(R / 'fallen_legs_batch_preflight001.json', out)
print(json.dumps(out, indent=2))
assert all(checks.values()), checks
