"""Audit legacy source parity, preserved checkpoint state and diagnostic ONNX."""
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration036_compare import read
from iteration044_selective_memory import equal
from wheeled_algo.robust_getup import quiet_gate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    parent_path = RUNS / "pilot027_rolling_feedback050/model_final.pt"
    target_path = RUNS / "quiet_all_prediction052/model_migrated.pt"
    old, new = [torch.load(p, map_location="cpu", weights_only=False) for p in (parent_path, target_path)]
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        assert equal(old[key], new[key]), key
    for key in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"):
        assert equal(old["infos"][key], new["infos"][key]), key
    settings = deepcopy(new["infos"]["settings"])
    assert settings.pop("quiet_leg_prediction_all_heights") is True
    settings["version"] = old["infos"]["settings"]["version"]
    assert settings == old["infos"]["settings"]
    assert new["infos"]["completed_updates"] == old["infos"]["completed_updates"] == 505
    before, _ = load_mean(parent_path, migration_source_root=parent_path.parent / "source")
    after, _ = load_mean(target_path)
    archive = parent_path.parent / "source/src/wheeled_algo/robust_getup.py"
    assert sha(archive) == old["infos"]["source_sha256"]["src/wheeled_algo/robust_getup.py"]
    spec = importlib.util.spec_from_file_location("wheeled_algo._legacy052", archive)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kwargs = {name: getattr(before, name) for name in inspect.signature(module.RobustGetupActor).parameters
              if name != "parent" and hasattr(before, name)}
    kwargs.update(quiet_alpha=before.quiet_alpha.tolist(),
                  quiet_wheel_bound=float(before.quiet_bound[-1]), leg_action_clip=float(before.action_bound[0]))
    legacy = module.RobustGetupActor(deepcopy(before.parent), **kwargs)
    legacy.load_state_dict(before.state_dict(), strict=True)
    chunks = [read(name)[3]["observations"][::20].reshape(-1, 35).copy() for name in
              ("quiet_velocity_baseline036", "side_confirmation032", "learned_rolling_static050")]
    x = torch.from_numpy(np.concatenate(chunks))
    with torch.inference_mode():
        original, bound, changed = legacy(x), before(x), after(x)
        assert torch.equal(original, bound)
        frozen = (quiet_gate(x).ravel() == 0) | (x[:, 3] / 5. >= .4)
        assert torch.equal(bound[frozen], changed[frozen])
    # Match the archived actor to actual050 native action inputs as well.
    data = read("learned_rolling_static050")[3]
    real = torch.from_numpy(data["observations"][::20].reshape(-1, 35).copy())
    recorded = data["actions"][::20].reshape(-1, 6)[:, [0, 1, 3, 4, 2, 5]]
    with torch.inference_mode():
        replay_error = float(np.max(np.abs(legacy(real).numpy() - recorded)))
    assert replay_error < 2e-4
    with tempfile.TemporaryDirectory(prefix="prediction052-negative-") as folder:
        bad = deepcopy(new)
        bad["infos"]["settings"]["quiet_leg_prediction_all_heights"] = "true"
        bad_path = Path(folder) / "bad.pt"
        torch.save(bad, bad_path)
        try:
            load_mean(bad_path)
        except ValueError as error:
            assert "explicit boolean" in str(error)
        else:
            raise AssertionError("Malformed prediction-scope contract was accepted")
    graph = target_path.parent / "diagnostic_prediction_scope.onnx"
    torch.onnx.export(after, torch.zeros(1, 35), str(graph), input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    model = onnx.load(graph)
    onnx.helper.set_model_props(model, dict(policy_hz="200", observation_dim="35", action_dim="6",
        source_checkpoint_sha256=sha(target_path), quiet_leg_prediction_scope="all_quiet_heights",
        quiet_leg_prediction_seconds="0.005", qualification="diagnostic_only_not_admitted"))
    onnx.save(model, graph)
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
    probes = x[::max(1, len(x)//3072)][:3072].clone()
    probes[::7, 3] = .333 * 5.
    with torch.inference_mode():
        expected = after(probes).numpy()
    actual = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in probes.numpy()])
    onnx_error = float(np.max(np.abs(expected - actual)))
    assert onnx_error < 2e-4
    result = dict(schema="all-height052-source-state-numeric-audit-v1", checkpoint_sha256=sha(target_path),
        parent_sha256=sha(parent_path), optimizer_updates=0, lineage_clock=505, robust_consumed=1025,
        actor_critic_Adam_RNG_exact=True, only_settings_change="quiet_leg_prediction_all_heights false/absent->true, version",
        prediction_duration_s=.005, legacy_archived_source_actions_exact_inputs=len(x),
        high_or_nonquiet_actions_exact_inputs=int(frozen.sum()),
        native050_action_replay_inputs=len(real), native050_max_raw_action_error=replay_error,
        malformed_scope_contract_rejected=True,
        onnx=dict(graph=str(graph.relative_to(ROOT)), sha256=sha(graph), input_shape=[1,35], output_shape=[1,6],
                  probes=len(probes), max_raw_action_error=onnx_error, scope="Diagnostic graph only; not deployed or admitted"),
        long_training_admitted=False, hardware_ready=False)
    (REPORTS / "iteration052_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
