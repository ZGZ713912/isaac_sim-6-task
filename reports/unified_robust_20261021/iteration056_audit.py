"""Explicit position ABI: inherited state, archived parity and nonzero ONNX path."""
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
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration036_compare import read
from iteration044_selective_memory import equal
from wheeled_algo.standing_displacement import observation_contract
from wheeled_algo.rolling_velocity import TABLE_RELATIVE, TABLE_SHA256


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    old_path = RUNS/"quiet_velocity036/model_migrated.pt"
    new_path = RUNS/"quiet_position056_sealed/model_migrated.pt"
    old, new = [torch.load(p, map_location="cpu", weights_only=False) for p in (old_path, new_path)]
    before, _ = load_mean(old_path, migration_source_root=old_path.parent/"source")
    after, info = load_mean(new_path)
    assert all(equal(v, new["actor_state_dict"][k]) for k,v in old["actor_state_dict"].items())
    added = set(new["actor_state_dict"])-set(old["actor_state_dict"])
    assert added == {f"mlp.mean.quiet_velocity.{name}.weight" for name in ("mid", "low", "high")}
    assert all(new["actor_state_dict"][k].shape == (4,2) and not new["actor_state_dict"][k].count_nonzero() for k in added)
    for key in ("critic_state_dict",):
        assert equal(old[key], new[key])
    assert equal(old["optimizer_state_dict"]["state"], new["optimizer_state_dict"]["state"])
    for a,b in zip(old["optimizer_state_dict"]["param_groups"], new["optimizer_state_dict"]["param_groups"], strict=True):
        assert equal({k:v for k,v in a.items() if k != "params"}, {k:v for k,v in b.items() if k != "params"})
        if a["name"] == "actor":
            assert b["params"][:-3] == a["params"]
        else:
            assert equal(a,b)
    for key in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"):
        assert equal(old["infos"][key], info[key])
    semantic = {"version", "quiet_velocity_feedback", "quiet_velocity_feedback_mode", "standing_position_observation"}
    assert equal({k:v for k,v in old["infos"]["settings"].items() if k not in semantic},
                 {k:v for k,v in info["settings"].items() if k not in semantic})
    assert info["settings"]["standing_position_observation"] == observation_contract()
    assert info["source_sha256"][TABLE_RELATIVE] == TABLE_SHA256
    assert sha(new_path.parent/"source"/TABLE_RELATIVE) == TABLE_SHA256
    archive = old_path.parent/"source/src/wheeled_algo/robust_getup.py"
    assert sha(archive) == old["infos"]["source_sha256"]["src/wheeled_algo/robust_getup.py"]
    spec = importlib.util.spec_from_file_location("wheeled_algo._legacy056", archive)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kwargs = {n:getattr(before,n) for n in inspect.signature(module.RobustGetupActor).parameters if n != "parent" and hasattr(before,n)}
    kwargs.update(quiet_alpha=before.quiet_alpha.tolist(), quiet_wheel_bound=float(before.quiet_bound[-1]),
                  leg_action_clip=float(before.action_bound[0]))
    legacy = module.RobustGetupActor(deepcopy(before.parent), **kwargs)
    legacy.load_state_dict(before.state_dict(), strict=True)
    x = torch.from_numpy(np.concatenate([read(n)[3]["observations"][::20].reshape(-1,35).copy()
        for n in ("quiet_velocity_baseline036", "side_confirmation032")]))
    altered = x.clone()
    altered[:,14:16] = torch.from_numpy(np.random.default_rng(5601).uniform(-5.,5.,(len(x),2)).astype(np.float32))
    with torch.inference_mode():
        assert torch.equal(legacy(x), before(x))
        assert torch.equal(before(x), after(altered))
    with tempfile.TemporaryDirectory(prefix="position056-negative-") as folder:
        damaged = deepcopy(new)
        del damaged["infos"]["settings"]["standing_position_observation"]
        path = Path(folder)/"bad.pt"
        torch.save(damaged,path)
        try:
            load_mean(path)
        except ValueError as error:
            assert "exact versioned" in str(error)
        else:
            raise AssertionError("Missing changed-ABI contract accepted")
    # Exercise nonzero position-dependent actions only in a deep diagnostic copy.
    diagnostic = deepcopy(after)
    with torch.no_grad():
        for head in diagnostic.quiet_velocity.values():
            head.weight.copy_(torch.tensor([[.1,.2],[-.2,.1],[.3,.1],[-.1,-.3]]))
    graph = new_path.parent/"diagnostic_nonzero_position.onnx"
    torch.onnx.export(diagnostic, torch.zeros(1,35), str(graph), input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    model = onnx.load(graph)
    onnx.helper.set_model_props(model, dict(policy_hz="200", observation_dim="35", action_dim="6",
        standing_position_observation=json.dumps(observation_contract(),sort_keys=True),
        qualification="diagnostic_deepcopy_nonzero_weights_not_deliverable"))
    onnx.save(model,graph)
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph),sess_options=options,providers=["CPUExecutionProvider"])
    probes = altered[::max(1,len(x)//3072)][:3072].clone()
    probes[::7,3] = .333*5.
    with torch.inference_mode():
        expected = diagnostic(probes).numpy()
        nonzero_difference = float((diagnostic(probes)-after(probes)).abs().max())
    actual = np.concatenate([session.run(None,{"obs":row[None]})[0] for row in probes.numpy()])
    error = float(np.abs(actual-expected).max())
    assert error < 2e-4 and nonzero_difference > 1e-3
    result = dict(schema="position056-zero-migration-v1", checkpoint_sha256=sha(new_path), parent_sha256=sha(old_path),
        old_actor_critic_named_adam_rng_exact=True, legacy_archived_default_exact_inputs=len(x),
        zero_heads_actions_exact_with_changed_position_slots=len(x), new_trainable_values=24,
        changed_abi_negative_load_rejected=True, geometry_table_sha256=TABLE_SHA256,
        initial_clock=441, new_macro_updates=0, robust_consumed=1089,
        onnx_probes=len(probes), onnx_max_raw_action_error=error, diagnostic_nonzero_action_difference=nonzero_difference,
        diagnostic_graph=str(graph.relative_to(ROOT)), diagnostic_graph_sha256=sha(graph),
        diagnostic_weights_installed_in_pt=False, installed_in_rmcs=False, long_training_admitted=False)
    (REPORTS/"iteration056_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
