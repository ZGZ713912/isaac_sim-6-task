"""Diagnostic ONNX parity after joint quiet tuning, without a deployment export."""
import hashlib
import json
from pathlib import Path
import sys

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
from wheeled_algo.robust_getup import quiet_gate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    directory = RUNS / "pilot032_joint_quiet064"
    progress = json.loads((directory / "progress.json").read_text())
    assert progress["status"] == "completed" and progress["new_optimizer_updates"] == 32
    checkpoint = directory / "model_final.pt"
    before, _ = load_mean(RUNS / "quiet_position_reference062/model_migrated.pt")
    after, info = load_mean(checkpoint)
    digest = sha(checkpoint)
    arrays = [read(name)[3]["observations"][::20].reshape(-1, 35).copy()
              for name in ("position_reference_static062", "position_wheels_static063", "side_confirmation032")]
    x = torch.from_numpy(np.concatenate(arrays))
    nonquiet = quiet_gate(x).ravel() == 0
    with torch.inference_mode():
        modified = x.clone(); modified[:, 14] = 5.; modified[:, 15] = -5.
        assert torch.equal(before(modified)[nonquiet], after(modified)[nonquiet])
        maximum_quiet_change = float((before(x) - after(x)).abs().max())
    graph = directory / "diagnostic_joint_quiet064.onnx"
    torch.onnx.export(after, torch.zeros(1, 35), str(graph), input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    model = onnx.load(graph)
    onnx.helper.set_model_props(model, dict(policy_hz="200", physics_hz="1000", observation_dim="35", action_dim="6",
        source_checkpoint_sha256=digest, qualification="diagnostic_only_not_admitted",
        observation_contract=json.dumps(info["settings"]["standing_position_observation"], sort_keys=True)))
    onnx.save(model, graph)
    onnx.checker.check_model(model)
    options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
    probes = x[::max(1, len(x) // 3072)][:3072].clone()
    probes[::7, 3] = .333 * 5.
    with torch.inference_mode():
        expected = after(probes).numpy()
    actual = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in probes.numpy()])
    difference = float(np.max(np.abs(expected - actual)))
    assert difference < 2e-4 and sha(checkpoint) == digest
    result = dict(schema="joint064-diagnostic-numeric-v1", checkpoint_sha256=digest,
                  actual_inputs=len(x), nonquiet_actions_bitwise_exact=int(nonquiet.sum()),
                  maximum_quiet_raw_change=maximum_quiet_change, graph=str(graph.relative_to(ROOT)),
                  graph_sha256=sha(graph), onnx_probes=len(probes), max_raw_action_error=difference,
                  input_shape=[1, 35], output_shape=[1, 6], new_macro_updates=0,
                  scope="Diagnostic graph only; position-observation preprocessing not yet installed in RMCS",
                  installed_in_rmcs=False, long_training_admitted=False)
    (REPORTS / "iteration064_numeric.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
