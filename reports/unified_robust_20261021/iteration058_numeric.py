"""Trained position feedback: exact nonquiet actions and run-only ONNX parity."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT/"runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts"),str(REPORTS)]
from export_robust_getup import load_mean
from iteration036_compare import read
from wheeled_algo.robust_getup import quiet_gate
from wheeled_algo.standing_displacement import observation_contract


def main():
    torch.set_num_threads(1)
    checkpoint = RUNS/"pilot031_standing_position058/model_final.pt"
    before,_ = load_mean(RUNS/"quiet_position056_sealed/model_migrated.pt")
    after,info = load_mean(checkpoint)
    x = torch.from_numpy(np.concatenate([read(n)[3]["observations"][::20].reshape(-1,35).copy()
        for n in ("side_confirmation032","position_zero_static056")]))
    altered = x.clone()
    altered[:,14:16] = torch.from_numpy(np.random.default_rng(5701).uniform(-5.,5.,(len(x),2)).astype(np.float32))
    with torch.inference_mode():
        frozen = quiet_gate(x).ravel() == 0
        assert torch.equal(before(x)[frozen],after(altered)[frozen])
    graph = checkpoint.parent/"diagnostic_trained_position.onnx"
    torch.onnx.export(after,torch.zeros(1,35),str(graph),input_names=["obs"],output_names=["actions"],
        opset_version=17,dynamo=False)
    model = onnx.load(graph)
    onnx.helper.set_model_props(model,dict(policy_hz="200",observation_dim="35",action_dim="6",
        source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        standing_position_observation=json.dumps(observation_contract(),sort_keys=True),
        qualification="diagnostic_not_admitted_no_RMCS_preprocessor"))
    onnx.save(model,graph)
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph),sess_options=options,providers=["CPUExecutionProvider"])
    probes = altered[::max(1,len(x)//3072)][:3072].clone()
    probes[::7,3] = .333*5.
    with torch.inference_mode():
        expected = after(probes).numpy()
    actual = np.concatenate([session.run(None,{"obs":row[None]})[0] for row in probes.numpy()])
    error = float(np.abs(actual-expected).max())
    assert error < 2e-4
    result = dict(schema="position058-numeric-v1",checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        actual_inputs=len(x),nonquiet_actions_exact_even_with_arbitrary_positions=int(frozen.sum()),
        learned_feedback_weights={k:v.tolist() for k,v in after.quiet_velocity.state_dict().items()},
        onnx_probes=len(probes),onnx_max_raw_error=error,diagnostic_graph=str(graph.relative_to(ROOT)),
        diagnostic_graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest(),lineage_clock=info["completed_updates"],
        new_macro_updates=0,new_delivery=False,long_training_admitted=False)
    (REPORTS/"iteration058_numeric.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
