#!/usr/bin/env python3
"""Export a fresh Wheelbipe-port actor and verify ONNX Runtime numerical parity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow-pilot",action="store_true",help="Permit diagnostic exports before the full budget")
    args = parser.parse_args()
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    source = args.checkpoint.resolve()
    receipt = json.loads((source.parent/"receipt.json").read_text())
    if receipt["status"]!="training_complete":
        raise ValueError("Export requires a completed training receipt")
    if not args.allow_pilot and receipt["completed_updates"]!=20000:
        raise ValueError("Final export requires the complete original 20000-update budget")
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest()!=receipt["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the completed training receipt")
    data = torch.load(source,map_location="cpu",weights_only=False)
    state = data["model_state_dict"]
    # Same ELU actor from the original ordinary V14 PPO configuration. Infer
    # dimensions from the authenticated state, then check the source topology.
    layers = []
    selected = sorted((int(k.split('.')[1]),k,v) for k,v in state.items() if k.startswith("actor.") and k.endswith(".weight"))
    if [(v.shape[1],v.shape[0]) for _,_,v in selected]!=[(35,256),(256,128),(128,64),(64,6)]:
        raise ValueError("Checkpoint is not the original V14 ELU 35→256→128→64→6 actor")
    for i,(_,key,weight) in enumerate(selected):
        layer = torch.nn.Linear(weight.shape[1],weight.shape[0])
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(state[key.removesuffix("weight")+"bias"])
        layers.append(layer)
        if i<len(selected)-1:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).eval()
    # Source empirical normalization is disabled. Refuse a different policy
    # instead of silently dropping an inferred preprocessing layer.
    normalization = {k:v for k,v in state.items() if "obs_normalizer" in k}
    if normalization:
        raise ValueError("This source actor unexpectedly contains observation normalization")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    target = args.output/"policy.onnx"
    torch.onnx.export(actor,torch.zeros(1,35),str(target),input_names=["obs"],output_names=["actions"],
        dynamic_axes={"obs":{0:"batch"},"actions":{0:"batch"}},opset_version=17,dynamo=False)
    onnx.checker.check_model(str(target))
    session = ort.InferenceSession(str(target),providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(107)
    sample = rng.uniform(-5.,5.,size=(1024,35)).astype(np.float32)
    with torch.inference_mode():
        expected = actor(torch.from_numpy(sample)).numpy()
    actual = session.run(None,{"obs":sample})[0]
    error = float(np.max(np.abs(actual-expected)))
    if not np.allclose(actual,expected,rtol=1e-5,atol=1e-5):
        raise ValueError(f"ONNX actor parity failed: {error}")
    contract = dict(schema="wheelbipe-v202-onnx-contract-v1",checkpoint=str(source),
        checkpoint_sha256=hashlib.sha256(content).hexdigest(),
        onnx_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),completed_updates=receipt["completed_updates"],
        source_snapshot_sha256=receipt["source_snapshot_sha256"],
        asset_manifest_sha256=receipt["asset_manifest_sha256"],
        native=receipt["native"],actor="deterministic Gaussian mean; no teacher or previous policy",
        obs_dim=35,action_dim=6,opset=17,parity_samples=len(sample),max_absolute_error=error,
        behavior_qualified=False,hardware_mapping_verified=False,
        observation_order="source V14 command(3), height(1), gyro(3), gravity(3), leg-relative-position(4), muted-wheel-position(2), joint-velocity(6), previous-action(6), control-mode(7)",
        preprocessing="original V14 clip/scale and v2 sensor model; see params/env.yaml", 
        postprocessing="leg defaults + 0.5*action; wheels 10*action rad/s; v202 PD, spring compensation, limits and measured wheel response required")
    # The textual order above is checked against dimensions rather than being
    # used as executable preprocessing; exact per-field scales remain in YAML.
    (args.output/"contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    print(json.dumps(dict(path=str(target),max_absolute_error=error,completed_updates=receipt["completed_updates"]),indent=2))


if __name__=="__main__":
    main()
