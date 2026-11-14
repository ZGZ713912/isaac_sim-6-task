#!/usr/bin/env python3
"""Export a fresh Wheelbipe-port actor and verify ONNX Runtime numerical parity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from wheeled_tasks.wheelbipe_port.training_contract import check_constraint_gap


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
    gap_limit = receipt.get("constraint_gap_limit_m",.003)
    check_constraint_gap(receipt["max_constraint_gap_m"],gap_limit,"Completed training")
    if not args.allow_pilot and receipt["completed_updates"]!=20000:
        raise ValueError("Final export requires the complete original 20000-update budget")
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest()!=receipt["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the completed training receipt")
    data = torch.load(source,map_location="cpu",weights_only=False)
    state = data["model_state_dict"]
    from wheeled_tasks.wheelbipe_port.motion import observation_contract,actor_from_state_dict
    from wheeled_tasks.wheelbipe_port.training_profile import specification
    profile = receipt.get("arguments",{}).get("training_profile","legacy")
    observation_abi = observation_contract(None if profile == "legacy" else specification(profile))
    if receipt["native"]["actor_dim"] != observation_abi["actor_dim"]:
        raise ValueError("Receipt actor dimension differs from its observation history contract")
    actor_dim = observation_abi["actor_dim"]
    actor = actor_from_state_dict(state,actor_dim)
    # Source empirical normalization is disabled. Refuse a different policy
    # instead of silently dropping an inferred preprocessing layer.
    normalization = {k:v for k,v in state.items() if "obs_normalizer" in k}
    if normalization:
        raise ValueError("This source actor unexpectedly contains observation normalization")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    target = args.output/"policy.onnx"
    torch.onnx.export(actor,torch.zeros(1,actor_dim),str(target),input_names=["obs"],output_names=["actions"],
        dynamic_axes={"obs":{0:"batch"},"actions":{0:"batch"}},opset_version=17,dynamo=False)
    onnx.checker.check_model(str(target))
    session = ort.InferenceSession(str(target),providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(107)
    sample = rng.uniform(-5.,5.,size=(1024,actor_dim)).astype(np.float32)
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
        obs_dim=actor_dim,observation_contract=observation_abi,action_dim=6,opset=17,parity_samples=len(sample),max_absolute_error=error,
        constraint_gap_limit_m=gap_limit,training_constraint_gap_peak_m=receipt["max_constraint_gap_m"],
        constraint_gap_hard_limit_enabled=gap_limit is not None,
        diagnostic_sampling_scope=receipt.get("diagnostic_sampling_scope","all physics steps; historical receipt"),
        behavior_qualified=False,hardware_mapping_verified=False,
        observation_order="each 35D frame: source V14 command(3), height(1), gyro(3), gravity(3), leg-relative-position(4), muted-wheel-position(2), joint-velocity(6), previous-action(6), control-mode(7); flatten frames newest first",
        preprocessing="original V14 clip/scale and v2 sensor model, then observation_contract history; repeat first frame on reset and append once per 20ms policy tick; see params/env.yaml",
        postprocessing="leg defaults + 0.5*action; wheels 10*action rad/s; v202 PD, physical spring, limits and measured wheel response required; cancellation FF is selected by native.gas_cancellation_feedforward",
        gas_cancellation_feedforward=receipt["native"].get("gas_cancellation_feedforward",True),
        deployment_requires_matching_low_level_control=True)
    # The textual order above is checked against dimensions rather than being
    # used as executable preprocessing; exact per-field scales remain in YAML.
    (args.output/"contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    print(json.dumps(dict(path=str(target),max_absolute_error=error,completed_updates=receipt["completed_updates"]),indent=2))


if __name__=="__main__":
    main()
