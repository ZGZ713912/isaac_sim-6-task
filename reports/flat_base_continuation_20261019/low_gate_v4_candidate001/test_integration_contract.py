from pathlib import Path
from copy import deepcopy
import json
import torch
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.low_demand_feedback import LowDemandFeedbackActor, KIND
from wheeled_tasks.chassis.flat_base_continuation import read_plan,write_bundle
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.preflight import design_preflight
ROOT=Path(__file__).resolve().parents[3]
PLAN="reports/flat_base_continuation_20261019/low_gate_v4_candidate001/plan_low_gate.json"

def test_declared_v4_worker_and_separate_design_are_both_accepted(tmp_path):
    plan=read_plan(ROOT,PLAN)
    worker,seed=write_bundle(ROOT,tmp_path/"config",4096,plan_file=PLAN)
    assert worker["actor_adaptation"]==plan["actor_adaptation"]==KIND
    result=design_preflight(worker,load_chassis_control(ROOT/plan["control_file"]),ROOT,
        stage="speed",role="train",num_envs=4096,seed=seed)
    assert result["status"]=="static_passed"

def test_v4_checkpoint_loader_retains_all_actual_parent_tensors():
    parent=torch.load(ROOT/"runs/flat_base_20261019/pilot024/model_001264.pt",map_location="cpu",weights_only=True)
    source=ROOT/"models/wheel_leg_flat_base_20261018/source.onnx"
    original=mean_from_checkpoint(parent,source)
    child=LowDemandFeedbackActor(original)
    same=deepcopy(parent);same["infos"]["actor_adaptation"]=KIND
    restored=mean_from_checkpoint(same,source)
    assert all(torch.equal(v,restored.state_dict()[n]) for n,v in original.state_dict().items())
    x=torch.randn(32,35);x[:,0]*=3;x[:,2]*=4
    assert torch.equal(child(x),restored(x))
