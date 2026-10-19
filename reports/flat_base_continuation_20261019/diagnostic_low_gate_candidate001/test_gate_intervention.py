"""Same-input separation and provenance checks, with no simulator or optimizer."""
from copy import deepcopy
import ast
from pathlib import Path
import sys
import pytest
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'))
from wheeled_algo.command_residual import mean_from_checkpoint
text=(HERE/'evaluate.py').read_text()
function=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='diagnostic_ablation')
ns={};exec(compile(ast.Module(body=[function],type_ignores=[]),'<gate-diagnostic>','exec'),ns)
intervene=ns['diagnostic_ablation']
torch.set_num_threads(1)


@pytest.fixture(scope='module')
def model():
    parent=torch.load(ROOT/'runs/flat_base_20261019/pilot023/model_001248.pt',map_location='cpu',weights_only=True)
    return mean_from_checkpoint(parent,ROOT/'models/wheel_leg_flat_base_20261018/source.onnx').eval()


def obs(vx=0.,yaw=0.):
    g=torch.Generator().manual_seed(191019)
    x=torch.randn(32,35,generator=g)*.03;x[:,0]=vx;x[:,1]=0;x[:,2]=yaw;x[:,3]=1.525
    x[:,9]=-1;x[:,28:]=0;x[:,28]=1
    return x


def test_full_control_retains_exact_mean(model):
    copy=intervene(deepcopy(model),'full');x=obs(-.5)
    assert torch.equal(copy(x),model(x))


def test_all_weights_and_buffers_remain_exact(model):
    copy=intervene(deepcopy(model),'low_gate_floor')
    assert copy.state_dict().keys()==model.state_dict().keys()
    assert all(torch.equal(v,copy.state_dict()[n]) for n,v in model.state_dict().items())


def test_all_stationary_actions_exact_for_arbitrary_sensor_values(model):
    copy=intervene(deepcopy(model),'low_gate_floor');x=obs()
    assert torch.equal(copy(x),model(x))


@pytest.mark.parametrize('vx,yaw',[(3.,0.),(-3.,0.),(0.,4.),(0.,-4.),(1.5,0.),(0.,3.)])
def test_fast_endpoint_actions_exact_for_arbitrary_sensor_values(model,vx,yaw):
    copy=intervene(deepcopy(model),'low_gate_floor');x=obs(vx,yaw)
    assert torch.equal(copy(x),model(x))


def test_low_gate_is_full_without_changing_command_mean(model):
    copy=intervene(deepcopy(model),'low_gate_floor');x=obs(-.5)
    expected=torch.stack((x[:,0],-x[:,0]),dim=1)/.6;x[:,20:22]=expected
    assert torch.equal(copy.tracking_feedback_gate(x),torch.ones(32,1))
    assert torch.equal(copy.command(x),model.command(x))
    assert not torch.equal(copy(x),model(x))
    assert (copy.feedback_actions(x).abs()<=copy.output_scale+1e-7).all()


def test_gate_floor_is_continuous_at_both_blend_endpoints(model):
    copy=intervene(deepcopy(model),'low_gate_floor')
    for vx in (.75,1.5):
        values=[]
        for delta in (-1e-6,1e-6):
            x=obs(vx+delta);x[:,20]=x[:,0]/.6;x[:,21]=-x[:,0]/.6
            values.append(copy.tracking_feedback_gate(x))
        assert (values[0]-values[1]).abs().max()<1e-5


def test_unknown_intervention_is_rejected(model):
    with pytest.raises(ValueError):intervene(deepcopy(model),'unknown')
