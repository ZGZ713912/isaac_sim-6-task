"""Execution equivalence, replay storage and failure guards; zero PPO."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.encoder_spring_feedforward import EncoderGasSpringFeedforward, resolve
from wheeled_tasks.chassis.training_gas_spring import resolve as resolve_spring
from wheeled_tasks.chassis.tensor_cuda_graph import TensorCudaGraph
from wheeled_tasks.chassis.prone_motion_performance import RouteReadCache

ROOT = Path(__file__).resolve().parents[1]


def model():
    plan = json.loads((ROOT/'contracts/wheel_leg_prone_motion_v1.json').read_text())
    cfg = json.loads((ROOT/'contracts/scratch_standing_base_mechanism_v2.json').read_text())
    spec_path = ROOT/cfg['asset_directory']/'model_spec.json'
    spring = resolve_spring(ROOT,plan['gas_spring_model'],json.loads(spec_path.read_text()))
    return EncoderGasSpringFeedforward(resolve(ROOT,plan['gas_spring_feedforward'],spring,spec_path))


def test_cpu_forward_preserves_all_pure_outputs_and_invalid_input_guard():
    ff = model()
    q = torch.tensor([[-.42,.14,.42,-.14]],dtype=torch.float64)
    expected, values = ff._evaluate(q)
    torch.testing.assert_close(ff(q),expected,atol=0.,rtol=0.)
    assert set(values) == set(ff.last)
    for name,value in values.items():
        torch.testing.assert_close(ff.last[name],value,atol=0.,rtol=0.)
    for bad in (float('nan'),float('inf'),-float('inf')):
        with pytest.raises(ValueError,match='Nonfinite'):
            ff(q+bad)


def test_graph_rejects_cpu_instead_of_silently_changing_execution():
    with pytest.raises(ValueError,match='CUDA inference'):
        TensorCudaGraph(lambda x:x+1,torch.ones(256,4))


def test_route_cache_invalidates_native_reads_resets_and_new_environment():
    env = SimpleNamespace(root_kinematics=SimpleNamespace(read_epoch=1),prone_reset_generation=0)
    calls = []
    def native_route():
        calls.append(object())
        return calls[-1]
    cache = RouteReadCache(env,native_route)
    first = cache()
    assert cache() is first and len(calls) == 1
    env.root_kinematics.read_epoch += 1
    second = cache()
    assert second is not first and cache() is second and len(calls) == 2
    env.prone_reset_generation += 1
    assert cache() is not second and len(calls) == 3
    assert RouteReadCache(env,native_route)() is not cache() and len(calls) == 4


def test_role_metadata_refreshes_when_an_evaluation_changes_roles():
    from wheeled_tasks.chassis import prone_motion as task
    # Importing the native environment would require a running Kit application.
    # Execute its actual independent metadata method for this CPU regression.
    tree = ast.parse((ROOT/'src/wheeled_tasks/chassis/prone_motion_env.py').read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'ProneMotionEnv')
    method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == '_refresh_role_tensors')
    namespace = {'torch':torch,'task':task}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'prone_motion_env.py','exec'),namespace)
    refresh = namespace['_refresh_role_tensors']
    ROLES = task.ROLES
    env = SimpleNamespace(course_roles=list(ROLES)*2,_role_signature=None,device='cpu')
    refresh(env)
    original = env._role_index
    refresh(env)
    assert env._role_index is original
    assert env._stand_roles.sum() == 2
    env.course_roles[0] = 'stand'
    refresh(env)
    assert env._stand_roles.sum() == 3 and env._role_index is not original


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA execution equivalence')
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_replay_dense_compensation_is_exact_and_keeps_outputs_independent(dtype):
    eager = model().to(device='cuda',dtype=dtype)
    replay = deepcopy(eager)
    generator = torch.Generator(device='cuda').manual_seed(261029)
    q = torch.empty(256,4,device='cuda',dtype=dtype)
    with torch.inference_mode():
        for side in range(2):
            x = eager.get(side,'delta_rad')
            q[:,2*side] = torch.linspace(-4.,4.,256,device='cuda',dtype=dtype)
            q[:,2*side+1] = q[:,2*side]+torch.linspace(x[0]-.002,x[-1]+.002,256,device='cuda',dtype=dtype)
        replay.tensor_replay = TensorCudaGraph(replay._evaluate,q)
        previous = None
        for i in range(6):
            now = q+i*6.283185307179586
            pd = torch.randn(256,4,device='cuda',dtype=dtype,generator=generator)*60.
            expected = eager.motor_efforts(pd,now)
            actual = replay.motor_efforts(pd,now)
            torch.testing.assert_close(actual,expected,atol=0.,rtol=0.)
            assert set(eager.last) == set(replay.last)
            for key,value in eager.last.items():
                torch.testing.assert_close(replay.last[key],value,atol=0.,rtol=0.)
            if previous is not None:
                # Replaying a graph cannot mutate a recorded preceding sample.
                for key,value in previous[0].items():
                    torch.testing.assert_close(value,previous[1][key],atol=0.,rtol=0.)
            previous = (dict(replay.last),{k:v.clone() for k,v in replay.last.items()})
        with pytest.raises(ValueError,match='Nonfinite'):
            replay(q+float('nan'))
        with pytest.raises(ValueError,match='shape/device/dtype'):
            replay(q[:1])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA execution equivalence')
def test_replay_full_encoder_geometry_matches_validity_and_derivatives():
    from wheeled_algo.encoder_leg_geometry import EncoderLegGeometry
    from wheeled_tasks.chassis.standing_geometry107 import table_path
    geom = EncoderLegGeometry(json.loads(table_path().read_text())).to('cuda')
    ff = model().to('cuda',dtype=torch.float32)
    with torch.inference_mode():
        q = torch.zeros(256,4,device='cuda')
        for side in range(2):
            x = ff.get(side,'delta_rad')
            q[:,2*side+1] = torch.linspace(x[0]-.002,x[-1]+.002,256,device='cuda')
        dq = torch.linspace(-25.,25.,1024,device='cuda').reshape(256,4)
        replay = TensorCudaGraph(geom,q,dq)
        for shift in (0.,.1,1.,6.283185307179586):
            expected, actual = geom(q+shift,dq), replay(q+shift,dq)
            for key,value in expected.items():
                torch.testing.assert_close(actual[key],value,atol=0.,rtol=0.)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA execution equivalence')
def test_physical_spring_replay_preserves_measured_knots_extensions_and_guards():
    from wheeled_tasks.chassis.training_gas_spring import TrainingGasSpring
    plan = json.loads((ROOT/'contracts/wheel_leg_prone_motion_v1.json').read_text())
    cfg = json.loads((ROOT/'contracts/scratch_standing_base_mechanism_v2.json').read_text())
    spec = json.loads((ROOT/cfg['asset_directory']/'model_spec.json').read_text())
    settings = resolve_spring(ROOT,plan['gas_spring_model'],spec)
    eager,replay = [TrainingGasSpring(settings,'cuda') for _ in range(2)]
    # Include all measured knots, unmeasured ends, and numerical overtravel.
    samples = torch.linspace(-.002,.082,512,device='cuda').reshape(256,2)
    with torch.inference_mode():
        replay.tensor_replay = TensorCudaGraph(replay._evaluate,samples)
    # Native startup/reset may query without a surrounding inference_mode.
    for shift in (0.,.00001,-.00002):
        expected,actual = eager.force(samples+shift),replay.force(samples+shift)
        torch.testing.assert_close(actual,expected,atol=0.,rtol=0.)
        torch.testing.assert_close(replay.last_outside,eager.last_outside,atol=0.,rtol=0.)
    with pytest.raises(ValueError,match='Nonfinite'):
        replay.force(samples+float('inf'))
    with pytest.raises(ValueError,match='shape/device/dtype'):
        replay.force(samples.requires_grad_())
