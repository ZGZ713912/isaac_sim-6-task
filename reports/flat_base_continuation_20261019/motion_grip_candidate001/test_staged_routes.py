"""CPU integration of proposed reader/plan routes in an isolated source tree."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
STAGED = Path(__file__).resolve().parent / 'staged_root'
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(STAGED / 'src'))
from test_static_grip_adaptation import means, observations
from test_guarded_feedback_residual import write_source
from wheeled_algo.command_residual import mean_from_checkpoint
from wheeled_algo.motion_grip import MotionGripActor, KIND
from wheeled_algo.static_grip import KIND as PARENT_KIND
from wheeled_tasks.chassis.flat_base_continuation import read_plan, write_bundle
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.preflight import design_preflight


def test_routes_use_staged_sources():
    assert '/staged_root/' in mean_from_checkpoint.__code__.co_filename
    assert '/staged_root/' in read_plan.__code__.co_filename


def test_checkpoint_read_uses_authenticated_v7_static_reference(means, tmp_path):
    parent = means[1]
    with torch.no_grad():
        parent.static_feedback[2].bias.add_(.015)
    recalled = means[2]
    source = tmp_path / 'source.onnx'
    write_source(source, SimpleNamespace(mlp=parent.command.base))
    actor_state = lambda mean: {'mlp.' + name: value for name, value in mean.state_dict().items()}
    parent_path = tmp_path / 'parent.pt'
    torch.save(dict(infos=dict(actor_adaptation=PARENT_KIND,
        static_feedback_recall=dict(checkpoint=str(recalled), sha256=hashlib.sha256(recalled.read_bytes()).hexdigest())),
        actor_state_dict=actor_state(parent)), parent_path)
    reference = dict(checkpoint=str(parent_path), sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest())
    child = MotionGripActor(parent, reference['sha256'])
    with torch.no_grad():
        child.fast_feedback[2].bias.add_(.02)
    state = dict(infos=dict(actor_adaptation=KIND, motion_grip_parent=reference),
                 actor_state_dict=actor_state(child))
    loaded = mean_from_checkpoint(state, source)
    obs = observations()
    assert torch.equal(loaded(obs), child(obs))
    assert type(loaded) is MotionGripActor
    bad = deepcopy(state)
    bad['actor_state_dict']['mlp.static_feedback.2.bias'].add_(.001)
    with pytest.raises(ValueError, match='Frozen adapted static'):
        mean_from_checkpoint(bad, source)
    bad = deepcopy(state)
    bad['infos']['motion_grip_parent']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='parent checkpoint changed'):
        mean_from_checkpoint(bad, source)
    bad = deepcopy(state)
    bad['infos']['motion_grip_parent'] = None
    with pytest.raises(ValueError, match='authenticated v7 parent'):
        mean_from_checkpoint(bad, source)


@pytest.mark.parametrize('evaluation', [False, True])
def test_new_plan_reaches_cpu_preflight(evaluation, tmp_path):
    plan_file = 'reports/flat_base_continuation_20261019/motion_grip_candidate001/proposed_plan.json'
    count = 32 if evaluation else 4096
    plan = read_plan(ROOT, plan_file)
    config, seed = write_bundle(ROOT, tmp_path / 'config', count, evaluation=evaluation, plan_file=plan_file)
    assert config['contact_material']['ground_static_friction_range'] == [.2, .5]
    assert config['physics_dt'] == .001 and config['policy_dt'] == .02
    assert config.get('actor_adaptation') == (None if evaluation else KIND)
    assert config.get('motion_grip_parent') == (None if evaluation else plan['motion_grip_parent'])
    receipt = design_preflight(config, load_chassis_control(ROOT / config['control_math_source']),
        ROOT, stage='speed', role='quick' if evaluation else 'train', num_envs=count, seed=seed)
    assert receipt['status'] == 'static_passed'


def test_plan_rejects_reference_changed_or_other_kind(tmp_path):
    path = ROOT / 'reports/flat_base_continuation_20261019/motion_grip_candidate001/proposed_plan.json'
    plan = json.loads(path.read_text())
    bad_path = tmp_path / 'bad_plan.json'
    bad = deepcopy(plan)
    bad['motion_grip_parent']['sha256'] = '0' * 64
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='exact declared v7'):
        read_plan(ROOT, bad_path)
    bad = deepcopy(plan)
    bad['actor_adaptation'] = PARENT_KIND
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='only valid for the declared v8'):
        read_plan(ROOT, bad_path)
