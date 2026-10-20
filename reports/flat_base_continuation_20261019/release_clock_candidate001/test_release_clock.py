"""Release evidence follows the actual trained clock and keeps deployment1000Hz."""
import importlib.util
import json
from pathlib import Path
import pytest
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

export=module('flat_release_clock_candidate',HERE/'export_flat_base.py')
export.ROOT=ROOT
fixture=module('flat_release_clock_existing_fixtures',ROOT/'tests/test_flat_release_evidence.py')

def actual_evidence(tmp_path):
    primary,manifest,_=fixture.evidence(tmp_path)
    plan=json.loads((ROOT/'reports/flat_base_continuation_20261019/wheel_authority_candidate001/plan_wheel_from1280.json').read_text())
    assert plan['physics_dt']==.001
    primary['physics_hz']=1000
    fixture.mutate(tmp_path,manifest,'training_envelope',lambda data:data.update(physics_hz=1000))
    return primary,manifest,plan

def test_historical200_training_clock_keeps_all_eleven_cohorts(tmp_path):
    primary,manifest,plan=fixture.evidence(tmp_path)
    assert plan['physics_dt']==.005
    assert len(export.validate_release(primary,manifest,plan,'candidate',tmp_path))==10

def test_actual1000_training_plan_accepts_matching_full_coverage(tmp_path):
    assert len(export.validate_release(*actual_evidence(tmp_path),'candidate',tmp_path))==10

@pytest.mark.parametrize('cohort',('training_basic','training_envelope'))
def test_actual1000_plan_rejects_historical200_evidence(tmp_path,cohort):
    primary,manifest,plan=actual_evidence(tmp_path)
    if cohort=='training_basic':
        primary['physics_hz']=200
    else:
        fixture.mutate(tmp_path,manifest,cohort,lambda data:data.update(physics_hz=200))
    with pytest.raises(ValueError,match='different model, clock or seed'):
        export.validate_release(primary,manifest,plan,'candidate',tmp_path)

def test_actual_composed_checkpoint_plan_is_bound_to1000Hz():
    state=torch.load(ROOT/'runs/flat_base_20261019/pilot025/model_001280_wheel_authority_composed002.pt',map_location='cpu',weights_only=True)
    plan=export.read_checkpoint_plan(ROOT,state['infos'])
    assert plan['physics_dt']==.001 and plan['policy_dt']==.02
