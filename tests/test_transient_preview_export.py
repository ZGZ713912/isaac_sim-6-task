from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from export_flat_speed_preview import validate_comparison
from flat_transient_metrics import RETENTION, STEADY, TRANSIENT


def response():
    return dict(status='evaluated', probe_mode='checkpoint_compare', full_envelope=True,
        replicas=4, qualified=False, horizon_seconds=16., transient_response_protocol=dict(segment_seconds=4.),
        case_names=list(RETENTION + STEADY + TRANSIENT), policy_hz=200, physics_hz=1000,
        heights=[.305, .42], ground_friction=[.2, .5])


def test_response_export_is_explicit_and_does_not_regrade_historical_comparisons():
    validate_comparison(response(), True)
    with pytest.raises(ValueError, match='explicit response mode'):
        validate_comparison(response())
    old = {**response(), 'horizon_seconds': 32.}
    old.pop('transient_response_protocol')
    validate_comparison(old)
    with pytest.raises(ValueError, match='prospective'):
        validate_comparison(old, True)


@pytest.mark.parametrize('field,value', [('replicas', 2), ('horizon_seconds', 8.),
    ('case_names', ['forward_3']), ('policy_hz', 50), ('ground_friction', [.5]),
    ('diagnostic_inference_override', {'alpha': .25})])
def test_partial_response_evidence_cannot_back_the_export(field, value):
    report = deepcopy(response())
    report[field] = value
    with pytest.raises(ValueError):
        validate_comparison(report, True)
