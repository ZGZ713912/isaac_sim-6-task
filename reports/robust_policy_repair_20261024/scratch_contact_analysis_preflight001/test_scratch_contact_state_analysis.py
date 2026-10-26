"""Reject clock/provenance ambiguity before interpreting native contact gaps."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("contact_analysis",
    Path(__file__).resolve().parents[1] / "scripts/analyze_scratch_contact_state.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def recorded():
    force = np.zeros((1, 5, 1, 2, 3))
    force[..., 2] = 10.
    force[0, :4, 0, 0, 2] = 2.  # Strictly >2N is required; equality is missing support.
    physical = dict(valid=np.ones((1, 1), dtype=bool), force=force)
    filtered = force.reshape(5, 1, 2, 3).copy()
    net, point = filtered.copy(), filtered.copy()
    net[:2, 0, 0, 2] = 10.
    point[[0, 2], 0, 0, 2] = 10.
    state = dict(valid=np.ones((5, 1), dtype=bool), filtered_wheel_force=filtered,
        wheel_net_force=net, wheel_point_force=point,
        joint_position=np.zeros((5, 1, 2)), joint_velocity=np.zeros((5, 1, 2)),
        root_link_pose_local=np.zeros((5, 1, 7)), root_com_velocity_world=np.zeros((5, 1, 6)),
        wheel_link_pose_local=np.zeros((5, 1, 2, 7)),
        wheel_contact_point_count=np.ones((5, 1, 2), dtype=int),
        wheel_minimum_contact_separation=np.zeros((5, 1, 2)),
        contact_buffer_overflow=np.zeros((5, 1, 2), dtype=bool))
    cases = [dict(height_m=.305, pitch_deg=0., physical_ms=5, missing_physical_ms=4, standing_pass=False)]
    metadata = dict(joint_names=['left', 'right'], wheel_names=['left_wheel', 'right_wheel'])
    return physical, state, cases, metadata


def test_all_readout_patterns_are_classified_without_relaxing_original_acceptance():
    report = reader.classify(*recorded())
    row, = report['cases']
    assert row['original_missing_physical_ms'] == row['missing_wheel_samples'] == 4
    assert list(row['missing_wheel_readout_categories'].values()) == [1, 1, 1, 1]
    assert row['original_standing_pass'] is False
    assert row['mid_height_break_events'][0]['duration_ms'] == 4
    assert report['qualification'] is False


@pytest.mark.parametrize('fault', ['clock', 'force', 'overflow', 'nonfinite', 'denominator', 'outcome', 'separation'])
def test_ambiguous_native_evidence_is_rejected(fault):
    physical, state, cases, metadata = recorded()
    if fault == 'clock': state['valid'][2, 0] = False
    if fault == 'force': state['filtered_wheel_force'][0, 0, 0, 2] += 1.
    if fault == 'overflow': state['contact_buffer_overflow'][0, 0, 0] = True
    if fault == 'nonfinite': state['joint_velocity'][0, 0, 0] = np.nan
    if fault == 'denominator': cases[0]['physical_ms'] += 1
    if fault == 'outcome': cases[0]['missing_physical_ms'] -= 1
    if fault == 'separation': state['wheel_minimum_contact_separation'][0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        reader.classify(physical, state, cases, metadata)
