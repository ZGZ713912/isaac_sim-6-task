"""Meaningful boundary cases for post-completion contact accounting."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("standing_acceptance_review",
    Path(__file__).resolve().parents[1]/"scripts/review_standing_acceptance.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fallen_pre_completion_is_excluded():
    raw=np.r_[np.zeros(4000,dtype=bool),np.ones(1000,dtype=bool)]
    result=module.support_review(raw,standing_start_sample=4000)
    assert result["support_pass"] and result["raw_missing_ms"]==0
    assert result["evaluated_physical_ms"]==1000 and result["excluded_pre_standing_samples"]==4000


def test_missing_completion_is_not_invented_from_good_tail():
    result=module.support_review(np.ones(1000,dtype=bool),standing_start_sample=None)
    assert not result["evaluated"] and not result["support_pass"]


def test_one_and_two_ms_pulses_ignored_but_three_ms_retained():
    raw=np.ones(100,dtype=bool)
    raw[10]=False;raw[30:32]=False;raw[70:73]=False
    original=raw.copy()
    result=module.support_review(raw,standing_start_sample=0)
    assert result["raw_missing_ms"]==6 and result["ignored_loss_ms"]==3
    assert result["raw_loss_events"]==3 and result["ignored_loss_events"]==2
    assert result["retained_loss_ms"]==3 and not result["support_pass"]
    assert np.array_equal(raw,original)


def test_exact_two_ms_and_end_boundary_are_not_off_by_one():
    result=module.support_review(np.array([False,False,True,True,False,False]),standing_start_sample=0)
    assert result["support_pass"] and result["ignored_loss_ms"]==4
    assert result["longest_raw_loss_ms"]==2 and result["effective_support_fraction"]==1.


def test_later_fall_cannot_be_removed_by_recovery_mask():
    raw=np.r_[np.ones(100,dtype=bool),np.zeros(100,dtype=bool),np.ones(100,dtype=bool)]
    result=module.support_review(raw,standing_start_sample=0)
    assert not result["support_pass"] and result["retained_loss_ms"]==100


@pytest.mark.parametrize("start",[-1,5,1.5,True])
def test_bad_completion_indices_rejected(start):
    with pytest.raises(ValueError):module.support_review(np.ones(5,dtype=bool),standing_start_sample=start)


@pytest.mark.parametrize("raw",[np.array([],dtype=bool),np.ones((2,2),dtype=bool),np.array([1.,0.])])
def test_invalid_contact_sequences_rejected(raw):
    with pytest.raises(ValueError):module.support_review(raw,standing_start_sample=0)
