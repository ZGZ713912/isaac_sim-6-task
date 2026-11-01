"""Measured posture clustering and static-versus-moving evidence labels."""
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from summarize_fallen_pose_bank import classify, group_modes


def record(index, tilt=0.):
    g = Rotation.from_euler("y", tilt, degrees=True).inv().apply([0.,0.,-1.]).tolist()
    return dict(index=index,status="not_settled",final_pose=dict(gravity_body=g,root_position_m=[0.,0.,.2],
        support_bodies=["base_link"],window_rest=dict(passed=True),full_joint_positions={"M_joint1":0.}))


def test_mechanical_rejections_are_never_saved_as_rest_modes():
    a = record(0)
    a["status"] = "penetration"
    assert classify(a) is None
    assert group_modes([a]) == []
    a["status"] = "not_settled"
    a["final_pose"]["cad_contact_audit"] = dict(review_required=True)
    assert classify(a) is None


def test_body_direction_stability_is_never_labelled_as_full_rest():
    a = record(0)
    a["final_pose"]["window_rest"] = dict(passed=False,seconds=2.,root_translation_range_m=.001,
        gravity_range_deg=.1,mean_support_weight_fraction=1.)
    assert classify(a) == "direction"
    a["final_pose"]["chassis_window_rest"] = dict(passed=True)
    assert classify(a) == "chassis"


def test_gravity_clusters_do_not_chain_across_large_angles_or_erase_gimbal_values():
    records = [record(0,0.),record(1,9.),record(2,18.)]
    records[1]["final_pose"]["full_joint_positions"]["M_joint1"] = 2.
    untouched = deepcopy(records)
    modes = group_modes(records)
    assert [m["record_indices"] for m in modes] == [[0,1],[2]]
    assert records == untouched
    assert sum(len(m["record_indices"]) for m in modes) == 3
    for m in modes:
        gravity = np.array([records[i]["final_pose"]["gravity_body"] for i in m["record_indices"]])
        assert np.rad2deg(np.arccos(np.clip(gravity@gravity.T,-1.,1.))).max() <= 12.
