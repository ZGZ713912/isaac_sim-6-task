"""Diagnostic speed lanes must retain their real command and frozen reset."""
import importlib.util
from pathlib import Path


def test_speed_seam_cases_keep_pose_and_do_not_mutate_the_full_suite_template():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_robust_getup_batch.py"
    spec = importlib.util.spec_from_file_location("robust_evaluation_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template = dict(name="forward", recovery=False, pose_index=17, command=[1., 0., .305])
    cases = module.fixed_forward_cases(template, [1.5, 1.75, 2.], 16)
    assert len(cases) == 48
    assert all(case["pose_index"] == 17 and not case["recovery"] for case in cases)
    for speed in (1.5, 1.75, 2.):
        assert sum(case["command"] == [speed, 0., .305] for case in cases) == 16
    cases[0]["command"][0] = 0.
    assert cases[3]["command"] == [1.5, 0., .305]
    assert template == dict(name="forward", recovery=False, pose_index=17, command=[1., 0., .305])


def test_forward_yaw_seam_records_each_true_two_axis_command():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_robust_getup_batch.py"
    spec = importlib.util.spec_from_file_location("robust_evaluation_yaw_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template = dict(name="forward", recovery=False, pose_index=17, command=[1., 0., .305])
    cases = module.fixed_forward_cases(template, [3.], 8, [0., .05, .125, .25, .5, -.125])
    assert len(cases) == 48
    assert len({case["name"] for case in cases}) == 6
    assert all(case["pose_index"] == 17 for case in cases)
    assert sum(case["command"] == [3., -.125, .305] for case in cases) == 8
