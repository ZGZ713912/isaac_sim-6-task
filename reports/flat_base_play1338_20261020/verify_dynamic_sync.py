"""Seal measured GUI link-transform regression evidence; no native/PPO launch."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def angle(a, b):
    a, b = [float(z) for z in a], [float(z) for z in b]
    na, nb = math.sqrt(sum(z*z for z in a)), math.sqrt(sum(z*z for z in b))
    a, b = [z/na for z in a], [z/nb for z in b]
    d = math.sqrt(min(sum((x-y)**2 for x, y in zip(a, b)), sum((x+y)**2 for x, y in zip(a, b))))
    return math.degrees(4 * math.asin(min(1., d/2)))


def summarize(rows):
    return dict(frames=len(rows), body_count=len(rows[0]["before"]),
        end_time_s=rows[-1]["time_s"],
        max_position_error_m=max(v["position_error_m"] for r in rows for v in r["before"]),
        max_orientation_error_deg=max(angle(v["physics_pose"][3:], v["usd_pose"][3:]) for r in rows for v in r["before"]),
        publisher_did_not_change_physics=all(r["physics_unchanged"] for r in rows),
        explicit_publish_count=sum(r["explicit_publish"] for r in rows))


def main():
    target = HERE / "gui_dynamic_validation001.json"
    if target.exists():
        raise FileExistsError(target)
    files, rows, reports = {}, {}, {}
    for tag in ("001", "002", "003"):
        path = HERE / ("dynamic_sync" + tag)
        rows[tag] = read(path / "transforms.json")
        reports[tag] = read(path / "gui/report.json")
        assert reports[tag]["status"] == "completed", reports[tag].get("error")
        for name in ("transforms.json", "probe.py", "native.log", "gui/report.json"):
            files[str((path / name).relative_to(ROOT))] = digest(path / name)
    overlap = min(len(rows["001"]), len(rows["002"]))
    paired = all(a["time_s"] == b["time_s"] and
        [v["physics_pose"] for v in a["before"]] == [v["physics_pose"] for v in b["before"]]
        for a, b in zip(rows["001"][:overlap], rows["002"][:overlap]))
    topology = read(HERE / "dynamic_sync003/visual_topology.json")
    summaries = {k: summarize(v) for k, v in rows.items()}
    controls = HERE / "dynamic_sync002/cpu_controls.xml"
    import xml.etree.ElementTree as ET
    suite = ET.parse(controls).getroot().find("testsuite")
    policy = ROOT / "models/flat_static_grip1338_20261020/policy.onnx"
    checks = dict(old_visual_failure_reproduced=summaries["001"]["max_position_error_m"] > .49,
        all_fixed_positions_match=summaries["002"]["max_position_error_m"] == summaries["003"]["max_position_error_m"] == 0.,
        all_fixed_orientations_match=max(summaries[k]["max_orientation_error_deg"] for k in ("002", "003")) < .0001,
        paired_physics_exact=paired and overlap >= 120,
        production_has_no_extra_publisher=summaries["003"]["explicit_publish_count"] == 0,
        production_transform_transport=reports["003"]["gui_render_settings"] == dict(use_fabric=False, update_to_usd=True, fabric_extension_enabled=False),
        visual_meshes_inherit_body_transforms=all(v["valid"] and v["meshes"] > 0 and not v["reset_xform_paths"] for v in topology.values()),
        command_tests_passed=int(suite.attrib["tests"]) == 19 and int(suite.attrib["failures"]) == int(suite.attrib["errors"]) == 0,
        model_unchanged=digest(policy) == "9cc82fd9355458305d372cd5b0299c9f35ae0d58ce81f05a94bdd7f66ee9f874",
        no_training=all(r["formal_updates_consumed"] == 0 and not r["optimizer_constructed"] for r in reports.values()))
    for f in ("src/wheeled_tasks/chassis/env.py", "scripts/play_chassis_suite.py", "scripts/play_flat_base.py", "docs/FLAT_BASE_GUI_PLAY.md"):
        files[f] = digest(ROOT / f)
    for path in (controls, HERE / "dynamic_sync003/visual_topology.json", policy):
        files[str(path.relative_to(ROOT))] = digest(path)
    result = dict(status="passed" if all(checks.values()) else "failed", checks=checks,
        comparison=dict(common_frames=overlap, common_body_poses=overlap * summaries["002"]["body_count"],
            paired_physics_exact=paired), native=summaries,
        production_render_settings=reports["003"]["gui_render_settings"],
        qualification=None, behavior_training_qualified=False, real_time_qualified=False,
        actual_keyboard_events_tested=False, tests="Programmed commands through the production command path and existing CPU keyboard tests",
        source_and_evidence_sha256=files)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert all(checks.values()), checks


if __name__ == "__main__":
    main()
