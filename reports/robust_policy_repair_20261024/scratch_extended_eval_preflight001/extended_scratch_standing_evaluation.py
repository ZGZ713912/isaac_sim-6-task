"""Extend only the archived standing horizon; retain every acceptance rule."""
import ast
import hashlib
import math
from pathlib import Path


def configure(config, design, settings, duration_s):
    if type(duration_s) is not int or duration_s != 30:
        raise ValueError("The explicit extended evaluation is bounded to30 seconds")
    selected = design["stages"]["speed"]["roles"]["train"]["contract"]
    if (config["episode_seconds"] != 8. or selected["episode_seconds"] != 8.
            or settings["episode_seconds"] != 8. or config["physics_dt"] != .001
            or config["policy_dt"] != .005):
        raise ValueError("Need the authenticated8s/200Hz/1000Hz original standing protocol")
    for value in (config, selected, settings):
        value["episode_seconds"] = float(duration_s)
    return dict(schema="scratch-extended-standing-evaluation-v1", original_duration_s=8,
        duration_s=duration_s, policy_frames=6000, physical_frames=30000,
        policy_dt_s=.005, physical_dt_s=.001, first_ms_included=True,
        contact_threshold_n=2., startup_grace_s=0., acceptance_unchanged=True,
        new_PPO=0, optimizer_constructed=False, adoption_by_training=False,
        scope="Explicit longer read-only evaluation. Only episode timeout and archived loop horizon "
              "extend8->30s; original21 cases, physics, controller, reset, protection and strict criteria remain.")


def extend_function(source, duration_s, namespace, destination):
    if type(duration_s) is not int or duration_s != 30:
        raise ValueError("The explicit extended evaluation is bounded to30 seconds")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != "evaluate":
        raise ValueError("Need exactly the archived evaluate function")
    original = ast.dump(tree, include_attributes=False)
    matching = [node for node in tree.body[0].body if isinstance(node, ast.Assign)
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "horizon"]
    if len(matching) != 1:
        raise ValueError("The archived horizon assignment changed")
    value = matching[0].value
    expected = ast.parse("round(8. / env.policy_dt)", mode="eval").body
    if ast.dump(value, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError("The archived8-second horizon expression changed")
    value.args[0].left.value = float(duration_s)
    adapted = ast.dump(tree, include_attributes=False)
    # Restoring the single literal must restore the entire original AST.
    value.args[0].left.value = 8.
    if ast.dump(tree, include_attributes=False) != original:
        raise ValueError("An expression besides the horizon changed")
    value.args[0].left.value = float(duration_s)
    path = Path(destination)
    if path.exists():
        raise ValueError("Preserve the earlier adapted evaluation source")
    path.write_text(ast.unparse(tree) + "\n")
    # The original globals retain the authenticated geometry, reward-free
    # evaluation machinery, trace writers and imports. No learner is present.
    local = {}
    exec(compile(tree, str(path), "exec"), dict(namespace), local)
    return local["evaluate"], dict(original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_ast_sha256=hashlib.sha256(original.encode()).hexdigest(),
        extended_ast_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
        extended_source=str(path), extended_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        semantic_change="Only evaluate.horizon numerator8.0->30.0; restoring it restores the entire archived AST")


def assert_runtime(env, contract):
    if (env.dt != contract["physical_dt_s"] or env.policy_dt != contract["policy_dt_s"]
            or env.max_episode_length != contract["policy_frames"]
            or env.getup_cfg["episode_seconds"] != contract["duration_s"]):
        raise ValueError("Actual physical clocks/episode horizon differ from the30s protocol")
    return dict(policy_frames=env.max_episode_length, policy_dt_s=env.policy_dt,
        physical_dt_s=env.dt, episode_seconds=env.getup_cfg["episode_seconds"])


def assert_result(result, contract):
    if result["case_count"] != 21 or len(result["cases"]) != 21:
        raise ValueError("Long evaluation must retain every original21 case")
    expected = [(height, pitch) for height in (.23, .305, .42)
                for pitch in (0., -.25, .25, -.5, .5, -1., 1.)]
    if [(case["height_m"], case["pitch_deg"]) for case in result["cases"]] != expected:
        raise ValueError("Long evaluation changed the original case population or order")
    for case in result["cases"]:
        count = case["physical_ms"]
        if (type(count) is not int or count <= 0 or count > contract["physical_frames"]
                or count % 5 or case["full_horizon"] != (count == contract["physical_frames"])):
            raise ValueError("A case has an inconsistent physical clock or truncated full-horizon claim")
        if (type(case["missing_physical_ms"]) is not int
                or not 0 <= case["missing_physical_ms"] <= count
                or any(not math.isfinite(case[name]) for name in ("tilt_peak_deg", "height_mae_m",
                    "forward_speed_rms_m_s", "leg_speed_peak_rad_s", "closure_gap_peak_m"))):
            raise ValueError("Nonfinite or invalid original standing measurements")
        support = bool(case["full_horizon"] and case["missing_physical_ms"] == 0)
        standing = bool(support and case["tilt_peak_deg"] < 10. and case["height_mae_m"] < .01
            and case["forward_speed_rms_m_s"] < .03 and case["leg_speed_peak_rad_s"] < 3.
            and case["closure_gap_peak_m"] < .003)
        if case["continuous_support_pass"] != support or case["standing_pass"] != standing:
            raise ValueError("The original strict standing/support acceptance changed")
    if (result["continuous_support_pass"] != sum(case["continuous_support_pass"] for case in result["cases"])
            or result["standing_pass"] != sum(case["standing_pass"] for case in result["cases"])):
        raise ValueError("Case-level and aggregate long evaluation decisions disagree")
