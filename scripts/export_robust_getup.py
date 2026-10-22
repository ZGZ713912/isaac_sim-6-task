#!/usr/bin/env python3
"""Export an exact behavior-reviewed 35D->6D deterministic candidate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_native_pairing(candidate, baseline):
    """Require the same physical placement, not just nominal scene settings."""
    legacy = "spaced_corridors_v1"
    if candidate.get("native_origin_mode", legacy) != baseline.get("native_origin_mode", legacy):
        raise ValueError("Candidate and baseline native collision-isolation modes differ")
    lanes, parents = candidate["lanes"], baseline["lanes"]
    if not lanes or len(lanes) != len(parents):
        raise ValueError("Candidate and baseline lane populations differ")
    peak = 0.
    for lane, parent in zip(lanes, parents, strict=True):
        origin = lane.get("world_origin_m")
        if (not isinstance(origin, list) or len(origin) != 3
                or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in origin)
                or origin != parent.get("world_origin_m")
                or any(lane.get(k) != parent.get(k) for k in ("name", "pose_index", "command", "recovery", "sequence", "push", "push_sign"))):
            raise ValueError("Candidate and baseline require identical recorded native origins and cases")
        peak = max(peak, *(abs(x) for x in origin[:2]))
    # Keep the complete 96 m support inside the coordinate range checked by
    # the bounded-layout experiment. This is an engineering limit, not a
    # claim that every placement within it has equivalent PhysX trajectories.
    if peak + 48. > 1024.:
        raise ValueError("Native support exceeds the bounded world-coordinate range")
    return dict(matched_native_origins=True, max_abs_origin_xy_m=peak,
                max_abs_support_coordinate_m=peak + 48.)


def load_mean(checkpoint, *, migration_source_root=None):
    import torch
    from wheeled_algo.command_residual import mean_from_checkpoint
    from wheeled_algo.robust_getup import RobustGetupActor
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    info = state["infos"]
    if info["schema"] != "robust-getup-checkpoint-v2":
        raise ValueError("Expected a v2 training checkpoint")
    if info.get("actor_variant", "robust") != "robust":
        raise ValueError("Comparison variants are not deployment candidates")
    if any(not key.startswith("mlp.") for key in state["actor_state_dict"]):
        raise ValueError("Unsupported external actor normalization or distribution state")
    for actor_source, expected_sha in info["source_sha256"].items():
        if actor_source.startswith("src/wheeled_algo/") and digest(ROOT / actor_source) != expected_sha:
            # Only explicit migrations may read the archived implementation of
            # this backwards-compatible, opt-in actor extension.
            if (actor_source != "src/wheeled_algo/robust_getup.py" or migration_source_root is None
                    or digest(Path(migration_source_root) / actor_source) != expected_sha):
                raise ValueError("Actor source differs from the trained source: " + actor_source)
    parent_file = Path(info["base_checkpoint"])
    if digest(parent_file) != info["base_checkpoint_sha256"]:
        raise ValueError("Locomotion parent changed")
    parent = torch.load(parent_file, map_location="cpu", weights_only=True)
    plan = json.loads((ROOT / parent["infos"]["plan_file"]).read_text())
    mean = RobustGetupActor(mean_from_checkpoint(parent, ROOT / plan["actor_bundle"] / "source.onnx"),
                           quiet_alpha=info["settings"]["quiet_actor_alpha"],
                           canonical_recovery=info["settings"].get("canonical_recovery", False),
                           motion_residual_fraction=info["settings"].get("motion_residual_fraction", 0.),
                           cyclic_leg_targets=info["settings"].get("cyclic_leg_targets", False),
                           leg_action_clip=info["settings"]["leg_action_clip"],
                           recovery_leg_step=info["settings"].get("recovery_leg_step", 1.2),
                           smooth_knee_targets=info["settings"].get("smooth_knee_targets", False),
                           recovery_knee_center=info["settings"].get("recovery_knee_center", [.924544, .924537]),
                           quiet_velocity_feature_scale=info["settings"].get("quiet_velocity_feature_scale"),
                           quiet_velocity_feature_gain=info["settings"].get("quiet_velocity_feature_gain", 1.),
                           quiet_velocity_feedback=info["settings"].get("quiet_velocity_feedback", False),
                           quiet_wheel_bound=info["settings"].get("quiet_wheel_bound", .5),
                           stance_aware_recovery=info["settings"].get("stance_aware_recovery", False),
                           recovery_shape_aware=info["settings"].get("recovery_shape_aware", False),
                           quiet_high_common_alpha=info["settings"].get("quiet_high_common_alpha"),
                           quiet_high_leg_prediction_s=info["settings"].get("quiet_high_leg_prediction_s", 0.),
                           quiet_height_experts=info["settings"].get("quiet_height_experts", False),
                           quiet_native_height_anchors=info["settings"].get("quiet_native_height_anchors", False),
                           fast_forward_reference=info["settings"].get("fast_forward_reference", False),
                           fast_forward_reference_range=info["settings"].get("fast_forward_reference_range", [1.5, 2.]),
                           fast_forward_reference_yaw_fade=info["settings"].get("fast_forward_reference_yaw_fade", .25))
    weights = {k.removeprefix("mlp.mean."): v for k, v in state["actor_state_dict"].items() if k.startswith("mlp.mean.")}
    mean.load_state_dict(weights, strict=True)
    expected_bounds = torch.tensor([info["settings"]["leg_action_clip"]] * 4 + [info["settings"]["wheel_action_clip"]] * 2)
    if (not torch.equal(mean.action_bound, expected_bounds)
            or not torch.equal(mean.quiet_alpha, torch.tensor(info["settings"]["quiet_actor_alpha"]))
            or not torch.equal(mean.quiet_bound, torch.tensor([.4] * 4 + [info["settings"].get("quiet_wheel_bound", .5)] * 2))):
        raise ValueError("Declared action bounds or quiet filter differ from trained buffers")
    expected = {k.removeprefix("mlp."): v for k, v in parent["actor_state_dict"].items() if k.startswith("mlp.")}
    if any(not torch.equal(value, mean.parent.state_dict()[key]) for key, value in expected.items()):
        raise ValueError("Frozen locomotion parent was modified")
    if not all(torch.isfinite(v).all() for v in mean.state_dict().values()):
        raise ValueError("Nonfinite policy state")
    return mean.eval(), info


def assess(reports, baselines, checkpoint, protocol):
    from analyze_robust_getup import analyze
    checkpoint_sha = digest(checkpoint)
    failures, results = [], []
    if len(reports) != 2 or len(baselines) != 2:
        raise ValueError("Two matched confirmation seeds are required")
    for evaluation, baseline in zip(reports, baselines, strict=True):
        data = json.loads((evaluation / "report.json").read_text())
        parent = json.loads((baseline / "report.json").read_text())
        if any(row.get("evaluation_overrides") or row.get("case_subset") for row in (data, parent)):
            raise ValueError("Diagnostic overrides/subsets cannot qualify a deployment candidate")
        if (data["checkpoint_sha256"] != checkpoint_sha or parent["checkpoint_sha256"] != checkpoint_sha
                or data["actor_variant"] != "robust"
                or parent["actor_variant"] != "parent" or data["seed"] != parent["seed"]
                or data.get("ablation_variants", False)):
            raise ValueError("Evaluation/checkpoint/baseline identity mismatch")
        if any(row["seconds"] != 12. or row["policy_hz"] != 200 or row["physics_hz"] != 1000
               for row in (data, parent)):
            raise ValueError("Confirmation requires complete 12 s, 200/1000 Hz native episodes")
        if (data["task_contract_sha256"] != parent["task_contract_sha256"]
                or data["material"] != parent["material"]
                or data["robust_domains"]["enabled_environments"] != 0
                or parent["robust_domains"]["enabled_environments"] != 0):
            raise ValueError("Use matched nominal scene and domain settings for candidate confirmation")
        if data.get("paired_feedback_winding") is not True or parent.get("paired_feedback_winding") is not True:
            raise ValueError("Confirmation must use coherent paired encoder feedback for PD")
        pairing = verify_native_pairing(data, parent)
        analysis, original = analyze(evaluation), analyze(baseline)
        if analysis["passive_motor_peak_nm"] != 0. or analysis["max_gap_m"] > protocol["closure_gap_max_m"]:
            failures.append("passive torque or closed-chain protection")
        for family, rate in protocol["family_min_complete_rate"].items():
            row = analysis["actual_settled_attitudes"].get(family, {})
            successes = row.get(protocol.get("complete_metric", "recovery_and_motion_success"), 0)
            if successes < protocol["minimum_successful_episodes_per_family"] or successes < rate * row.get("episodes", 0):
                failures.append(f"seed {data['seed']} actual {family} recovery and resumed motion")
        if analysis["motion_survived"] < protocol["ordinary_motion_survival_rate_min"] * analysis["motion_episodes"]:
            failures.append(f"seed {data['seed']} ordinary motion survival")
        for name in ("arc_forward_left", "arc_forward_right", "arc_reverse_left", "arc_reverse_right"):
            row = analysis["cases"].get(name, {})
            if (row.get("vx_mae_m_s") is None or row["vx_mae_m_s"] > protocol["ordinary_arc_vx_mae_max_m_s"]
                    or row["yaw_mae_rad_s"] > protocol["ordinary_arc_yaw_mae_max_rad_s"]):
                failures.append(f"seed {data['seed']} {name} simultaneous tracking")
        quiet_gains, fast_gains, physical_gains = [], [], []
        for name in ("stand", "stand_low", "stand_high"):
            row, old = analysis["cases"][name], original["cases"][name]
            quiet, old_quiet = row.get("quiet", {}), old.get("quiet", {})
            if (row["survived"] < protocol["stand_survival_rate_min"] * row["episodes"]
                    or quiet.get("vx_rms_m_s", float("inf")) > protocol["stand_vx_rms_max_m_s"]
                    or quiet.get("yaw_rms_rad_s", float("inf")) > protocol["stand_yaw_rms_max_rad_s"]):
                failures.append(f"seed {data['seed']} {name} standing")
            for field in ("leg_target_hf_rms_rad", "wheel_target_hf_rms_rad_s"):
                if old_quiet.get(field, 0.) > 1e-6 and field in quiet:
                    quiet_gains.append(1. - quiet[field] / old_quiet[field])
            for field in ("leg_target_fast_rms_rad_3_95", "wheel_target_fast_rms_rad_s_3_95"):
                if old_quiet.get(field, 0.) > 1e-8 and field in quiet:
                    fast_gains.append(1. - quiet[field] / old_quiet[field])
            for field in ("leg_motion_fast_rms_rad_3_95", "wheel_motion_fast_rms_rad_s_3_95"):
                if old_quiet.get(field, 0.) > 1e-8 and field in quiet:
                    physical_gains.append(1. - quiet[field] / old_quiet[field])
            if quiet.get("body_xy_worst_episode_excursion_m", float("inf")) > protocol.get("stand_xy_excursion_max_m", float("inf")):
                failures.append(f"seed {data['seed']} {name} 5 cm sway envelope")
        if not quiet_gains or sum(quiet_gains) / len(quiet_gains) < protocol["nominal_quiet_hf_target_rms_improvement_fraction"]:
            failures.append(f"seed {data['seed']} quiet target vibration reduction")
        for gains, key, label in ((fast_gains, "quiet_3_95_target_improvement_fraction", "3..95 Hz target reduction"),
                                 (physical_gains, "quiet_3_95_physical_improvement_fraction", "3..95 Hz physical motion")):
            if key in protocol and (not gains or sum(gains) / len(gains) < protocol[key]):
                failures.append(f"seed {data['seed']} {label}")
        moving_success = (analysis["practical_moving_fall_recoveries"] if protocol.get("complete_metric", "").startswith("practical")
            else sum(row.get("push") and row.get("in_motion_fall") and row.get("recovered_after_push")
                     and row["followup_success"] and row["survived"] for row in data["lanes"]))
        if not moving_success:
            failures.append(f"seed {data['seed']} actual in-motion fall, recovery and resumed command")
        results.append(dict(evaluation=str(evaluation), evaluation_sha256=digest(evaluation / "report.json"),
            baseline=str(baseline), baseline_sha256=digest(baseline / "report.json"),
            native_pairing=pairing,
            mean_quiet_hf_improvement_fraction=sum(quiet_gains) / len(quiet_gains) if quiet_gains else None,
            mean_quiet_3_95_target_improvement_fraction=sum(fast_gains) / len(fast_gains) if fast_gains else None,
            mean_quiet_3_95_physical_improvement_fraction=sum(physical_gains) / len(physical_gains) if physical_gains else None,
            actual_settled_attitudes=analysis["actual_settled_attitudes"],
            exact_180_reset_cases={name: {key: analysis["cases"][name][key]
                for key in ("episodes", "recovered", "practical_recovery_and_motion_success")}
                for name in ("inverted_symmetric", "inverted_asymmetric")},
            ordinary_motion=dict(survived=analysis["motion_survived"], episodes=analysis["motion_episodes"]),
            ordinary_arcs={name: {key: analysis["cases"][name][key]
                for key in ("vx_mae_m_s", "yaw_mae_rad_s", "both_axes_tracking_fraction")}
                for name in ("arc_forward_left", "arc_forward_right", "arc_reverse_left", "arc_reverse_right")},
            quiet_standing={name: analysis["cases"][name]["quiet"] for name in ("stand", "stand_low", "stand_high")},
            actual_moving_fall_recoveries=moving_success,
            max_gap_m=analysis["max_gap_m"]))
    if sorted(r["seed"] for r in (json.loads((p / "report.json").read_text()) for p in reports)) != sorted(protocol["required_confirmation_seeds"]):
        failures.append("confirmation seeds")
    return dict(adequate=not failures, failures=failures, comparisons=results)


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, nargs=2, required=True)
    parser.add_argument("--baseline", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "reports/unified_robust_20261021/acceptance_protocol003.json")
    args = parser.parse_args()
    behavior = assess(args.evaluation, args.baseline, args.checkpoint, json.loads(args.protocol.read_text()))
    if not behavior["adequate"]:
        raise ValueError("Candidate still needs iteration: " + "; ".join(behavior["failures"]))
    mean, info = load_mean(args.checkpoint)
    if not info["settings"].get("paired_feedback_winding", False):
        raise ValueError("Deployment candidate must bind the corrected paired-feedback contract")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    policy = args.output / "policy.onnx"
    torch.onnx.export(mean, torch.zeros(1, 35), str(policy), input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    model = onnx.load(policy)
    onnx.helper.set_model_props(model, dict(actor_adaptation="wheel-leg-robust-getup-v2", policy_hz="200",
        observation_dim="35", action_dim="6", motor_control="full_RL_after_passive_enable",
        source_checkpoint_sha256=digest(args.checkpoint),
        recovery_coordinates=("smooth_absolute_knee_circular_common" if info["settings"].get("smooth_knee_targets") else "circular_common"),
        recovery_authority=("stance_and_knee_shape_v1" if info["settings"].get("recovery_shape_aware")
            else "stance_aware_v1" if info["settings"].get("stance_aware_recovery") else "tilt_only_v1"),
        leg_winding_version="paired_branch107_v1",
        leg_target_model_spec_sha256="0ac82045ab584f20091538aa782d93ea545382dc701faca1f8ae92aeb7cd1997"))
    onnx.save(model, policy)
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(policy), sess_options=options, providers=["CPUExecutionProvider"])
    worst, checked, vectors = 0., 0, []
    with torch.inference_mode():
        for directory in args.evaluation:
            obs = np.load(directory / "trajectory.npz")["observations"].reshape(-1, 35)
            for index, value in enumerate(obs[np.linspace(0, len(obs) - 1, min(2048, len(obs)), dtype=int)]):
                expected = mean(torch.from_numpy(value[None])).numpy()
                actual = session.run(None, {"obs": value[None]})[0]
                if not np.isfinite(actual).all():
                    raise ValueError("Nonfinite ONNX output")
                worst = max(worst, float(np.abs(actual - expected).max()))
                checked += 1
                if index % 128 == 0:
                    vectors.append(dict(evaluation=str(directory), observation=value.tolist(),
                                        expected_raw_action=expected[0].tolist()))
    if worst > 5e-4:
        raise ValueError(f"ONNX parity error: {worst}")
    timing_obs = np.asarray(vectors[0]["observation"], dtype=np.float32)[None]
    for _ in range(20):
        session.run(None, {"obs": timing_obs})
    milliseconds = []
    for _ in range(512):
        start = time.perf_counter()
        session.run(None, {"obs": timing_obs})
        milliseconds.append((time.perf_counter() - start) * 1000.)
    shutil.copyfile(args.checkpoint, args.output / "checkpoint.pt")
    shutil.copyfile(args.protocol, args.output / "acceptance_protocol.json")
    shutil.copyfile(ROOT / "scripts/verify_robust_getup_export.py", args.output / "verify_export.py")
    (args.output / "inference_vectors.json").write_text(json.dumps(dict(
        schema="robust-getup-offline-inference-v1", model_sha256=digest(policy),
        input_dtype="float32", input_shape=[1,35], output_shape=[1,6],
        max_abs_tolerance=5e-4, vectors=vectors), indent=2, allow_nan=False) + "\n")
    contract = dict(schema="robust-getup-deployment-v2", model_sha256=digest(policy),
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
        actor_dim=35, action_dim=6, history_length=1, policy_hz=200, physical_pd_hz=1000,
        deterministic_mean=True, new_robust_updates=info["completed_updates"],
        update_count_scope="selected lineage; unselected experiments are accounted separately",
        model_migration=info.get("control_revision"),
        resumed_from_checkpoint_sha256=info.get("migrated_from_sha256"),
        prior_recovery_updates=info["prior_recovery_updates"], prior_locomotion_updates=1346,
        parent_export_pretraining_updates="unknown", leg_action_clip=info["settings"]["leg_action_clip"], wheel_action_clip=9.,
        leg_position_scale=.25, wheel_velocity_scale=10., leg_kp=160., leg_kd=2.5, wheel_kp=.6,
        leg_winding_version="paired_branch107_v1",
        leg_target_model_spec_sha256="0ac82045ab584f20091538aa782d93ea545382dc701faca1f8ae92aeb7cd1997",
        policy_order=["L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"],
        nominal_model_pos=[-.42,.13742282595395358,.42,-.1374155762580851,0.,0.],
        required_runtime_changes=["bypass scripted recovery; policy owns all six motors from enable",
            "allow raw leg actions +/-16 (offsets +/-4 rad); the inspected RMCS policy.hpp/action.cpp still clip raw legs at +/-3",
            "use the confirmed actual 200 Hz policy clock and matching 5 ms history; do not reuse the inspected legacy 50 Hz compile-time policy period",
            "a dedicated full-RL entry must allow recovery attitudes beyond the old 45-degree lost-upright guard while retaining actual hardware fault protections",
            "zero all six motor efforts before enable; retain gas springs",
            "at first enable seed previous leg actions from paired common-angle-wrapped measured nominal offsets / .25",
            "use exported actions as the next previous-action observation in policy order",
            "retain paired_branch107 projection, torque limits and sensor freshness protections",
            "retain RMCS wind_leg_feedback before paired target decoding and PD; never subtract independently wrapped leg encoder angles",
            "retain yaw and reduce vx at abs(vx*yaw)>2.06; retain requested motion during recovery"],
        quiet_filter="inside ONNX, using the previous action; no additional deployment filter",
        paired_feedback_winding=info["settings"].get("paired_feedback_winding", False),
        observation_layout=[
            dict(start=0, count=3, value="body forward/lateral/yaw commands", scale=1.),
            dict(start=3, count=1, value="requested body height in metres", scale=5.),
            dict(start=4, count=3, value="body angular velocity in rad/s", scale=.5),
            dict(start=7, count=3, value="unit projected gravity in body axes", scale=1.),
            dict(start=10, count=4, value="atan2(sin(q-nominal),cos(q-nominal)); policy leg order", scale=1.),
            dict(start=14, count=2, value="zero wheel position channels", scale=1.),
            dict(start=16, count=6, value="motor angular velocity in rad/s; policy order", scale=.1),
            dict(start=22, count=6, value="previous exported raw actions; policy order", scale=1.),
            dict(start=28, count=7, value=[1.,0.,0.,0.,0.,0.,0.], scale=1.)],
        recovery_coordinates=dict(smooth_absolute_knee=info["settings"].get("smooth_knee_targets", False),
            knee_center_signed_delta_rad=info["settings"].get("recovery_knee_center"),
            common_target_lead_max_rad=.35, knee_delta_target_lead_max_rad=.15),
        stance_aware_recovery=info["settings"].get("stance_aware_recovery", False),
        recovery_shape_aware=info["settings"].get("recovery_shape_aware", False),
        quiet_review_objective=dict(sway_envelope_m=.05, fast_motion_band_hz=[3.,95.],
            position_input_to_actor=False, long_term_position_bound_guaranteed=False),
        simulation_limits=dict(self_collisions_enabled=False,
            randomized_parameters_identified_on_vehicle=False,
            evidence_window_seconds=12., quiet_analysis_window_seconds=[3., 12.]),
        training_domain_settings=info["settings"],
        source_sha256=info["source_sha256"],
        ort_max_abs_error=worst, ort_checked_observations=checked,
        cpu_inference_ms=dict(samples=512, threads=1, p50=float(np.median(milliseconds)),
                              p99=float(np.quantile(milliseconds, .99))),
        exporter_sha256=digest(__file__), acceptance_protocol_sha256=digest(args.protocol),
        behavior=behavior, evidence_scope="Isaac Sim; user performs vehicle testing",
        hardware_tested=False)
    (args.output / "deployment_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    (args.output / "README.md").write_text(
        "# 200 Hz 全 RL 起身与运动候选\n\n"
        "`policy.onnx` 是 1×35 → 1×6 的确定性均值策略。具体结果、SHA 和通道约定见 `deployment_contract.json`。\n\n"
        "`checkpoint.pt` 保留本次导出来源；`inference_vectors.json` 提供真实观测与预期 raw action，"
        "可先在运行端做离线数值对照，再由用户启动实车。\n\n"
        "在具备 NumPy 和 ONNX Runtime 的环境运行 `python verify_export.py` 可核对 SHA、形状、"
        "动作范围和冻结推理向量；该脚本只做离线推理。\n\n"
        "腿动作乘 0.25 后是相对名义角的偏移，表示范围为 ±4 rad；轮动作乘 10 后为 ±90 rad/s。"
        "当前读取的 RMCS `rmcs_rl/src/policy.hpp` / `action.cpp` 仍将腿 raw action 限为 ±3（偏移 ±0.75 rad），会截断起身动作；"
        "运行端需要支持本合同的 ±16。用户已确认实际策略为 200 Hz；同时核对代码中仍为 50 Hz 的旧常量及其派生周期，"
        "确保本策略每 5 ms 执行和更新历史。保持 paired_branch107_v1 成对换绕与闭链投影、"
        "160/2.5 反馈 PD、1000 Hz PD 和既有真实力矩/新鲜度保护。35 维观测中的腿角必须按相对名义角绕回 ±π。\n\n"
        "全 RL 入口需要在失能阶段保留气弹簧、六电机零力矩；使能首帧用测得腿角作成对共同角换绕，除以 0.25 播种动作历史，"
        "轮历史置零。之后每个 5 ms 输入上一帧导出的 raw action，不能输入加比例/名义角后的电机目标。"
        "使能后由同一策略控制六电机，不先执行旧转腿脚本。\n\n"
        "组合命令遵守 abs(vx × yaw) ≤ 2.06 m/s²：例如 1 m/s + 1 rad/s 均保留，"
        "3 m/s + 4 rad/s 会保留转速并把前进降到约 0.515 m/s。该限幅也必须在运行端一致。\n\n"
        "RMCS 当前原生入口仍有准备姿态和 45° 失稳锁止；实车全 RL 起身需要专用入口允许恢复阶段的大倾角，"
        "并保留编码器/IMU 新鲜度、真实机械和力矩保护。仅替换旧 ONNX 不会启用任意姿态自起。\n\n"
        "静站平滑在 ONNX 内，姿态扰动时会释放；无需再叠加部署低通。静站验收允许约 5 cm 内缓慢调整，"
        "没有为 actor 增加位置输入，因而不能保证无限时长位置不漂移。"
        "当前仿真关闭自碰撞，域参数范围属于鲁棒性假设，尚未经过车辆辨识。"
        "本包只有 Isaac Sim 证据，实际车辆测试和启动由用户执行。\n", encoding="utf-8")
    (args.output / "SHA256SUMS").write_text("".join(
        digest(path) + "  " + path.name + "\n" for path in sorted(args.output.iterdir()) if path.is_file()))
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
