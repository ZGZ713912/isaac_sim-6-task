#!/usr/bin/env python3
"""Publish the measured new-gearbox research package and opt-in RL contracts.

No simulator, PPO, deployment, or hardware motion is launched by this script.
The new independent design table derives from the existing frozen table, with
only explicitly listed wheel-model changes supported by the fit/native report.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/wheel_real2sim_1562_20261028"
PACKAGE = ROOT / "model/wheel_identified_1562_20261028"
NATIVE = REPORT / "native_zero_gravity"
sys.path.insert(0, str(ROOT / "src"))


def relative(path):
    return str(Path(path).relative_to(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def publish_configs(model, native):
    from wheeled_tasks.chassis.identified_wheel_response import configure_identified_wheels
    from wheeled_tasks.chassis.preflight import design_preflight

    base_cfg_path = ROOT / "contracts/scratch_standing_base_mechanism_v2.json"
    base_control_path = ROOT / "contracts/flat_base_200hz_control_v2.json"
    base_design_path = ROOT / "contracts/scratch_standing_base_design_v2.json"
    cfg = json.loads(base_cfg_path.read_text())
    control = json.loads(base_control_path.read_text())
    cfg, control = configure_identified_wheels(cfg, control, PACKAGE / "model.json", ROOT)
    cfg["contract_id"] = "new-wheel-1562-real2sim-20261028-v1"
    cfg["response_migration"] = "new-wheel-1562-airborne-response-v1"
    cfg["transfer_critic"] = False
    cfg["wheel_reduction"].update(new_wheel_response_identified=True,
        response_scope="new_gearbox_airborne_effective_model; loaded torque-speed shape is still a prior",
        identified_response_gear_ratio=15.62)
    control["control_id"] = "new-wheel-1562-recorded-P05-20261028-v1"
    control["actuators"]["wheel"]["source_note"] = (
        "Both new-gearbox CSVs use 24V, full ratio15.62, output velocity P0.5 and software cap4.5Nm. "
        "Wheel-specific inertia/loss/aggregate delay and measured no-load ceilings are bound by identified_wheel_response. "
        "Rolloff shape at high loaded torque remains a prior; no second ratio multiplication.")
    control["actuators"]["wheel"]["scope"] = "new_gearbox_airborne_identified_candidate"
    write(PACKAGE / "control.json", control)
    cfg["control_math_source"] = relative(PACKAGE / "control.json")
    cfg["control_math_sha256"] = digest(PACKAGE / "control.json")
    evidence_files = [PACKAGE / "model.json", REPORT / "fit.json", REPORT / "source_audit.json",
        NATIVE / "report.json", ROOT / "src/wheeled_tasks/chassis/identified_wheel_response.py",
        ROOT / "scripts/replay_wheel_1562_isaac.py", ROOT / "scripts/replay_wheel_response_isaac.py",
        ROOT / "tools/identify_wheel_1562.py"]
    evidence = {"schema": "actuator-response-evidence-v1", "status": "validated_for_research",
        "hardware_release": False,
        "scope": "airborne new 15.62 gearbox effective dynamics and native component readback/replay; empirical residuals retained; no whole-robot qualification",
        "actuator_response": cfg["actuator_response"],
        "identified_wheel_response": cfg["identified_wheel_response"],
        "native_wheel_report": relative(NATIVE / "report.json"),
        "files_sha256": {relative(path): digest(path) for path in evidence_files}}
    write(PACKAGE / "evidence.json", evidence)
    cfg["actuator_response_evidence"] = {"file": relative(PACKAGE / "evidence.json"), "sha256": digest(PACKAGE / "evidence.json")}
    # Preserve the independent parent design. Do not rebuild its expectations
    # from the generated mechanism configuration or weaken behavioral criteria.
    design = json.loads(base_design_path.read_text())
    design["scope"] = "Opt-in new-wheel 15.62 airborne real2sim candidate; no PPO performed"
    design["review_basis"] = {"parent_files_sha256": {
        relative(path): digest(path) for path in (base_cfg_path, base_control_path, base_design_path)},
        "change": "Wheel model/P0.5 only, fresh critic; geometry, leg gains, tasks, behavioral criteria, randomization and budget retained",
        "wheel_model_sha256": digest(PACKAGE / "model.json"),
        "native_report_sha256": digest(NATIVE / "report.json"), "new_hardware_identification": True}
    changed_contract_keys = ("actuator_response", "actuator_response_evidence", "response_migration", "identified_wheel_response",
                             "wheel_reduction", "control_math_source", "control_math_sha256", "transfer_critic")
    for stage in design["stages"].values():
        for row in stage["roles"].values():
            for key in changed_contract_keys:
                row["contract"][key] = deepcopy(cfg[key])
            row["control"]["actuators.wheel"] = deepcopy(control["actuators"]["wheel"])
            row["runtime"].update(actuator_response=deepcopy(cfg["actuator_response"]),
                identified_wheel_response=deepcopy(cfg["identified_wheel_response"]),
                wheel_reduction=deepcopy(cfg["wheel_reduction"]), wheel_velocity_p=.5)
            row["runtime"]["modules"]["identified_wheel_response"] = True
    write(PACKAGE / "design.json", design)
    cfg["design_preflight"] = {"file": relative(PACKAGE / "design.json"), "sha256": digest(PACKAGE / "design.json"), "stage": "speed"}
    write(PACKAGE / "mechanism.json", cfg)
    invocation = design["stages"]["speed"]["roles"]["train"]["invocation"]
    receipt = design_preflight(cfg, control, ROOT, stage="speed", role="train",
        seed=invocation["seed"]["value"], num_envs=invocation["num_envs"])
    receipt.update(ppo_updates=0, default_entrypoint_changed=False,
        native_component_report=relative(NATIVE / "report.json"), full_robot_native_training_profile_executed=False,
        source_files_sha256={relative(path): digest(path) for path in (PACKAGE / "model.json", PACKAGE / "mechanism.json",
            PACKAGE / "control.json", PACKAGE / "design.json", PACKAGE / "evidence.json")})
    write(REPORT / "preflight.json", receipt)
    return receipt


def plot(model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname="/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False})
    traces = dict(np.load(NATIVE / "traces.npz"))
    fig, ax = plt.subplots(2, 2, figsize=(12.8, 8.3), constrained_layout=True)
    for joint, color, title in (("L_joint3", "#2673ba", "左轮"), ("R_joint3", "#d87525", "右轮")):
        row = model["axes"][joint]
        x, y = row["friction_speed_knots_rad_s"], row["moving_friction_nm"]
        ax[0, 0].plot(x, y, "o-", color=color, label=title)
        ax[0, 0].fill_between(x, np.array(y)-abs(row["direction_bias_nm"]), np.array(y)+abs(row["direction_bias_nm"]), color=color, alpha=.15)
    ax[0, 0].set_title("新减速箱的运动摩擦：左右轮独立", fontproperties=font)
    ax[0, 0].set_xlabel("输出轴转速 |ω| (rad/s)", fontproperties=font)
    ax[0, 0].set_ylabel("运动阻力矩 (N·m)", fontproperties=font)
    ax[0, 0].legend(prop=font)
    for panel, label, title in ((ax[0, 1], "left_high_r2_s12", "左轮高速保留段"),
                                 (ax[1, 0], "right_high_r2_s42", "右轮高速保留段")):
        t = traces[label + "_time_s"]
        panel.plot(t[::8], traces[label + "_measured"][::8], color="#444444", lw=1., label="实测")
        panel.plot(t[::8], traces[label + "_predicted"][::8], color="#2673ba", lw=1., alpha=.85, label="Isaac Sim")
        panel.set_title(title, fontproperties=font)
        panel.set_xlabel("时间 (s)", fontproperties=font)
        panel.set_ylabel("转速 (rad/s)", fontproperties=font)
        panel.legend(prop=font, loc="upper left")
    for label, color, title in (("left_high_r3_s3", "#2673ba", "左轮第三次运行"), ("right_high_r2_s59", "#d87525", "右轮反向")):
        t = traces[label + "_time_s"]
        ax[1, 1].plot(t[::8], abs(traces[label + "_measured"][::8]), color=color, lw=1.5, label=title + "实测")
        ax[1, 1].plot(t[::8], abs(traces[label + "_predicted"][::8]), color=color, ls="--", lw=1.5, label=title + "仿真")
    ax[1, 1].set_title("零电流滑行保留段：显示模型残差", fontproperties=font)
    ax[1, 1].set_xlim(0, 6)
    ax[1, 1].set_xlabel("时间 (s)", fontproperties=font)
    ax[1, 1].set_ylabel("|转速| (rad/s)", fontproperties=font)
    ax[1, 1].legend(prop=font, fontsize=8)
    for panel in ax.flat:
        panel.grid(alpha=.2)
    fig.suptitle("15.62 / 24 V 新轮子 real2sim · 真实曲线与原生物理回放", fontproperties=font, fontsize=16)
    fig.savefig(REPORT / "comparison.png", dpi=170)
    fig.savefig(REPORT / "comparison.pdf")
    plt.close(fig)


def document(model, native, receipt):
    rows = []
    for name, joint in (("左轮", "L_joint3"), ("右轮", "R_joint3")):
        a = model["axes"][joint]
        rows.append(f"| {name} | {a['effective_total_inertia_kg_m2']:.7f} | {a['additional_armature_kg_m2']:.7f} | "
            f"{np.interp(20,a['friction_speed_knots_rad_s'],a['moving_friction_nm']):.4f} | "
            f"{np.interp(55,a['friction_speed_knots_rad_s'],a['moving_friction_nm']):.4f} | {a['aggregate_delay_s']*1000:.1f} | "
            f"{a['high_speed_envelope']['measured_airborne_plateau_rad_s'][0]:.2f} / {a['high_speed_envelope']['measured_airborne_plateau_rad_s'][1]:.2f} |")
    score_lines = [f"| {r['label']} | {r['duration_s']:.3f} | {r['speed_rmse_rad_s']:.3f} | {r['speed_abs_error_p95_rad_s']:.3f} |" for r in native["scores"]]
    text = f"""# 新减速箱左右轮 real2sim 模型（2026-10-28）

已分别使用新减速箱 **15.62、24 V** 的低速和高速实测拟合。用户已明确确认两份 CSV 都属于新硬件。参数单位均为**轮输出轴**；没有套用旧 15.8 轮子的辨识结果，也没有再乘一次减速比。

## 可用参数与状态

| 轮子 | 总等效惯量 kg·m² | 额外 armature kg·m² | 20 rad/s 运动阻力 N·m | 55 rad/s 运动阻力 N·m | 有效总延迟 ms | 正/反空载平台 rad/s |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

总惯量包含 CAD 轮子惯量 **0.000575 kg·m²**；原生物理只添加表中的额外 armature。完整非线性摩擦曲线、方向差异和高速限幅位于 [model.json](model.json)。单一 `b·ω` 无法覆盖当前数据中静摩擦与左右轮的差异，运行时采用分段线性运动阻力曲线。静态起动名义值左 **0.19 N·m**、右 **0.375 N·m**，观测约束范围分别为 **0.15–0.23**、**0.30–0.45 N·m**；这些是有不确定性的起动参数，并非精确的静摩擦常数。

日志中左右轮的驱动器标称电流换算上限均为 **4.880434 N·m**，记录和当前候选的轮子软件限幅为 **4.5 N·m**。40 N·m 属于 DM 腿关节范围，不应套到这些轮子上。复现实验使用输出侧速度 **P=0.5**，无 I/D；这表示记录中的控制器，不表示本次另做了 PID 最优性证明。

## 拟合与验证

先用不含验证标签的有源激励及零电流训练滑行估计有效惯量；用训练力矩探针的频响约束延迟。低速 ≤2 rad/s 的运动阻力来自低速有源数据，≥6 rad/s 来自零电流训练滑行，区间内插值。这样避免长时间闭环保持段吞没被动摩擦信息。高转速的 65 rad/s 目标平台不作为“指令力矩全部有效”的样本。

高频探针以外的验证标签段、后续滑行段，以及高速文件第三次运行不进入参数求解。低速 CSV 的最后一个不完整行已排除；详细来源、SHA256、各段时长和缺口均在 [source_audit.json](../../reports/wheel_real2sim_1562_20261028/source_audit.json)。记录存在约 **38 ms** 的日志缺口，拟合窗口排除大于2 ms的间断；原生回放持有源参考并按1 ms运行，测量插值仅用于评分。

原生 **Isaac Sim 6 / CPU PhysX** 使用当前 V7 USD、固定底座及有限腿支撑关节锁定，保留原闭链。最终验证 **gravity=(0,0,0)**，共 **{native['physics_steps']}** 步、**{native['simulated_wheel_seconds']:.2f}** 个轮子仿真秒。每段只注入一次轮速初值；之后不注入实测状态。CAD惯量、额外armature、静/动摩擦均从PhysX读回核对。Native carrier最大漂移 {native['max_carrier_drift_native_units']:.3g}（各关节原生单位）。

| 原生回放段 | 时长 s | 转速 RMSE rad/s | 绝对误差 P95 rad/s |
|---|---:|---:|---:|
{chr(10).join(score_lines)}

滑行误差显著大于闭环跟踪误差，保留其原值；这份候选不能宣称所有工况都已精确复现。低速起动段作为诊断，未称独立盲测；部分慢版动态段原长30秒，原生验证使用前20秒，表中明确给出实际时长。第三次运行的左轮平台受温度/供电变化影响，平台误差亦未删去。

![实测和原生回放](../../reports/wheel_real2sim_1562_20261028/comparison.png)

## 在 RL 中使用

已接入 `ChassisEnv` 的可选 `identified_wheel_response` 模块。原生物理处理额外惯量和基础静/动摩擦，组件单独处理余下的速度相关被动阻力、有效延迟和方向相关高速驱动上限；制动维持独立的4.5 N·m软件限幅。被动阻力与电机电流统计分开，旧轮子摩擦/armature被清零，避免重复建模。

本包提供完整可选输入：

- [mechanism.json](mechanism.json)：含新轮子模型的研究训练配置。
- [control.json](control.json)：匹配记录的轮速度P0.5与输出侧4.5 N·m限幅。
- [design.json](design.json)、[evidence.json](evidence.json)：独立父设计的显式轮参数变更及原生证据哈希。
- [静态预检回执](../../reports/wheel_real2sim_1562_20261028/preflight.json)：`{receipt['status']}`；未启动PPO，未执行整车该训练配置的原生功能验收。

将现有启动入口的 mechanism/control 输入显式选为本包文件。默认训练入口保持其原配置；本包不会热改正在进行的实验。后续训练应使用新的actor、critic和空Adam。创建其他配置时可调用 `configure_identified_wheels()`，它保留腿参数，并移除历史轮子响应；新的独立设计与证据必须随配置一同更新。

## 仍需区分的物理量

1. **5.5/3.5 ms 是 CAN帧指令到上报RPM的有效总延迟**，不能分别识别电流环、传输和测速滤波。候选将它作为有效输入延迟；如果启用独立USB/反馈延迟模型，必须关闭 `apply_aggregate_delay` 并重新冻结对应证据，避免叠加。
2. 空载最高转速平台是实测的；**高速带载力矩—转速曲线形状未被这些悬空数据充分约束**。名义转矩下降宽度10%是显式先验，可在5–20%内研究随机化，不能说已通过测功机辨识。软件4.5 N·m上限也不是所有速度下都能持续达到的机械输出保证。
3. 转矩来自C620电流的额定换算，非轴端力矩传感器；惯量、摩擦均相对于这个尺度。地面接触/轮胎摩擦、齿隙、弹性、轮胎受载惯性及热模型不由本数据独立确定。
4. 起动静摩擦与低速齿槽/预滑动有混合影响。摩擦曲线在测量区间外保持端值；用于RL时应保留辨识残差与温度变化的参数不确定性。现有独立惯量±6%约定不自动扩大，也不在总惯量与armature上重复叠加随机化。

摩擦项的物理区分可参考 [MathWorks旋转摩擦文档](https://www.mathworks.com/help/simscape/ref/rotationalfriction.html)；本包的参数均取自上述本机新硬件数据，未使用该文档的示例默认值。

## 重复生成

```bash
/home/noir/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 tools/identify_wheel_1562.py prepare
/home/noir/miniconda3/envs/isaacsim6/bin/python tools/identify_wheel_1562.py fit
OMNI_KIT_ACCEPT_EULA=YES /home/noir/miniconda3/envs/isaacsim6/bin/python scripts/replay_wheel_1562_isaac.py --output /tmp/wheel_1562_native_new
```

准备步骤验证原CSV哈希后读取其本机数值缓存。现有 `native`、`native_final` 为本轮早期候选/含重力诊断记录，均保留；**本包最终证据仅绑定 `native_zero_gravity` 与当前模型SHA**。输出目录须为新目录，避免覆盖证据。
"""
    (PACKAGE / "README.md").write_text(text)


def main():
    model = json.loads((PACKAGE / "model.json").read_text())
    native = json.loads((NATIVE / "report.json").read_text())
    if (native.get("status") != "completed" or native.get("native_properties_verified") is not True
            or native["model_sha256"] != digest(PACKAGE / "model.json") or native.get("gravity_m_s2") != [0., 0., 0.]):
        raise ValueError("Final native evidence is missing, changed or not zero gravity")
    receipt = publish_configs(model, native)
    plot(model)
    document(model, native, receipt)
    print(json.dumps({"package": str(PACKAGE), "preflight": receipt["status"], "native_cases": len(native["scores"])}))


if __name__ == "__main__":
    main()
