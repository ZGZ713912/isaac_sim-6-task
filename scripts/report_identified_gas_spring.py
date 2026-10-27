#!/usr/bin/env python3
"""Audit bounded native runs, export the optional model, and render comparisons."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from identify_gas_spring import force
from wheeled_tasks.chassis.identified_gas_spring import IdentifiedGasSpringCurve


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.data.resolve()
    fit_path = out / "identified_model.json"
    candidates = out / "fitted_candidates.json"
    if not candidates.exists():
        shutil.copyfile(fit_path, candidates)
    fitted = json.loads(candidates.read_text())
    selected = fitted["models"][fitted["selected"]]
    geometry_meta = json.loads((out / "geometry.json").read_text())
    source_meta = json.loads((out / "real_data.json").read_text())
    g = np.load(out / "geometry.npz")
    data = np.load(out / "fit_data.npz")
    source_checks = {s["path"]: sha256(s["path"]) == s["sha256"] for s in source_meta["sources"]}
    assert all(source_checks.values()), "Raw source changed after fitting"
    bundle = Path(geometry_meta["bundle"])
    assert all(sha256(bundle / n) == h for n, h in geometry_meta["files_sha256"].items())
    probe_reports = []
    all_curve_cases = []
    for name in ("native_unit", "native_curves", "native_edge"):
        path = out / f"{name}.json"
        probe = json.loads(path.read_text())
        assert probe["status"] == "completed" and probe["all_cases_settled"]
        assert probe["gravity"] == [0., 0., 0.]
        assert not probe["spring_feedforward"] and probe["passive_pose_writes_during_physics"] == 0
        trace = np.load(path.with_suffix(".npz"))
        errors = []
        cases = []
        for index, row in enumerate(probe["cases"]):
            jac = np.array([np.interp(row["actual_beta_deg"], g["beta_deg"],
                                    g["jacobian_m_per_rad"][:, i]) for i in range(2)])
            residual = np.asarray(row["holding_torque_nm"]) + jac * row["force_n"]
            errors.extend(residual)
            samples = trace[f"case_{index}"]
            excess = np.maximum(selected["compression_domain_m"][0] - samples[:, 1],
                                samples[:, 1] - selected["compression_domain_m"][1])
            details = dict(model=row["model"], reference_beta_deg=row["reference_beta_deg"],
                           actual_beta_deg=row["actual_beta_deg"],
                           motor_mapping_error_nm=residual.tolist(),
                           strict_endpoint_hold_samples=int((excess > 0).sum()) if row["model"] != "unit" else 0,
                           peak_endpoint_excess_m=float(max(0., excess.max())) if row["model"] != "unit" else 0.)
            cases.append(details)
            if name != "native_unit":
                all_curve_cases.append(row)
                assert details["peak_endpoint_excess_m"] < 2e-5
        errors = np.array(errors)
        assert np.max(abs(errors)) < .05, "Native force-to-motor transmission mismatch"
        probe_reports.append(dict(file=path.name, sha256=sha256(path), cases=len(cases),
            script_sha256_at_run=probe["script_sha256"],
            motor_mapping_rms_nm=float(np.sqrt(np.mean(errors ** 2))),
            motor_mapping_abs_max_nm=float(abs(errors).max()),
            max_loop_gap_m=max(c["max_loop_gap_m"] for c in probe["cases"]),
            max_velocity_rms_rad_s=max(c["velocity_rms_rad_s"] for c in probe["cases"]),
            peak_motor_torque_nm=max(c["peak_motor_torque_nm"] for c in probe["cases"]),
            numerical_domain_tolerance_m=2e-5, details=cases))
    assert len(all_curve_cases) == 33
    curve = IdentifiedGasSpringCurve.from_json(candidates)
    s = torch.linspace(*selected["compression_domain_m"], 301, dtype=torch.float64, requires_grad=True)
    np.testing.assert_allclose(curve.force(s).detach().numpy(), force(selected, s.detach().numpy()), atol=1e-10)
    gradient = torch.autograd.grad(curve.potential_energy(s).sum(), s)[0]
    torch.testing.assert_close(gradient, curve.force(s), atol=1e-9, rtol=1e-10)
    exported = {key: value for key, value in selected.items()
                if not key.startswith("direction_") and key != "training_cells"}
    exported.update(schema_version=1, status="native_static_validated_effective_left_model",
        gas_pressure_pa=None, physical_pressure_identified=False, extrapolation="reject",
        extended_pin_distance_m=.232, stroke_m=.08,
        spring_slide_compression_at_q_zero_m=.06972718009112547,
        motor_order=["L_joint1", "LL_joint1"],
        holding_torque_rule="tau_hold_sim=-J_length.T*F; RMCS API torque has opposite sign",
        fitted_mean_load_only=True, direction_friction_enabled=False,
        production_default_replaced=False, training_started=False,
        source_sha256={source["path"]: source["sha256"] for source in source_meta["sources"]},
        asset_files_sha256=geometry_meta["files_sha256"],
        validation_report=str(out / "validation_receipt.json"))
    args.export_dir.mkdir(parents=True, exist_ok=True)
    export_path = args.export_dir.resolve() / "model.json"
    write_json(export_path, exported)
    receipt = dict(status="passed_bounded_static_validation", created_at=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        interpreter=sys.executable, source_hash_checks=source_checks,
        fitted_candidates_sha256=sha256(candidates), native=probe_reports,
        exported_model=str(export_path), exported_model_sha256=sha256(export_path),
        torch_force_parity=True, conservative_energy_gradient=True,
        tested_left_beta_domain_deg=[56., 101.], right_leg_identified=False,
        independent_motor_torque_sensor=False, gas_pressure_identified=False,
        dynamics_or_hardware_closed_loop_validated=False,
        numerical_armature_kg_m2=.02,
        armature_scope="native measurement fixture only; no production inertia change",
        source_script_hashes={n: sha256(ROOT / n) for n in (
            "tools/identify_gas_spring.py", "scripts/export_real_gas_spring_data.py",
            "scripts/probe_identified_gas_spring.py", "scripts/report_identified_gas_spring.py",
            "src/wheeled_tasks/chassis/identified_gas_spring.py", "tests/test_identified_gas_spring.py")})
    write_json(out / "validation_receipt.json", receipt)
    final_fit = dict(fitted, status=exported["status"], validation_receipt="validation_receipt.json",
                     validation_receipt_sha256=sha256(out / "validation_receipt.json"))
    write_json(fit_path, final_fit)
    render(out, fitted, g, data, all_curve_cases)
    report(out, fitted, receipt, export_path)
    print(json.dumps(dict(status=receipt["status"], native_cases=sum(p["cases"] for p in probe_reports),
                         export=str(export_path)), ensure_ascii=False), flush=True)


def render(out, fitted, g, data, native):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 10})
    names = ["catalogue", "monotone_cubic", "effective_monotone_table"]
    labels = ["原目录模型", "优化单调多项式", "辨识有效力曲线"]
    colors = ["#64748b", "#c38b00", "#c83232"]
    beta = np.linspace(56., 101., 600)
    compression = np.interp(beta, g["beta_deg"], g["compression_m"])
    jac = np.array([np.interp(beta, g["beta_deg"], g["jacobian_m_per_rad"][:, i]) for i in range(2)]).T
    mask = (data["source"] == 1) & (data["validation"] == 1)
    direction = np.sign(data["relative_velocity_rad_s"])
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10), constrained_layout=True)
    for index, (ax, field, title) in enumerate(zip(axes[0], ["hip_torque_nm", "knee_torque_nm"],
                                                   ["左 hip 电机保持力矩", "左 knee 电机保持力矩"])):
        for sign, color, label in [(1, "#008c95", "实测留出：相对速度 > 0"), (-1, "#9c68ad", "实测留出：相对速度 < 0")]:
            m = mask & (direction == sign)
            ax.scatter(data["beta_deg"][m], data[field][m], s=8, alpha=.25, color=color, label=label)
        for name, label, color in zip(names, labels, colors):
            ax.plot(beta, -jac[:, index] * force(fitted["models"][name], compression), color=color, lw=2, label=label)
            points = [c for c in native if c["model"] == name]
            ax.scatter([c["actual_beta_deg"] for c in points], [c["holding_torque_nm"][index] for c in points],
                       marker="D", s=26, facecolors="none", edgecolors=color, zorder=4)
        ax.set(xlabel="大小腿夹角 β [deg]", ylabel="模型坐标力矩 [N·m]", title=title, xlim=(55.5, 101.5))
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    axes[0, 1].text(.04, .95, "空心菱形：Isaac Sim 原生平衡点\n曲线：仅平均弹性负载；无方向摩擦", transform=axes[0, 1].transAxes,
                    va="top", fontsize=9, bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
    ax = axes[1, 0]
    for name, label, color in zip(names, labels, colors):
        ax.plot(compression[::-1] * 1000, force(fitted["models"][name], compression[::-1]), label=label, color=color, lw=2)
    selected = fitted["models"][fitted["selected"]]
    ax.scatter(np.array(selected["compression_knots_m"]) * 1000, selected["force_knots_n"], s=20, color=colors[-1])
    ax.set(xlabel="压缩量 [mm]（安装全伸长度 232 mm）", ylabel="轴向推力 [N]", title="有效安装负载曲线；尚未唯一分离纯气体压力")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    ax = axes[1, 1]
    x = np.arange(2)
    for i, (name, label, color) in enumerate(zip(names, labels, colors)):
        values = [fitted["validation"][name][str(s)]["mean_load_residual_rms_nm"] for s in (0, 1)]
        bars = ax.bar(x + (i - 1) * .23, values, width=.22, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    ax.set(xticks=x, xticklabels=["旧 PD 留出", "新 P/P 留出"], ylabel="差动力矩残差 RMS [N·m]",
           title="仅弹性曲线预测实际力矩（未补偿方向摩擦）", ylim=(0, 6.2))
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
    fig.suptitle("悬空水平左腿：实测与 Isaac Sim 气簧负载对照\n阶段 1–6 拟合；阶段 7 以后留出；仅 β=56–101°", fontsize=15)
    fig.savefig(out / "comparison.png", dpi=170)
    plt.close(fig)


def report(out, fit, receipt, exported):
    val = fit["validation"]
    labels = [("catalogue", "原目录气簧"), ("monotone_cubic", "优化单调多项式"), ("effective_monotone_table", "辨识有效力曲线")]
    table = "\n".join(f"| {label} | {val[k]['0']['mean_load_residual_rms_nm']:.3f} | {val[k]['1']['mean_load_residual_rms_nm']:.3f} | {val[k]['1']['with_direction_residual_rms_nm']:.3f} | {val[k]['1']['mean_joint_residual_rms_nm']:.3f} |" for k, label in labels)
    native_table = "\n".join(f"| {p['file']} | {p['cases']} | {p['motor_mapping_rms_nm']:.6f} | {p['max_loop_gap_m']*1000:.4f} | {p['max_velocity_rms_rad_s']:.6f} |" for p in receipt["native"])
    model = fit["models"][fit["selected"]]
    force_rows = "\n".join(f"| {s*1000:.3f} | {f:.2f} |" for s, f in zip(model["compression_knots_m"], model["force_knots_n"]))
    drop = 100 * (1 - val["effective_monotone_table"]["1"]["mean_load_residual_rms_nm"] / val["catalogue"]["1"]["mean_load_residual_rms_nm"])
    text = f"""# 左腿气簧 Real2Sim 标定（2026-10-27）

已完成真实力矩曲线拟合与 Isaac Sim 6 / PhysX 原生闭链验证。新 P/P 留出数据上，仅弹性负载模型的差动力矩残差 RMS 从 **5.114 降到 1.935 N·m，降低 {drop:.1f}%**。这是更贴近本次实测的左腿有效安装负载模型，尚不能称为已唯一辨识的纯气体压力模型。

![实测与原生仿真对照](comparison.png)

## 数据与拟合边界

用户确认水平完全悬空，无重力和外部负载。数据为 DM 上报力矩，未使用独立力传感器。两个原始 CSV 的 SHA-256 和逐文件来源见 [real_data.json](real_data.json)，已在最终导出时重新核验。旧文件 218309 条有效记录、末尾损坏行 218311 被排除；新文件 112144 条记录。共导出 4326 个预处理点，筛选后 2618 点。

只取慢扫：旧记录阶段 1–12、新记录 1–11，新记录不完整阶段 12 不用；每份数据的阶段 1–6 拟合，后续阶段留出。留出旧/新分别 619/610 点。先进行 51 点均值平滑和 101 点相对加速度平滑，避开 >2 ms 数据间隔前后 120 ms，再约每 20 ms 取点；筛选 |相对角速度|>0.06 rad/s、|相对角加速度|<3 rad/s²、|hip 角速度|<0.18 rad/s。训练时按往返周期、0.5° 角度、方向与来源分组等权，防止停滞点支配曲线。预处理源码哈希有记录。

仅覆盖 **左腿实际 β=56–101°、压缩量 22.997–61.786 mm**。未外推至机械全行程，未辨识右腿。当前 β 标定和 CAD 安装尺寸作为固定条件。

## 物理映射

采用当前 V6 `纯底盘_v6_232mm_stop105/urdf`，19 个刚体、18 个树关节、6 个球形闭链约束。左侧两个实际电机为 `L_joint1` 和 `LL_joint1`；`L_joint2` 是被动膝铰链，不能把它的反力当作 DM knee 力矩。V6 两电机相对角与 RMCS 角度表一致。

安装全伸销距 232 mm，行程 80 mm；压缩量 s=0.06972718009112547−q_slide。沿闭链求 J=∂l/∂[q_hip,q_knee]。气簧驱动力矩为 JᵀF，静态电机补偿为 **−JᵀF**。左腿仿真模型坐标是 [+T,−T]，RMCS 电机 API 坐标相反。方向必须在接入实车时按当前标定核对。

拟合标量 T=(τ_hip−τ_knee)/2。平均负载和方向相关的半宽 W(β) 同时估计，T_pred=T_spring(β)−W(β)sign(dq_knee−dq_hip)。W 只作为摩擦/密封滞回等的诊断项，**没有作为气簧轴向力导出**。两电机共同分量不能由一个轴向弹簧力解释，保留为残差。

## 留出验证（N·m）

| 模型 | 旧记录差动 RMS，仅弹性 | 新记录差动 RMS，仅弹性 | 新记录差动 RMS，另含方向项 | 新记录两个电机合并 RMS，仅弹性 |
|---|---:|---:|---:|---:|
{table}

“仅弹性”残差仍包含未补偿方向摩擦；0.426 N·m 需要额外方向项，不能当成气簧模型单独误差。两个电机合并 RMS 是 hip/knee 残差的均方合并。留出来自同一批实验后续往返，能够检验重复性，尚非第三次独立采集或高速验证。

原模型 F=280+122.735918491(s/0.08)²+53.969804409(s/0.08)³ N。非负系数的普通单调多项式也做了优化，其误差仍明显高于有效查表曲线，因此不能只把原气簧乘一个常数。

## 原生 Isaac Sim 验证

以下均在真实 Isaac Sim 6.0.0.1、Isaac Lab 6.1.14 / PhysX 中运行，无地面接触、固定基座、重力为零。物理步长 1 ms，求解器 64/32 次迭代；显式电机 PD=160/2.5，限幅 40 N·m。所有工况均达到静态判据。

| 原生记录 | 工况数 | 电机力矩与 −JᵀF 的 RMS [N·m] | 最大闭链间隙 [mm] | 最大末段速度 RMS [rad/s] |
|---|---:|---:|---:|---:|
{native_table}

6 个单位力工况使用 0/100 N 轴向力、无气簧前馈，自由达到平衡；用最终实际角度得到力到电机力矩的映射校准，比例约为 {fit['native_sensitivity_scale'][0]:.8f}/{fit['native_sensitivity_scale'][1]:.8f}。另外 33 个工况验证原模型、多项式和有效查表，包含 99/100/101° 末端。曲线工况使用固定位置参考预偏置让实际平衡角接近采样角；预偏置仅初始化一次，没有在线力矩前馈，也没有物理运行中覆写被动关节位置。这些工况验证载荷传递和数值平衡，不能替代独立传感器测力。

测量夹具对电机使用 0.02 kg·m² 数值 armature 以稳定显式控制，不代表已辨识真实惯量，没有写入生产惯量。零 armature 的两次不稳定尝试保留为 `native_unit_failed_*`，没有作为有效校准。闭链关节 incoming/projected force 包括约束反力，未当作纯气簧力矩使用。

数值端点允许最多 20 μm 的显式 endpoint hold，严格超域计数和实际超出量逐工况保存在 [validation_receipt.json](validation_receipt.json)。101° 附近的微小数值超域用端点力保持；导出的公共组件本身严格拒绝超域。初始 6/24 工况对应源码哈希记录在各 JSON；补充末端工况另保存了 `native_edge.source.py`，不会把后来修改的源码冒称为旧运行版本。

## 导出与使用

独立模型：`{exported}`。Torch 组件：`src/wheeled_tasks/chassis/identified_gas_spring.py`。查表保持正力、随压缩量非减，势能为同一分段线性力的精确积分，已核验 dU/ds=F；没有额外启用摩擦项。加载示例：

```python
from wheeled_tasks.chassis.identified_gas_spring import IdentifiedGasSpringCurve
curve = IdentifiedGasSpringCurve.from_json("model/gas_spring_identified_left_20261027/model.json")
force_n = curve.force(compression_m)  # positive outward along the spring slide
```

原生末端验证实际调用了这个 Torch 组件。模型不静默外推，右腿不可直接宣称已标定。当前默认训练气簧、生产合同和控制器未替换，未启动 PPO，未运行实车补偿闭环。

| 压缩量 [mm] | 有效轴向力 [N] |
|---:|---:|
{force_rows}

## 物理解读与尚未分离的因素

98–101° 处的有效力明显下降，特别是压缩约 23–26 mm 的段落；它能复现本批力矩，却不符合简单目录压力曲线的平缓形状。不能据此反推一个可信的单一气压。它可能还包含安装/角度标定、密封摩擦和滞回、两电机上报力矩偏差或未建模约束的影响。当前将其标记为 **effective mounted force**，`physical_pressure_identified=false`，不把摩擦写成可逆气体力。

阻尼、惯量、控制延迟需要另用高速数据在此平均载荷基础上辨识，本次静态模型不证明这些量已标定，也不证明实车加前馈后的跟踪改善。用于 RL 时可先作为该角域内的左腿候选；完整双腿训练需要右腿/其余角域验证和合同显式接入。

## 复现

先按 `scripts/export_real_gas_spring_data.py --help` 导出原始数据；Isaac 解释器为 `/home/noir/miniconda3/envs/isaacsim6/bin/python`。

```bash
python tools/identify_gas_spring.py geometry --out reports/gas_spring_real2sim_20261027
OMNI_KIT_ACCEPT_EULA=YES python scripts/probe_identified_gas_spring.py --data reports/gas_spring_real2sim_20261027 --mode unit --output reports/gas_spring_real2sim_20261027/native_unit.json
python tools/identify_gas_spring.py fit --out reports/gas_spring_real2sim_20261027 --native-probe reports/gas_spring_real2sim_20261027/native_unit.json
OMNI_KIT_ACCEPT_EULA=YES python scripts/probe_identified_gas_spring.py --data reports/gas_spring_real2sim_20261027 --mode curves --output reports/gas_spring_real2sim_20261027/native_curves.json
OMNI_KIT_ACCEPT_EULA=YES python scripts/probe_identified_gas_spring.py --data reports/gas_spring_real2sim_20261027 --mode curves --betas 99 100 101 --output reports/gas_spring_real2sim_20261027/native_edge.json
python scripts/report_identified_gas_spring.py --data reports/gas_spring_real2sim_20261027 --export-dir model/gas_spring_identified_left_20261027
python -m pytest -q tests/test_identified_gas_spring.py tests/test_gas_spring.py
```

以上命令会写同名结果；现有原始运行与失败证据应保留，复现建议用新的输出目录。全部输入、资产、脚本和导出模型 SHA-256 见回执。
"""
    (out / "README.md").write_text(text)


if __name__ == "__main__":
    main()
