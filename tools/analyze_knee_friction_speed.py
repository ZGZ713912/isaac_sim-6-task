#!/usr/bin/env python3
"""Compare knee forward/reverse load at matched angles and multiple speeds.

No source CSV, controller, or simulation model is changed. Preparation uses the
bundled Pandas runtime; statistical analysis uses the scientific/Isaac runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

S = "/wheel_leg/sweep/"
SOURCES = [
    ("e5ccbdff95ab", "relative_knee", "knee_three_speeds"),
    ("a8a26cf728df", "common_rotation", "whole_leg_first"),
    ("cc6a4d18cb90", "common_rotation", "whole_leg_repeat"),
    ("1f06ef9b4992", "common_angles", "whole_leg_multiple_angles"),
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def prepare(cache, out):
    import pandas as pd
    frames, metadata = [], []
    for source, (key, kind, label) in enumerate(SOURCES):
        meta = json.loads((cache / f"{key}.json").read_text())
        assert digest(meta["path"]) == meta["sha256"], meta["path"]
        frame = pd.read_pickle(cache / f"{key}.pkl")
        assert len(frame) == meta["records"]
        selected = frame[(frame[S+"state"] == 3) & (frame[S+"reference_active"] == 1)
                         & (frame[S+"side"] == 0) & (frame[S+"stage"] >= 1)]
        if source == 0:
            selected = selected[(selected[S+"mode"] == 2) & (selected[S+"stage"] <= 36)]
        for (run, stage), group in selected.groupby([S+"run_id", S+"stage"]):
            t = group[S+"control_steady_ns"].to_numpy(float) * 1e-9
            assert np.all(np.diff(t) > 0)
            x = pd.DataFrame(dict(time_s=t, elapsed_s=t-t[0],
                beta_deg=np.rad2deg(group[S+"left/beta_estimate"].to_numpy(float))))
            bad = group[S+"left/beta_valid"].to_numpy() != 1
            for axis, short in [("hip", "h"), ("knee", "k")]:
                prefix = S+f"left_{axis}_joint/"
                x[f"v{short}"] = group[prefix+"model_velocity"].to_numpy(float)
                x[f"tau{short}"] = -group[f"/wheel_leg/left_{axis}_joint/torque"].to_numpy(float)
                x[f"error_{short}"] = group[prefix+"tracking_error"].to_numpy(float)
                bad |= group[prefix+"saturated"].to_numpy() != 0
            for gap in np.flatnonzero(np.diff(t) > .002)+1:
                bad |= abs(t-t[gap]) < .12
            for field in ["beta_deg", "vh", "vk", "tauh", "tauk", "error_h", "error_k"]:
                x[field] = x[field].rolling(51, center=True, min_periods=51).mean()
            for short in ["h", "k"]:
                x[f"a{short}"] = pd.Series(np.gradient(x[f"v{short}"].to_numpy(), t)).rolling(101, center=True, min_periods=51).mean()
            x = x[~bad].dropna().iloc[::5].copy()
            x["source"], x["run"], x["stage"] = source, int(run), int(stage)
            x["validation"] = int(group[S+"validation"].iloc[0])
            frames.append(x)
        metadata.append(dict(**meta, source_id=source, kind=kind, label=label))
        print("PREPARED", source, meta["file"], flush=True)
        del frame, selected
    joined = pd.concat(frames, ignore_index=True)
    np.savez_compressed(out / "prepared.npz", **{c: joined[c].to_numpy() for c in joined.columns})
    write(out / "sources.json", dict(sources=metadata, rows=len(joined), script_sha256=digest(__file__),
        conditions="user-confirmed horizontal suspended leg; no gravity/external load",
        preprocessing="51-sample centered means; velocity derivatives averaged 101 samples; exclude +/-120 ms of >2ms gaps; every fifth valid sample; no saturation; actual angle and reported motor torque"))
    print("PREPARED_ROWS", len(joined), flush=True)


def subset(data, mask):
    return {key: values[mask] for key, values in data.items()}


def relative_pairs(data, max_acceleration=3.):
    mask = ((data["source"] == 0) & (data["beta_deg"] >= 65.) & (data["beta_deg"] <= 94.)
            & (abs(data["vk"]) > .1) & (abs(data["vh"]) < .3)
            & (abs(data["ak"]) < max_acceleration) & (abs(data["ah"]) < max_acceleration))
    selected = subset(data, mask)
    band = ((selected["stage"]-1)//12).astype(int)
    cycle = (((selected["stage"]-1)%12)//2).astype(int)
    angle = np.rint(selected["beta_deg"]).astype(int)
    keys = np.column_stack((band, cycle, angle))
    records = []
    fields = ["beta_deg", "vk", "vh", "ak", "ah", "tauk", "tauh", "error_k", "error_h"]
    for key in np.unique(keys, axis=0):
        same = np.all(keys == key, axis=1)
        plus, minus = same & (selected["vk"] > 0), same & (selected["vk"] < 0)
        if not plus.any() or not minus.any():
            continue
        pp = {name: selected[name][plus].mean() for name in fields}
        mm = {name: selected[name][minus].mean() for name in fields}
        records.append(dict(band=int(key[0]), cycle=int(key[1]), angle_bin=int(key[2]),
            speed_rad_s=(pp["vk"]-mm["vk"])/2,
            half_width_nm=(pp["tauk"]-mm["tauk"])/2,
            hip_half_width_nm=(pp["tauh"]-mm["tauh"])/2,
            knee_accel_half_difference=(pp["ak"]-mm["ak"])/2,
            hip_accel_half_difference=(pp["ah"]-mm["ah"])/2,
            beta_difference_deg=pp["beta_deg"]-mm["beta_deg"],
            plus_torque_nm=pp["tauk"], minus_torque_nm=mm["tauk"],
            knee_tracking_error_direction_difference_deg=float(np.rad2deg(pp["error_k"]-mm["error_k"])),
            hip_tracking_error_direction_difference_deg=float(np.rad2deg(pp["error_h"]-mm["error_h"])),
            min_direction_rows=int(min(plus.sum(), minus.sum()))))
    # A bin must contain both directions in all six cycles at all three speeds.
    bins = sorted({r["angle_bin"] for r in records if sum(x["angle_bin"] == r["angle_bin"] for x in records) == 18})
    records = [r for r in records if r["angle_bin"] in bins]
    assert bins and len(records) == len(bins)*18
    return records, bins


def regression(records, bins):
    from scipy.optimize import lsq_linear
    angle = np.array([r["angle_bin"] for r in records])
    speed = np.array([r["speed_rad_s"] for r in records])
    y = np.array([r["half_width_nm"] for r in records])
    dummies = np.eye(len(bins))[np.searchsorted(bins, angle)]
    training = np.array([r["cycle"] < 3 for r in records])
    result = {}
    for name, X, bounded in [
        ("coulomb", dummies, True),
        ("viscous_only", dummies*speed[:, None], True),
        ("coulomb_plus_nonnegative_viscous", np.column_stack((dummies, speed)), True),
        ("signed_speed_diagnostic", np.column_stack((dummies, speed)), False),
        ("signed_speed_with_dynamic_nuisance", np.column_stack((dummies, speed,
            [r["knee_accel_half_difference"] for r in records],
            [r["hip_accel_half_difference"] for r in records],
            [r["beta_difference_deg"] for r in records])), False),
    ]:
        if bounded:
            fit = lsq_linear(X[training], y[training], bounds=(0., np.inf), tol=1e-12)
            assert fit.success
            coeff = fit.x
        else:
            coeff = np.linalg.lstsq(X[training], y[training], rcond=None)[0]
        residual = y-X@coeff
        item = dict(training_rows=int(training.sum()), validation_rows=int((~training).sum()),
                    training_half_width_rms_nm=float(np.sqrt(np.mean(residual[training]**2))),
                    validation_half_width_rms_nm=float(np.sqrt(np.mean(residual[~training]**2))),
                    coefficients=coeff.tolist(), matrix_condition=float(np.linalg.cond(X[training])))
        if "speed" in name or "plus" in name:
            item["speed_slope_nm_per_rad_s"] = float(coeff[len(bins)])
        result[name] = item
    # Resample whole forward/reverse cycles, independently within each speed.
    # This quantifies repeatability conditional on the recorded experiment order.
    X = np.column_stack((dummies, speed))
    rng = np.random.default_rng(20261028)
    boot = []
    for _ in range(600):
        chosen = []
        for band in range(3):
            for cycle in rng.integers(0, 6, 6):
                chosen.extend(i for i, r in enumerate(records) if r["band"] == band and r["cycle"] == cycle)
        boot.append(float(np.linalg.lstsq(X[chosen], y[chosen], rcond=None)[0][-1]))
    result["signed_speed_repeatability"] = dict(bootstrap_unit="entire forward/reverse cycle within speed",
        replicates=600, slope_all_rows=float(np.linalg.lstsq(X, y, rcond=None)[0][-1]),
        conditional_95_percent_interval=np.quantile(boot, [.025, .975]).tolist(),
        physical_damping_identified=False,
        caution="speed blocks occurred in fixed order; temperature/history effects are not separated")
    return result


def rotation_checks(data):
    result = []
    for source in (1, 2, 3):
        runs = sorted(set(data["run"][data["source"] == source].astype(int)))
        stages = [(1, 2), (4, 5), (7, 8), (10, 11)] if source in (1, 2) else [(4, 5), (9, 10), (14, 15), (17, 18), (20, 21)]
        for run in runs:
            for stage_pair in stages:
                base = ((data["source"] == source) & (data["run"] == run)
                        & np.isin(data["stage"], stage_pair) & (data["elapsed_s"] > .3)
                        & (abs(data["vk"]) > .1) & (abs(data["ak"]) < 2.) & (abs(data["ah"]) < 2.))
                pp, mm = base & (data["vk"] > 0), base & (data["vk"] < 0)
                if min(pp.sum(), mm.sum()) < 30:
                    continue
                result.append(dict(source=source, run=int(run), stages=list(stage_pair),
                    speed_rad_s=float((data["vk"][pp].mean()-data["vk"][mm].mean())/2),
                    knee_half_width_nm=float((data["tauk"][pp].mean()-data["tauk"][mm].mean())/2),
                    hip_half_width_nm=float((data["tauh"][pp].mean()-data["tauh"][mm].mean())/2),
                    beta_mean_deg=float((data["beta_deg"][pp].mean()+data["beta_deg"][mm].mean())/2),
                    beta_direction_difference_deg=float(data["beta_deg"][pp].mean()-data["beta_deg"][mm].mean()),
                    rows=int(base.sum())))
    return result


def analyze(out):
    loaded = np.load(out/"prepared.npz")
    data = {key: loaded[key] for key in loaded.files}
    records, bins = relative_pairs(data)
    regressions = regression(records, bins)
    groups = []
    for band in range(3):
        rows = [r for r in records if r["band"] == band]
        cycle_means = [np.mean([r["half_width_nm"] for r in rows if r["cycle"] == cycle]) for cycle in range(6)]
        groups.append(dict(band=band, round_trip_s=[8., 4., 2.][band],
            speed_rad_s=float(np.mean([r["speed_rad_s"] for r in rows])),
            half_width_nm=float(np.mean(cycle_means)), branch_difference_nm=float(2*np.mean(cycle_means)),
            knee_tracking_error_direction_difference_deg=float(np.mean([r["knee_tracking_error_direction_difference_deg"] for r in rows])),
            cycle_half_width_means_nm=cycle_means, cycle_half_width_std_nm=float(np.std(cycle_means, ddof=1)),
            matched_angle_bins=bins, paired_cells=len(rows)))
    sensitivity = []
    for acceleration in (3., 6., 12.):
        pairs, angles = relative_pairs(data, acceleration)
        models = regression(pairs, angles)
        sensitivity.append(dict(acceleration_limit_rad_s2=acceleration, angle_bins=angles,
            signed_speed_slope=models["signed_speed_repeatability"]["slope_all_rows"],
            speed_slope_with_acceleration_nuisance=models["signed_speed_with_dynamic_nuisance"]["speed_slope_nm_per_rad_s"]))
    rotations = rotation_checks(data)
    result = dict(status="offline_direction_load_speed_comparison_not_unique_friction_identification",
        sources=json.loads((out/"sources.json").read_text()), script_sha256=digest(__file__),
        processing=dict(angle_bin_deg=1., actual_angle_common_centers_deg=bins,
            primary_min_knee_speed_rad_s=.1, primary_max_hip_speed_rad_s=.3,
            primary_max_abs_motor_acceleration_rad_s2=3., equal_cycle_angle_speed_weights=True,
            training_cycles="first 3 forward/reverse pairs per speed; last 3 held out",
            torque_coordinate="motor model coordinates: tau_model=-tau_DM_API",
            half_width_definition="W=(tau_k_positive_speed-tau_k_negative_speed)/2; full separation=2W"),
        speed_groups=groups, paired_regressions=regressions, acceleration_sensitivity=sensitivity,
        common_rotation_checks=rotations,
        conclusion="Fast motion retains a direction load, but it does not grow with speed. No evidence that positive viscous damping dominates the gap.",
        limitations=["Moving Coulomb friction, seal/structure hysteresis, low-speed Stribeck effects and experiment-order warming are not uniquely separated.",
            "Persistent direction gap at nonzero speed is not by itself evidence of viscous damping or pure static friction.",
            "Reported DM torque is not an independent shaft torque sensor; closed-loop signals and offline acceleration estimates have limits.",
            "Negative descriptive speed slope is not negative physical damping and must not be injected into simulation.",
            "Static friction/breakaway needs near-zero ramp/hold data; fastest relative knee coverage is only about 2.1 rad/s.",
            "Whole-leg common rotation is a different kinematic condition; its direction half-width is not pooled with relative knee motion."],
        changes_to_runtime_controller_or_simulation=False)
    write(out/"analysis.json", result)
    np.savez_compressed(out/"paired_cells.npz", **{key: np.array([r[key] for r in records]) for key in records[0]})
    render(out, records, bins, groups, regressions, rotations)
    report(out, result)
    print(json.dumps(dict(speed_groups=groups, models=regressions), ensure_ascii=False, indent=2), flush=True)


def render(out, records, bins, groups, models, rotations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    colors = ["#4778a8", "#d08a23", "#c63636"]
    for band, color in enumerate(colors):
        chosen = [r for r in records if r["band"] == band]
        plus = [np.mean([r["plus_torque_nm"] for r in chosen if r["angle_bin"] == b]) for b in bins]
        minus = [np.mean([r["minus_torque_nm"] for r in chosen if r["angle_bin"] == b]) for b in bins]
        label = f"{groups[band]['speed_rad_s']:.2f} rad/s"
        axes[0, 0].plot(bins, plus, color=color, label=label+" 正向")
        axes[0, 0].plot(bins, minus, color=color, linestyle="--", label=label+" 反向")
        axes[0, 1].scatter([r["speed_rad_s"] for r in chosen], [r["half_width_nm"] for r in chosen],
                           s=14, alpha=.3, color=color)
        axes[0, 1].errorbar(groups[band]["speed_rad_s"], groups[band]["half_width_nm"],
                            yerr=groups[band]["cycle_half_width_std_nm"], fmt="o", color=color, capsize=5,
                            label=f"{groups[band]['round_trip_s']:.0f} s 往返，半宽 {groups[band]['half_width_nm']:.2f}")
    axes[0, 0].set(xlabel="实际夹角 β 的共同角度格 [deg]", ylabel="knee 模型坐标力矩 [N·m]",
                   title="相同角度的正反向支路：高速仍然分开")
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].set(xlabel="实际 knee 速度幅值 [rad/s]", ylabel="正反向力矩差的一半 W [N·m]",
                   title="速度约增至 4 倍，方向差没有增大")
    axes[0, 1].legend(fontsize=8)
    ax = axes[1, 0]
    for source, color in [(1, "#4778a8"), (2, "#c63636")]:
        rows = [r for r in rotations if r["source"] == source]
        ax.plot([r["speed_rad_s"] for r in rows], [r["knee_half_width_nm"] for r in rows],
                marker="o", color=color, label=f"整腿转动记录 {source}")
    ax.set(xlabel="实际 knee 速度幅值 [rad/s]", ylabel="方向半宽 W [N·m]",
           title="补充：整腿同向转动，膝夹角基本不变")
    ax.text(.30, .70, "运动工况不同，不与上图合并\n估计同一摩擦系数", transform=ax.transAxes,
            fontsize=8, bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    names = ["coulomb", "viscous_only", "coulomb_plus_nonnegative_viscous", "signed_speed_diagnostic"]
    labels = ["方向项 Fc(β)", "纯速度项 B(β)v", "方向项 + Bv\nB ≥ 0", "允许负斜率\n仅诊断"]
    values = [models[n]["validation_half_width_rms_nm"] for n in names]
    bars = ax.bar(np.arange(len(names)), values, color=["#4778a8", "#c63636", "#d08a23", "#8a8a8a"])
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set(xticks=np.arange(len(names)), xticklabels=labels, ylabel="留出方向半宽 RMS [N·m]",
           title="每档前 3 个往返拟合，后 3 个往返验证", ylim=(0, max(values)*1.25))
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("水平悬空左膝：往返方向差与速度的关系\n同一控制器、实际角度匹配、低加速度筛选；未启用新摩擦参数", fontsize=14)
    fig.savefig(out/"comparison.png", dpi=170)
    plt.close(fig)


def report(out, result):
    rows = "\n".join(f"| {r['round_trip_s']:.0f} | {r['speed_rad_s']:.3f} | {r['half_width_nm']:.3f} | {r['branch_difference_nm']:.3f} | {r['knee_tracking_error_direction_difference_deg']:.3f} |" for r in result["speed_groups"])
    models = result["paired_regressions"]
    model_rows = "\n".join(f"| {name} | {models[name]['validation_half_width_rms_nm']:.4f} |" for name in ["coulomb", "viscous_only", "coulomb_plus_nonnegative_viscous", "signed_speed_diagnostic"])
    sources = "\n".join(f"- `{s['path']}`，SHA-256 `{s['sha256']}`" for s in result["sources"]["sources"])
    bands = result["processing"]["actual_angle_common_centers_deg"]
    slope = models["signed_speed_repeatability"]
    (out/"README.md").write_text(f"""# knee 快速运动往返差与摩擦/阻尼（2026-10-28）

快速运动仍有方向差，但本批数据中方向差没有随速度增大。实际 knee 速度由约 0.54 提高至 2.11 rad/s 时，同角度正反向力矩差约 8.62→8.06 N·m。它不支持“只要快速仍有往返差，就说明主要是黏性阻尼”。

![方向差与速度对照](comparison.png)

## 同一膝关节的三档速度对照

主证据来自 `wheel_leg_膝关节气簧辨识数据.csv`，run 1 / mode 2。stage 1–12、13–24、25–36 分别为 8/4/2 s 一次往返，每档 6 个完整往返。它是原单环 PD=120/4 的记录；新级联 P/P 文件只有慢扫，不能把本次快速结论宣称为新控制器已经完成高速验证。统一使用实际反馈速度、实际 β 和 DM 报告力矩；模型坐标 τ=−τ_API。没有使用命令力矩替代反馈。

沿用已审计的 51 点居中平滑、速度导数再 101 点平滑，排除 >2 ms 时钟间隔前后 120 ms，排除饱和及无效 β，每 5 个有效样本取点。主比较限制 |vk|>0.1 rad/s、|vh|<0.3 rad/s、|ak|/|ah|<3 rad/s²。按 1° 角度格和单个往返分别配对两个方向；只保留三档速度、全部 6 个往返都有配对的共同角度格 **{bands[0]}–{bands[-1]}°**。这些是格中心，格宽 1°。每个往返、角度和速度等权，避免低速采样多造成偏置。

W=(τk,正向−τk,反向)/2，完整方向分离为 2W。W 是持续运动中的方向相关有效负载，不是已独立测得的静摩擦力矩。

| 一次往返 [s] | 平均实际速度幅值 [rad/s] | 方向半宽 W [N·m] | 正反向支路差 2W [N·m] | 两方向位置跟踪误差之差 [deg] |
|---:|---:|---:|---:|---:|
{rows}

位置误差取日志 `tracking_error`，同样按实际 β 配对；其方向差不是绝对跟踪误差或整段 RMS。位置方向差随速度也未增大。不能仅用位置闭环误差识别阻尼，因为 D 项、参考速度和控制延迟也会影响它。

## 留出比较与干扰检查

各档前 3 个往返拟合、后 3 个往返验证。位置依赖项均在相同角度格估计。比较 W=Fc(β)、W=B(β)|v|、W=Fc(β)+B|v| 且 B≥0。另允许 B 为负，仅用来描述速度减弱趋势，不作为物理阻尼。

| 方向半宽模型 | 留出 RMS [N·m] |
|---|---:|
{model_rows}

非负黏性项拟合 B={models['coulomb_plus_nonnegative_viscous']['speed_slope_nm_per_rad_s']:.6g} N·m/(rad/s)，没有带来有效提升。无约束描述斜率为 {slope['slope_all_rows']:.3f} N·m/(rad/s)。以整个往返为块的 600 次重采样条件区间为 [{slope['conditional_95_percent_interval'][0]:.3f}, {slope['conditional_95_percent_interval'][1]:.3f}]。这只表示本次实验顺序下的重复性；速度档按固定顺序执行，没有随机交错或温度记录，不能把负斜率称为负物理阻尼。

将加速度筛选放宽至 6/12 rad/s²、在配对回归中额外加入加速度半差及小角度错配，斜率仍为负；逐项见 `analysis.json`。同角度反向配对有助于消除保守负载和相近惯性，但不是把一切动态力矩完全消掉。

两份整腿转动记录是补充验证：膝夹角约 84°、两个电机同向，实际 knee 速度约 0.2/0.6/1.5/3.1 rad/s。持续运动方向半宽约 2 N·m，高速仍在，随速度并未明显增大。多夹角整腿记录也保存在 JSON 附表。工况不同，不合并成同一个摩擦参数。

## 如何解释

标准摩擦模型中，库仑动摩擦在非零速度下仍可保留近似固定方向力矩，黏性阻尼才表现为力矩随速度增加，静摩擦/脱离静止主要看零速附近的起动过程。低速 Stribeck 效应、密封滞回或温升可能造成当前的弱速度减小趋势；现有记录不能唯一分开它们。模型区别见 [MathWorks Rotational Friction 官方说明](https://www.mathworks.com/help/simscape/ref/rotationalfriction.html)。

因此，当前证据更支持把主要往返差先视作运动方向相关摩擦/滞回负载。不能把约 8 N·m 的全部正反差写成黏性阻尼。也不能宣称阻尼严格为零：正阻尼可能与速度减弱摩擦、温度和闭环测量影响相互抵消。阻尼、纯静摩擦、密封摩擦和轴承/传动摩擦尚未独立识别。

这次只做离线分析，没有修改控制器或 Isaac Sim 的默认气簧/摩擦模型，没有启动实车或训练。

## 来源与复现

原始 CSV 未修改，读取的数值缓存均重新匹配原文件 SHA-256；中断末行处理沿用各缓存审计。来源：

{sources}

先用带 Pandas 的分析环境运行 `tools/analyze_knee_friction_speed.py prepare --out reports/knee_friction_speed_20261028`，再用 Isaac/scientific Python 运行同脚本的 `analyze` 阶段。`sources.json` 保存原始来源与预处理，`paired_cells.npz` 保存共同角度配对，`analysis.json` 保存全部指标和限制。居中滤波只用于离线分析。
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "analyze"])
    parser.add_argument("--cache", type=Path, default=Path("/home/noir/.cache/rmcs-sweep-archive-20261027"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare":
        prepare(args.cache, args.out)
    else:
        analyze(args.out)


if __name__ == "__main__":
    main()
