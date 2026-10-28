#!/usr/bin/env python3
"""Export the user's left/right symmetry assumption after native validation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from identify_gas_spring import force


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    native_path = args.out / "native_symmetric.json"
    native = json.loads(native_path.read_text())
    fitted_path = args.data / "identified_model.json"
    fitted = json.loads(fitted_path.read_text())
    model = fitted["models"][fitted["selected"]]
    assert native["status"] == "completed" and native["all_cases_settled"]
    assert native["right_leg_symmetry_assumed"] and not native["right_leg_measured"]
    assert native["gravity"] == [0., 0., 0.] and len(native["cases"]) == 7
    assert native["identified_model_sha256_at_start"] == digest(fitted_path)
    assert native["force_component_sha256"] == digest(ROOT / "src/wheeled_tasks/chassis/identified_gas_spring.py")
    assert native["geometry_tool_sha256"] == digest(ROOT / "tools/identify_gas_spring.py")
    geometries = {"left": np.load(args.data / "geometry.npz"),
                  "right": np.load(native_path.with_suffix(".right_geometry.npz"))}
    mapping_errors = {side: [] for side in geometries}
    mirror_errors = []
    for row in native["cases"]:
        for side, geometry in geometries.items():
            details = row if side == "left" else row["right"]
            jac = np.array([np.interp(details["actual_beta_deg"], geometry["beta_deg"],
                                    geometry["jacobian_m_per_rad"][:, i]) for i in range(2)])
            mapping_errors[side].extend(np.asarray(details["holding_torque_nm"]) + jac * details["force_n"])
        mirror_errors.extend(row["motor_torque_mirror_error_nm"])
    assert max(abs(np.array(mirror_errors))) < .01
    assert all(max(abs(np.array(v))) < .05 for v in mapping_errors.values())
    data = np.load(args.data / "fit_data.npz")
    mask = (data["source"] == 1) & (data["validation"] == 1)
    geometry = geometries["left"]
    beta = data["beta_deg"][mask]
    compression = np.interp(beta, geometry["beta_deg"], geometry["compression_m"])
    jac = np.array([np.interp(beta, geometry["beta_deg"], geometry["jacobian_m_per_rad"][:, i]) for i in range(2)]).T
    actual = np.column_stack([data["hip_torque_nm"][mask], data["knee_torque_nm"][mask]])
    measured = {}
    for name in ("catalogue", fitted["selected"]):
        prediction = -jac * force(fitted["models"][name], compression)[:, None]
        error = actual - prediction
        measured[name] = dict(hip_rms_nm=float(np.sqrt(np.mean(error[:, 0] ** 2))),
                             knee_rms_nm=float(np.sqrt(np.mean(error[:, 1] ** 2))),
                             combined_rms_nm=float(np.sqrt(np.mean(error ** 2))))
    test_log = args.out / "tests.log"
    test_text = test_log.read_text()
    assert "failed" not in test_text
    passed = int(re.search(r"(\d+) passed", test_text).group(1))
    assert passed >= 19
    original_path = ROOT / "model/gas_spring_identified_left_20261027/model.json"
    exported = json.loads(original_path.read_text())
    exported.update(status="native_static_validated_symmetric_assumption",
        calibration_side="left", application_sides=["left", "right"],
        right_leg_symmetry_assumed=True, right_leg_measured=False,
        right_leg_authority="User instruction: temporarily assume right leg symmetric to left",
        motor_order_by_side={"left": ["L_joint1", "LL_joint1"], "right": ["R_joint1", "RR_joint1"]},
        holding_torque_rule="Use each leg's own -J_length.T*F; mirror geometry changes motor signs, axial force stays positive",
        calibration_validation_report=exported["validation_report"],
        validation_report=str((args.out / "validation_receipt.json").resolve()))
    args.export_dir.mkdir(parents=True, exist_ok=True)
    export_path = args.export_dir / "model.json"
    write(export_path, exported)
    receipt = dict(status="passed_native_bilateral_static_assumption", created_at=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        native_sha256=digest(native_path), native_cases=len(native["cases"]),
        native_script_sha256=native["script_sha256"],
        fitted_left_input_sha256=digest(fitted_path), original_left_export_sha256=digest(original_path),
        right_leg_measured=False, right_leg_symmetry_assumed=True,
        beta_domain_deg=model["beta_domain_deg"], gravity=native["gravity"],
        motor_mirror_error_abs_max_nm=float(max(abs(np.array(mirror_errors)))),
        mapping_rms_nm={side: float(np.sqrt(np.mean(np.array(v) ** 2))) for side, v in mapping_errors.items()},
        max_loop_gap_m=max(c["max_loop_gap_m"] for c in native["cases"]),
        left_measured_validation=measured, measured_validation_rows=int(mask.sum()),
        tests_passed=passed, tests_log_sha256=digest(test_log),
        exported_model=str(export_path.resolve()), exported_model_sha256=digest(export_path),
        production_default_replaced=False, training_started_by_this_task=False,
        source_hashes={path: digest(ROOT / path) for path in [
            "src/wheeled_tasks/chassis/identified_gas_spring.py", "tools/identify_gas_spring.py",
            "scripts/probe_identified_gas_spring.py", "scripts/report_symmetric_gas_spring.py",
            "tests/test_identified_gas_spring.py"]})
    write(args.out / "validation_receipt.json", receipt)
    lines = "\n".join(f"| {r['reference_beta_deg']:.0f} | {r['holding_torque_nm'][0]:.3f} | {r['holding_torque_nm'][1]:.3f} | {r['right']['holding_torque_nm'][0]:.3f} | {r['right']['holding_torque_nm'][1]:.3f} |" for r in native["cases"])
    result = measured[fitted["selected"]]
    (args.out / "README.md").write_text(f"""# 气簧左右对称假设与无重力电机力矩验证

按用户最新指示，右腿暂时沿用左腿轴向力曲线。这是对称假设，不是新增右腿实测辨识。没有修改左腿拟合值或原始 39 个原生工况记录。

本次新增 7 个双腿同时受力的 Isaac Sim 6 原生工况，全部达到静态条件。两边均在各自气簧滑动关节施加同一 F(s)，由各自闭链几何传递到 hip/knee；正轴向推力不人为翻号。左右两侧电机坐标方向相反，力矩镜像差绝对值最大 {receipt['motor_mirror_error_abs_max_nm']:.6f} N·m，抽样闭链间隙最大 {receipt['max_loop_gap_m']*1000:.4f} mm。维持前一报告相同的无重力、固定基座、无地面接触、无摩擦、测量夹具 armature、PD 与限幅；使用固定位置参考预偏置，无在线气簧力矩前馈，运行中不覆写被动关节位置。

| β 参考 [deg] | 左 hip [N·m] | 左 knee [N·m] | 右 hip [N·m] | 右 knee [N·m] |
|---:|---:|---:|---:|---:|
{lines}

新 P/P 左腿留出 610 点的平均弹性负载预测误差：hip RMS **{result['hip_rms_nm']:.3f} N·m**，knee RMS **{result['knee_rms_nm']:.3f} N·m**，合并 **{result['combined_rms_nm']:.3f} N·m**。原目录模型分别为 {measured['catalogue']['hip_rms_nm']:.3f}/{measured['catalogue']['knee_rms_nm']:.3f} N·m。可以认为平均负载已经接近本次实测，但不能称逐时刻完全一致；knee 剩余方向摩擦误差更明显。之前 0.426 N·m 是差动模型另加方向项后的结果，该项没有作为气簧力启用。

适用左/右候选角域仍为 56–101°；右腿有效性来自用户的对称假设。模型依旧标记为有效安装负载，未唯一识别纯气体压力。101° 浮点端点按前一探针规则最多容许 20 μm endpoint hold，公共组件严格拒绝超域。

独立导出：`{export_path.resolve()}`。默认训练入口仍加载资产目录下 `fit_10mpa.json`，本次没有替换生产默认模型或启动 PPO。加载双腿候选示例：

```python
from wheeled_tasks.chassis.identified_gas_spring import IdentifiedGasSpringCurve
left = IdentifiedGasSpringCurve.from_json("model/gas_spring_identified_symmetric_20261028/model.json")
right = left.for_side("right", assume_symmetry=True)
force_left_n = left.force(compression_left_m)
force_right_n = right.force(compression_right_m)
```

{passed} 项相关测试通过，包括右侧使用须显式记录对称假设、轴向力符号、势能与闭链镜像映射。CAD 轴的微小左右差异低于约 10 ppm，不要求几何浮点数逐位相等。第一次数值检查误设了过紧容差、初次探针错用了旧右膝拼写，失败证据均保留；修正后结果见 `tests.log`、`native_symmetric.json`、`validation_receipt.json`。

复现（使用 Isaac Sim 环境解释器）：

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/probe_identified_gas_spring.py --data reports/gas_spring_real2sim_20261027 --mode curves --models effective_monotone_table --paired --betas 56 60 75 90 98 100 101 --output reports/gas_spring_real2sim_20261027/symmetric_20261028/native_symmetric.json
python -m pytest -q tests/test_identified_gas_spring.py tests/test_gas_spring.py
python scripts/report_symmetric_gas_spring.py --data reports/gas_spring_real2sim_20261027 --out reports/gas_spring_real2sim_20261027/symmetric_20261028 --export-dir model/gas_spring_identified_symmetric_20261028
```

原始左腿拟合、来源和完整对照图见 [上一报告](../README.md)。
""")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
