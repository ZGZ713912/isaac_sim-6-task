# 轮腿轮端减速比口径（15.62 / 15.8）

2026-10-26 核对用户提出的“wheel_leg 两个轮子 wheel 减速比 15.62”。结论：**实车侧正确，训练合同里的 15.8 只是历史元数据**。用户同日选择方案 A：只更新文档与注释，不改 design JSON、合同校验、测试或辨识脚本，不重签哈希。本文件即该决定的口径记录。

## 1. 实车（RMCS）：15.62，双轮、只换算一次

- [wheel-leg.cpp](/home/noir/Documents/workspace/RMCS/rmcs_ws/src/rmcs_core/src/hardware/wheel-leg.cpp:409) 第 409–414 行：左右两轮在同一循环里显式 `set_reduction_ratio(15.62)` 并 `set_reversed()`；注释写明 M3508 原装减速箱已拆除，外部减速箱承担编码器到轮端的全部减速。
- 不写这句时，[dji_motor.hpp](/home/noir/Documents/workspace/RMCS/rmcs_ws/src/rmcs_core/src/hardware/device/dji_motor.hpp:33) 对 kM3508 默认 3591/187 ≈ 19.2032，会差 23%。该默认值是历史通用配置，不是本车口径。
- 换算各一次：角度 ×1/15.62、速度 ×1/15.62、力矩/电流 ×15.62（dji_motor.hpp 第 92/96/125 行）。`torque_constant = 0.3×187/3591 = 0.015622 N·m/A` 已是电机侧，不再乘第二次。
- 轮端满流力矩 = 15.62 × 0.015622 × 20 ≈ **4.88 N·m**；部署侧软件上限 [policy.hpp](/home/noir/Documents/workspace/RMCS/rmcs_ws/src/rmcs_core/src/controller/chassis/wheel_leg/policy.hpp:26) 为 4.5 N·m，与仿真 `effort_limit=4.5` 一致（软件帽小于物理帽）。
- 控制律 [action.cpp](/home/noir/Documents/workspace/RMCS/rmcs_ws/src/rmcs_core/src/controller/chassis/wheel_leg/action.cpp:91) 在轮端 rad/s 与轮端 N·m 计算 PD，再交给 DjiMotor 换算，未二次乘减速比。该 15.62 于 2026-10-14（255088d0）合入；RMCS 仓内已无 15.8 残留。

## 2. 本训练仓库：仍写 15.8，但只做身份记录

| 位置 | 值 | 是否参与动力学 |
|---|---|---|
| [scratch_standing_base_design_v1.json](contracts/scratch_standing_base_design_v1.json:1784) `actuators.wheel.gear_ratio` | 15.8 | 否，provenance |
| [control_contract.py](src/wheeled_tasks/chassis/control_contract.py:69) 身份断言 | 15.8 | 否，只做校验 |
| [v40/contract.py](src/wheeled_tasks/v40/contract.py:84) 旧 V4 合同 | 11 | 仅旧线 |
| [core.py](src/wheeled_tasks/v40/core.py:197) `motor_torque_limit` | — | `curve_side=wheel_output` 时只用 `effort_limit`，ratio 不参与 |
| [test_v6_asset_bundle.py](tests/test_v6_asset_bundle.py:33) | 15.8 | 断言，不参与动力学 |

仿真的轮关节力矩/速度本身就是轮端量（`effort_limit=4.5`、`kd=0.6`），轮端 armature 0.00387 kg·m² 也是轮端口径，因此 15.8→15.62 不改变 PhysX 动力学，也不改变策略 35 维观测/6 维动作 ABI。只有拿该字段去做电机端换算时才会体现 **15.8/15.62 = 1.15%** 的差别。

## 3. 旧轮响应标定是 15.8 口径

- 2026-09-29 的轮响应录包早于 15.62 合入（10-14），轮端速度/力矩按当时的 15.8 换算。当前 [identify_wheel_response.py](scripts/identify_wheel_response.py:43) 仍用 15.8 做 rpm↔`dq_api` 的 1e-9 自检，对旧数据成立；用 15.62 固件复录的新数据会直接失败。
- 若只是 15.8→15.62 精确化，辨识出的轮端惯量/摩擦约有 **2.3% / 1.1%** 的比例偏差（反射惯量项随 ratio²）；若换的是整套外部减速箱（如原装 19.2 箱改成 15.62 箱），反射惯量变化约 34%，应重做轮响应辨识，而不是沿用 0.00387 kg·m²。
- 本轮不重标定、不改数值，旧证据按 15.8 口径原样保留。

## 4. 本次决定与后续

- 方案 A（已执行）：新增本文件；在 [验收要求](WHEEL_LEG_ACCEPTANCE.md)、[V5零位/传动对齐](V5_ZERO_TRANSMISSION_ALIGNMENT_20260926.md)、AGENTS.md 和两处代码注释加指针。design JSON、`control_contract` 断言、测试、`identify_wheel_response.py` 逻辑、合同/期望哈希均未改动。
- 若未来选择方案 B（统一到 15.62）：需同时改 design JSON、`control_contract` 校验、相关测试与辨识脚本，并重签合同/期望哈希，已冻结的 playback expectations 会失配，需重新冻结。届时应重新确认轮响应标定是否也要复录。
- 本文件是口径说明，不是新的验收通过或硬件发布依据。
