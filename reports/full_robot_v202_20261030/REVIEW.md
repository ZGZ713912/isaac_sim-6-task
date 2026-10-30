# 整车 v2.0.2 物理资产迁移与验证

2026-10-30 补充：已按用户确认，将可动云台设置为 yaw 无位置限位、pitch 上仰 35°/下俯 30°；六电机训练云台仍固定零位。新版 manifest 与 37 项验证详见 [本次限位补充](gimbal_limits_update001/REVIEW.md)。下文保留首次物理迁移时的资产身份和原生运行证据。

已生成 `model/wheel_leg_v2.0.2`。主 CAD URDF 保留云台 yaw/pitch；当前六电机训练用 USD 固定云台零位，保留整车全部质量、质心和惯量。v2.0.1 和纯底盘原资产通过 SHA-256 检查，保持原字节。未启动 PPO 或实车动作。

## 交付内容

- 完整模型：`model/wheel_leg_v2.0.2/轮腿整车URDFv2.0.2/urdf/轮腿整车URDFv2.0.2.urdf`。
- 固定云台的完整物理机构：`model/wheel_leg_v2.0.2/isaac/robot.usda`；另提供 `robot_gimbal_free.usda` 参考模型。
- 构建：`scripts/build_full_robot_v202.py`、`tools/full_robot_v202.py`。
- 经过哈希验证的组件入口：`src/wheeled_tasks/chassis/full_robot_v202.py`，复用 `V5Control`、`GasSpringCompensation`、`IdentifiedWheelResponse`、`EncoderLegGeometry`。
- 实际 PhysX 运行入口：`scripts/probe_full_robot_v202.py`；每个运行目录保留当时的 `probe.source.py`、完整日志和 JSON，成功运行另有 NPZ 轨迹。

整车 **21.44174 kg**，21 个刚体、18 个活动树关节、2 个固定云台关节、6 个 articulation 树外闭链球约束。约束雅可比秩 12，内部剩余 6 个独立电机自由度。腿和轮的力矩符号、六电机映射及编码器零位没有更改。

根坐标 X 前/Y 左/Z 上；零位云台 link 坐标与 base_link 平行。保留实际 CAD 轴线约 0.261° 的偏斜，yaw 约 +Z、pitch 约 −Y，没有将真实轴线强行改成理想正交轴。65 个姿态的独立重表达检查通过，质量逐项相同；外形变换误差约 4.2e−12，质心误差约 3.4e−13 m，惯量误差约 7.1e−15 kg·m²。

## 物理与控制迁移

沿用纯底盘 P/E 闭链、两侧气簧上安装点约束、40–107° 被动膝内角、80 mm 气簧行程、232 mm 全伸长安装距离。限位按整车 CAD 零位换算。新版气簧压缩量、闭链几何和虚功补偿雅可比重新求解，原测得力曲线及端点趋势保持原值，补偿增益 1。

腿 PD 160/2.5，PD 与气簧补偿合成后限幅 ±40 N·m，再执行向外相对力矩投影。轮输出速度 P0.5、24 V、减速比 15.62、软件限幅 4.5 N·m；左/右聚合延迟 5.5/3.5 ms，每 1 ms 物理步只推进一次。轮子实测总惯量保持约 0.0042695713/0.0043667742 kg·m²，新 CAD 轴向惯量均为 0.00058325 kg·m²，附加 armature 重新计算为总量减 CAD；PhysX 回读验证两者相加没有重复或遗漏。

气簧物理力及补偿均由 Python 组件执行一次；USD 加载本身只建立刚体、约束、限位和碰撞，不能代替运行组件。策略 100 Hz，物理与反馈 PD 1 kHz；TGS 64/32，外力每次求解迭代生效。`runtime.json` 引用当前 randomization v2；未运行域随机化 PPO，也未授予任何鲁棒性资格。

碰撞体从最初 CAD 装配件逐件凸化的 969 个压缩为 **46 个**，显示网格不变。每个非轮凸包最多 64 个顶点；基础/云台按空间分组，以凸包或 26 平面保守外包表示。`collision_coverage.json` 检查 19 个非轮刚体的全部 CAD 网格顶点均被覆盖；轮子保留原 60 mm 半径、25 mm 宽柱体。最终 CPU/GPU 日志没有初版的 GPU 凸包 cooking 回退告警。保守外包可能填充 CAD 凹区，地形、自碰撞与倒地姿态仍需单独审核。

## 实际验证

| 检查 | 结果与范围 |
| --- | --- |
| CPU 单元/回归 | `tests.log`：36 passed，覆盖坐标重表达、USD 双云台版本、闭链秩、虚功、气簧曲线、轮惯量/延迟、新旧编码器表身份及旧逻辑回归。 |
| CPU 固定根零重力 | `native004/report.json`：42/50/75/100/106° 五组，每组 1000 步；生产腿 PD、气簧力/补偿、轮子驱动/制动同时运行；无被动姿态回写。 |
| GPU 固定根零重力 | `native_gpu001/report.json`：同五组，共 5000 步，实际 CUDA PhysX 验证通过。 |
| CPU 自由根、有重力 | `native_freefall002/report.json`：50/75/100° 三组，各 500 步，共 1500 步；无地面。质心与连续抛物线最大差 0.0446 mm，固定云台最大位置误差 0.486 µm，旋转误差约 4.12e−7 rad。 |
| 动力学总体 | 最终资产共 11500 个真实物理步；刚体/关节名称、质量/质心/惯量、轮 armature/摩擦及膝/气簧限位均由引擎回读。最大闭链误差约 0.0206 mm。 |

这些运行验证组件接线和有界离地动力学；没有加载策略，没有地面接触、平衡、自起、地形或随机化行为成绩。全车质量回读差来自 float32 量化，单刚体最大约 6.72e−7 kg；质心最大差约 3.59e−9 m，惯量最大差约 7.76e−8 kg·m²。

首次物理迁移资产 manifest SHA-256：`ad1744f8d9a71050703384654a31b0339dba3a0adcd523db037c2e614747d5cf`。
model_spec SHA-256：`f4a9b373c1dcf448f967ed1eeb0647b1d8652392e3c236b6773dfccb998650d5`。
根目录 `validation_receipt.json` 绑定最终资产、各成功原生报告、运行代码与测试日志；构建时 manifest 中的 `physx_validation=pending` 是构建阶段状态，最新验证状态以该独立回执为准。

## 中间失败与修正

`native001/002` 在物理步开始前因检查器重复旋转惯量张量而拒绝：Omni Physics `get_inertias()` 已返回以质心为原点、在刚体 prim 坐标表达的张量。按[官方张量 API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/109.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html) 修正检查器后通过；没有修改 CAD 惯量或放宽其容差。初版微小螺钉凸包的 GPU cooking 告警随碰撞代理合并消除，初版身份保存在 `initial_generation/`。

`native_freefall001` 的其余机械指标通过，但连续重力轨迹被错误的半隐式离散参考多算半个物理步而拒绝；保留该负结果。`native_freefall002` 改用零初速连续轨迹并保存实际/参考质心完整采样，最大误差 0.0446 mm，原 1 mm 标准未放宽。云台角度检查也改为归一化四元数差，消除 float32 `acos(dot)` 约 1 mrad 的虚假数值底噪。

`native003` 是前一个仅元数据不同版本的 CPU 成功记录，同一 model_spec。其构建身份存于 `cpu_pass_generation/`，不计入上表最终资产的 11500 步。

## 训练与行为准入

已完成本次整车物理资产及六电机低层组件迁移。尚未移植整车姿态/落地库和完整课程入口；这些依赖整车质量、质心与碰撞边界，不能直接复用纯底盘认证。构建 manifest 因此保持 `training_ready=false`，原训练默认模型保持原配置。后续整车训练应建立新几何绑定的姿态库与合同，并从全新 actor/critic/空 Adam 开始。

气簧右侧仍是假定对称，端点为明确的模型趋势，硬止挡未实测；轮侧接地和带载力矩曲线未辨识。当前所有永久能力均未在整车策略上评测：

| 永久能力 | 本版行为状态 |
| --- | --- |
| 前后左右/仰翻与非对称腿初始自起 | 未测 |
| 运动中摔倒后的自主恢复并续行 | 未测 |
| 低/中/高位静站及完成后的双轮有效支撑 | 未测 |
| 正反低速直行 | 未测 |
| 正反高速直行 | 未测 |
| 双向原地自旋 100 rpm 目标/80 rpm 最低保持 | 未测 |
| 前左/前右/后左/后右弧线及切换 | 未测 |
| 静站变高 | 未测 |
| 运动中变高 | 未测 |
| 自旋中变高 | 未测 |
| 正反高速急停 | 未测 |
| 全任务平顺性 | 未测 |
| 随机化、噪声、延迟、摩擦及扰动鲁棒性 | 未测 |

腿轮协同、地形、跳跃及爬阶等继承目标也未测；旧策略成绩不转授整车。每项机器可读状态见 `capability_status.json`。

## 复现

在仓库根目录运行，使用 `/home/noir/miniconda3/envs/isaacsim6/bin/python`。生成器拒绝覆盖已有版本；先选新的 `--output model/新目录` 可重新生成。

```bash
python scripts/build_full_robot_v202.py --output model/wheel_leg_v2.0.2_rebuild
python -m pytest -q tests/test_full_robot_v202.py tests/test_encoder_leg_geometry.py tests/test_rolling_velocity.py tests/test_identified_wheel_response.py tests/test_gas_spring_compensation.py
python scripts/probe_full_robot_v202.py --device cpu --output reports/full_robot_v202_20261030/new_cpu_probe
python scripts/probe_full_robot_v202.py --device cuda:0 --output reports/full_robot_v202_20261030/new_gpu_probe
python scripts/probe_full_robot_v202.py --fixture free_fall --betas 50 75 100 --steps 500 --output reports/full_robot_v202_20261030/new_freefall_probe
python reports/full_robot_v202_20261030/collision_coverage.source.py
```
