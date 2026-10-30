# 轮腿整车 v2.0.2

由 `model/wheel_leg_v2.0.1/轮腿整车URDFv2.0.1` 生成，原目录及纯底盘参考资产保持原字节。
本版是整车物理资产及六电机控制组件的迁移，不包含新的 PPO 训练或实车动作。

## 模型入口

- `轮腿整车URDFv2.0.2/urdf/轮腿整车URDFv2.0.2.urdf`：归一化后的完整 CAD 开链 URDF，保留 yaw/pitch 两轴；配套 CSV、ROS 包和 STL。
- `isaac/robot.usda`：底盘训练用模型；云台在零位固定，包含原整车全部 21 个刚体、质量、质心与惯量。
- `isaac/robot_gimbal_free.usda`：云台两轴可动的参考模型；yaw 连续转动、无位置限位，pitch 上仰 35°/下俯 30°，未配置云台电机或控制器。
- `isaac/robot.urdf`：六电机训练模型的树结构表示，含云台固定关节和两个气簧滑动关节。普通 URDF 无法独立表示六个闭链约束；应通过 USD 加载完整物理机构。
- `isaac/runtime.json`、`control.json`、`model_spec.json`、`manifest.json`：控制、几何、参数及 SHA-256 绑定。

整车总质量 **21.44174 kg**。不复制纯底盘的旧质量、质心或惯量，也不从碰撞凸包重新估算惯量。
根坐标为 X 前、Y 左、Z 上。整车 CAD 根坐标通过 Rz(+90°) 转换；共享 link 的局部轴采用纯底盘约定，云台两个 link 在零位与 base_link 平行。实际 CAD 云台轴约 0.261° 的偏斜予以保留：yaw 约沿 +Z，pitch 约沿 −Y。
这是坐标重表达，不是电机反向或编码器重标定；六电机顺序和符号仍为：

`L_joint1, LL_joint1, L_joint3, R_joint1, RR_joint1, R_joint3`

用户于 2026-10-30 确认云台限位：`M_joint1`（yaw）为 continuous；`M_joint2`（pitch）正角度上仰，URDF 范围 `[-0.5235987756, +0.6108652382] rad`，USD 范围 `[-30, +35]°`。`isaac/gimbal_limits.json` 记录这一约定。URDF 的 pitch `effort=0` 表示无驱动参考，`velocity=100` 沿用被动关节数值保护，均不是已辨识的云台电机规格；底盘训练版本仍将两轴固定在零位。

坐标变换详见 `coordinate_mapping.json`；`frame_validation.json` 检查 65 组姿态的网格位置、质心、惯量及关节轴不变。

## 迁移的底盘组件

| 项目 | v2.0.2 处理 |
| --- | --- |
| 闭链 | 左右 P、E、气簧上安装点共六个球约束，置于 articulation 树外；18 个活动树关节扣除 12 个独立约束后为六电机自由度。 |
| 关节限位 | 被动膝内角 40–107°，按整车真实 CAD 零位换算；髋/辅助电机相对角软限位和向外力矩投影沿用现有控制。 |
| 气簧 | 每侧 80 mm 行程、232 mm 全伸长安装点距离；原转动树切为两个滑动副，受力只施加一次。安装点硬止挡仍未经过实测。 |
| 气簧 real2sim | 沿用 20261028 的有效力曲线及显式端点趋势，以整车闭链几何重新求解压缩量和虚功雅可比；右侧曲线仍是假定对称。 |
| 电机控制 | 腿 PD 160/2.5；PD + 气簧补偿后限幅 ±40 N·m，再投影向外相对力矩；轮输出速度 P0.5，软件上限 4.5 N·m。 |
| 轮子 real2sim | 15.62 减速比、24 V，保留实测总惯量/摩擦/空载速度上限；附加 armature = 实测总惯量 − 新 CAD 轴向惯量，避免重复计算。 |
| 延迟 | 轮侧聚合延迟左 5.5 ms、右 3.5 ms，每物理步推进一次；额外驱动延迟为零。 |
| 时钟和求解器 | 策略 100 Hz、物理及低层控制 1 kHz、TGS 64/32，外力每次求解迭代生效。 |
| 编码器几何 | `encoder_geometry.json` 绑定新 model_spec；`FullRobotV202.encoder_geometry()` 明确验证新 SHA，旧纯底盘默认哈希检查保留。 |
| 碰撞 | 保留轮柱体；装配件按空间合并为 46 个保守凸包/柱体，原显示网格不变。沿用关闭自碰撞的研究配置，外界碰撞仍存在；尚未授予自碰撞或地形通过资格。 |
| 域随机化 | 继续引用 `contracts/wheel_leg_randomization_study_v2.json`；腿 P/D、气簧系数固定 1，独立惯量最终 ±6%，无 IMU 晃动。该合同引用不代表已运行整车随机化训练。 |

测得的气簧力只覆盖原数据区间；端点趋势、右腿对称、轮胎接地及带载轮力矩曲线均不被本次迁移变成新的实测证据。

## 生成与验证

在仓库根目录，用 Isaac Sim 6 环境执行：

```bash
/home/noir/miniconda3/envs/isaacsim6/bin/python scripts/build_full_robot_v202.py
```

生成器拒绝覆盖既有输出；可用 `--output model/另一个新目录` 独立复现。`build_receipt.json` 和 `source_provenance.json` 记录输入/输出身份。USD 可以加载约束与刚体；实测气簧和轮子响应由 Python 运行组件持续施加，因此打开 USD 本身不会自动启用这些控制力。

```python
from wheeled_tasks.chassis.full_robot_v202 import FullRobotV202
asset = FullRobotV202()
control, wheels = asset.components(count=1, device="cpu")
encoder = asset.encoder_geometry("cpu")
```

实际 PhysX 探针复用这些生产控制组件，每次 reset 后不回写被动关节姿态：

```bash
/home/noir/miniconda3/envs/isaacsim6/bin/python scripts/probe_full_robot_v202.py \
  --device cpu --output reports/full_robot_v202_20261030/example_native
```

可用 `--device cuda:0` 验证 GPU，或 `--fixture free_fall --betas 75 --steps 500` 检查有重力、自由根的整车质心运动。探针不加载策略、不执行 PPO、不连接机器人。最新实测结果另见 `reports/full_robot_v202_20261030/REVIEW.md` 及本目录 `validation_receipt.json`，避免将生成时的静态 manifest 误读为动力学验收。

## 训练接入边界

本版固定云台，动作仍是六个底盘电机。新训练必须使用全新 actor、critic、空 Adam。旧纯底盘 checkpoint、姿态库和行为成绩不能直接作为整车资格：整车质量、质心、碰撞边界已经改变，完整训练入口仍需生成并验证整车姿态/落地库及课程配置，再按现有能力清单逐项评测。因此构建 manifest 保持 `training_ready=false`；原训练默认资产不会被静默切换。
