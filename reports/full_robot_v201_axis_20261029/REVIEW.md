# 全车 v2.0.1 关节轴统一与导出检查

后续变更：本报告记录轴向修复完成时点。用户随后要求沿用旧底盘圆柱轮碰撞，已在新的[物性与碰撞报告](/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/full_robot_v201_properties_20261029/REVIEW.md)记录；当前 URDF SHA 和碰撞定义以该报告为准。本阶段的质量、惯量、外观和关节结果仍适用。

日期：2026-10-29。源目录为 `/home/noir/Documents/workspace/isaac_wheeled_rl_train/model/wheel_leg_v2.0.1/轮腿urdf.SLDASM`。

## 修改结果

- 18 个腿部转动关节统一为与纯底盘 v7 相同的左右朝外约定；10 个原向内关节反号。
- `RR_joint1` 原为 fixed 且 axis=[0,0,0]，现恢复 continuous 并设置右侧朝外轴。本次只发现这 1 个误设 fixed 的关节。
- 同步 URDF、配套 CSV 和 YAML；清除 YAML 开头空名称、补回 RR_joint1，清单与 20 个可动关节完全对应。
- 21 个 link 元素、质量、质心、惯量、网格、碰撞、关节 origin、limit 和云台两关节均保留本次 SW 导出值；全部包文件哈希检查确认仅上述 3 份文件改变。
- root 沿用 CAD 坐标；朝外方向在全车 root 中左侧约为 `[+0.999989403, +0.000003673, -0.004603657]`，右侧反向，与此前修正的 v2.0.0 一致。局部 axis 需按各 joint frame 计算，不能全部机械填写同一向量。

## 小数精度与轮径

- 21/21 惯量张量正定，21/21 满足主惯量三角不等式，最小主惯量为 4.78925767705e-07 kg·m²。
- v2.0.0 中原为零的 16 个对角分量已全部恢复非零；两件气弹簧上半件现为 `(4.8e-7, 3.039e-5, 3.064e-5)` kg·m²，并保留非零惯性积。
- 两轮 STL 的径向最大半径约 60.000 mm，即直径 120.000 mm；v2.0.0 为半径 52.500 mm、直径 105.000 mm。这里读取的是 URDF 实际引用网格及其 scale。
- 新轮惯量对角项为 `(0.000301, 0.000301, 0.00058325)` kg·m²；全车总质量 21.44174 kg。本次没有覆写任何物性。

## 逐关节处理

| 关节名 | 处理 |
|---|---|
| L_joint1 | 向内 → 朝外，q/dq/力矩正方向反号 |
| L_joint2 | 保持朝外，并统一微小轴线偏差 |
| L_joint3 | 保持朝外，并统一微小轴线偏差 |
| LLL_joint2 | 向内 → 朝外，q/dq/力矩正方向反号 |
| LLL_jointt1 | 向内 → 朝外，q/dq/力矩正方向反号 |
| R_joint1 | 向内 → 朝外，q/dq/力矩正方向反号 |
| R_jonit2 | 保持朝外，并统一微小轴线偏差 |
| R_joint3 | 保持朝外，并统一微小轴线偏差 |
| RRR_joint2 | 向内 → 朝外，q/dq/力矩正方向反号 |
| RRR_joint1 | 向内 → 朝外，q/dq/力矩正方向反号 |
| LL_joint1 | 向内 → 朝外，q/dq/力矩正方向反号 |
| LL_joint2 | 保持朝外，并统一微小轴线偏差 |
| LL_joint3 | 保持朝外，并统一微小轴线偏差 |
| LL_link4 | 保持朝外，并统一微小轴线偏差 |
| RR_joint1 | fixed / 零轴 → continuous / 朝外 |
| RR_joint2 | 向内 → 朝外，q/dq/力矩正方向反号 |
| RR_joint3 | 向内 → 朝外，q/dq/力矩正方向反号 |
| RR_joint4 | 向内 → 朝外，q/dq/力矩正方向反号 |

既有名称 `R_jonit2`、`LLL_jointt1`、`LL_link4` 沿用源文件，避免额外改变关节接口。

## 验证与边界

- 修改前后 21 个 link 的零位变换最大差为 0。
- 18 个关节分别做正角度有限差分，方向误差峰值 1.01e-13。
- 64 组独立随机关节姿态中，18 条腿轴均保持对应侧朝外，方向误差峰值 1e-15。
- XML 树、所有网格引用、CSV 字段、YAML 清单、全部惯量基本物理检查均通过。
- 本次为离线模型检查，没有启动仿真、训练或实车；闭链与气弹簧伸缩副仍按此前约定留待后续。
- 轴反号关节的 q、dq、力矩正方向对应变化见回执，已有这些关节的非零轨迹应同步换号。没有修改已存在的训练资产或控制配置。

## 原始文件与回执

- 原始 SW 导出 URDF SHA-256：`6e2b437dc2f4f26b78e2a3ca1ac4e35625e4304a7d574ace0049d67997f5a8ff`。
- 修正后 URDF SHA-256：`b46af65f0b625b3dcbe996dd0339ff1b30d59f67b4b7da0ae7e6f395393e0a8b`。
- [修正后的 URDF](/home/noir/Documents/workspace/isaac_wheeled_rl_train/model/wheel_leg_v2.0.1/轮腿urdf.SLDASM/urdf/轮腿urdf.SLDASM.urdf)
- [关节清单](/home/noir/Documents/workspace/isaac_wheeled_rl_train/model/wheel_leg_v2.0.1/轮腿urdf.SLDASM/config/joint_names_轮腿urdf.SLDASM.yaml)
- [原始导出备份](/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/full_robot_v201_axis_20261029/before/轮腿urdf.SLDASM.urdf)
- [逐关节修改与方向映射](/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/full_robot_v201_axis_20261029/axis_changes.json)
- [完整验证与物性/网格数据](/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/full_robot_v201_axis_20261029/validation.json)
