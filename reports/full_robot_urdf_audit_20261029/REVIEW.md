# 全车 wheel_legv2.0.0 URDF 静态检查与底盘对比

> 本文保留修改前 SHA `777d875f…a601f4` 的检查记录。腿轴及 RR_joint1 已在后续修改中修正，当前状态与质量/惯量逐件对比见[最新报告](/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/full_robot_axis_inertia_20261029/REVIEW.md)。其余下文原始检查结果按历史快照理解。

检查日期：2026-10-29。输入为用户指定的最新全车目录。

结论：**主腿和传动连杆几何基本沿用已有底盘，且与 wheelbipeV14_2_1 属于同一类同轴双输入、多连杆传动轮腿结构；当前全车文件还存在明确的导出错误和仿真建模缺项，不能直接替代已处理的底盘资产。**

本次读取 XML、全部 21 个 STL、正向运动学、惯量特征值，以及参考 USD 的实际组合层与关节。未运行物理步、PPO 或实车；未修改原 URDF、网格、训练配置及旧资产。

## 输入与可复现结果

- [全车 URDF](/home/noir/Documents/workspace/isaac_wheeled_rl_train/model/wheel_legv2.0.0/urdf/轮腿urdf.SLDASM.urdf)，SHA-256：`777d875faaf4c150225f606076cae5a7e827b34dc3eb02b31712f2c1b9a601f4`。
- 全车导出日志记录 2026-10-29 17:45 完成；目录是 SolidWorks 原始导出包。
- 底盘：逐一读取 `model/纯底盘*/urdf/robot.urdf`，并对照 V5/V6 `source` 下保留的原始导出。
- 参考：读取 `/home/noir/Documents/workspace/example/wheeled-legged_RL/assets/robots/wheelbipeV14_2_1/wheelbipeV14_2.usd` 及其全部组合层。参考项目的 `src/agent_world/assets/wheelbipe_V14_2.py` 也指定这一入口。
- [逐项数据与输入哈希](/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/full_robot_urdf_audit_20261029/audit.json)、[检查脚本](/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/full_robot_urdf_audit_20261029/audit.py)。

复现命令：

```bash
/home/noir/miniconda3/envs/isaacsim6/bin/python /home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/full_robot_urdf_audit_20261029/audit.py
```

## 需要优先修正的地方

### 1. 右侧主动输入 RR_joint1 被错误固定，轴向也异常

[原文件位置](/home/noir/Documents/workspace/isaac_wheeled_rl_train/model/wheel_legv2.0.0/urdf/轮腿urdf.SLDASM.urdf:900)：

```xml
<joint name="RR_joint1" type="fixed">
  ...
  <axis xyz="-0.0935364114890288 -0.241325999083241 -0.965925826289074"/>
```

- 对应左侧 `LL_joint1` 为 `continuous`；V5 原始导出及处理后的 V5/V6/V7 都保留两侧可转动输入。
- 这不是膝部的被动铰：它是六轴控制表中的右侧第二个腿输入，固定后会丢失一个主动自由度。
- 当前 `fixed` 类型不使用其轴字段。但如果只把类型改为可转动、保留现有轴，轴线会偏离公共腿铰轴约 **15°**。
- 原 V5 在相同局部坐标约定下的 `RR_joint1` 轴为 `0 0 1`。应回 CAD 同时检查关节类型和轴线，而不是只改 `fixed` 一处。
- CSV 同样记录 `fixed`，配套 `controller_joint_names` 也已缺少 `RR_joint1`，因此修正时需要同步重新导出这些配套文件。

### 2. 多连杆只是摆在一起，没有形成动力学闭链

当前全车：21 个 link、20 个树关节，其中 19 个 `continuous`、1 个 `fixed`；无 `prismatic`，无树外约束文件或 USD/MJCF 闭链资产。

每侧主腿树为：

```text
base -> L_link1 -> L_link2 -> L_link3（轮）
base -> LL_link1 -> LL_link2 -> LL_link3 -> LL_link4（传动连杆）
```

右侧同理。仅靠这些父子关系，传动连杆末端没有与主腿重新连接。关节零位时销孔重合，并不意味着运动时它们会保持连接。

处理后的 V6/V7 使用每侧三个树外球约束：

| 约束 | 连接内容 |
|---|---|
| `L/R_rocker_P` | 主大腿与传动摇杆，即 `L/R_link1` ↔ `LL/RR_link3` |
| `L/R_upper_E` | 主小腿与传动末杆，即 `L/R_link2` ↔ `LL/RR_link4` |
| `L/R_spring_upper_mount` | 大腿与气簧上端，配合树内气簧滑动副 |

这些约束实际存在于 `robot.usda` / `robot.xml`。旧目录中的 **`robot.urdf` 本身也仅表示树**；它依赖 `constraints.json` 和导出后的仿真资产，不能只拿旧 URDF 单文件当闭链正确的证据。

将旧 V6 测量销点按已有局部坐标变换映射回新全车后，四个传动闭合点的零位残差约 **0.004–0.011 mm**。这支持传动几何仍相同；它只是旧销点移植诊断，不构成新模型的全工作域闭合认证。

### 3. 气弹簧缺少伸缩运动副和力学描述

全车将上、下半支气簧分别挂在两个主腿构件上：

```text
L_link1 -> LLL_link1
L_link2 -> LLL_link2
```

两半气簧之间没有滑动关节，也没有其它连接。因此当前文件中的气簧不能保持同轴伸缩、限制行程或传递正确的轴向力。

V5/V6/V7 的处理资产包含 `L_spring_slide`、`R_spring_slide` 两个 `prismatic`，再补上端闭环。V6 安装定义为全伸销距 232 mm、行程 80 mm。参考 V14 的两侧 `spring2_joint` 同样是滑动副，行程约 `[-0.015, +0.065] m`，总计 80 mm。

即使补齐滑动副，仍要配置与该车安装相符的气簧力、摩擦和阻尼；旧力曲线不能因为结构相似就自动视为新全车实测值。

### 4. 12 个 link 的惯量张量不合格

所有质量均为正且有限，但以下 12 件的惯量含零主惯量，部分还违反主惯量三角不等式：

```text
LLL_link1, LLL_link2, RRR_link1, RRR_link2,
LL_link1, LL_link2, LL_link3, LL_link4,
RR_link1, RR_link2, RR_link3, RR_link4
```

例如 [LLL_link1](/home/noir/Documents/workspace/isaac_wheeled_rl_train/model/wheel_legv2.0.0/urdf/轮腿urdf.SLDASM.urdf:280) 质量 0.035 kg，而整个惯量张量为零；`LL_link1` 为 `[0, 0.00012422, 0.00014195] kg·m²`，后两项不等，即使允许理想线状零主惯量，也不满足刚体惯量关系。

旧处理后的底盘文件在同样检查下均无不合格项。优先从 CAD 以足够有效位数重新导出小零件惯量；如果采用网格积分替代，需要逐件记录材料/质量分布假设，不能统一填极小正数冒充正确惯量。

### 5. 被动膝铰和云台 Pitch 缺少机械限位

- 新全车 `L_joint2` / `R_jonit2` 均为 `continuous`。其中写了 `lower=-3.14`、`upper=3.14`，不能把这当作有限转角铰的保护。
- 旧处理资产已将这两个关节设为 `revolute`，V6 对应腿内角 40–110°，stop105 为 40–105°，V7 为 40–107°。这些历史域需要与新全车机械止挡核对，不能默认全部沿用。
- 新云台 `M_joint2` 为 `continuous`，没有机械角度界。参考 V14 的 Pitch 则有 `[-68.7549°, 0°]` 限位。这个参考值只说明参考资产明确限制了 Pitch，不能照搬成新车的实测范围。
- 全车腿/轮导出字段几乎统一 `effort=100`、`velocity=1`；云台没有 effort/velocity 字段。这些尚不能作为各电机实际能力参数。

### 6. 全精细 STL 同时用作碰撞体，需要重新准备

21 个视觉 STL 均可在本地找到，XML 也没有断开的父子引用；但所有 collision 都直接引用对应视觉 STL，总计 **867,791 个三角面**。仅 base、Yaw、Pitch 分别约 47.2 万、12.8 万、17.2 万面，且这三件均非封闭网格。

这不意味着任何导入器都会失败，但导入器如何生成凸包/分解会影响穿插、孔洞和接触；不能沿用旧资产的碰撞验收。旧 V6/V7 已另做分件凸碰撞和轮圆柱。

针对旧导出中曾出现的 RR_link4 重复网格，当前新全车检查只在 RR_link4 自身找到对应完整几何，未在其余 link 检出同样的重复件。

## 与已有底盘的实际差异

### 几何基本相同，坐标约定不同

新全车 root 的左右方向主要沿 X；旧 V6/V7 为 X 前、Y 左、Z 上。本次用 `p_control = Rz(+90°) * p_full` 做几何对齐，未写回模型。

- 与 V5 原始导出比较，18 个原有树关节的 origin 平移变化最大约 **2.02 μm**，origin 旋转差最大约 **0.00287°**。
- 与处理后的 V6 比较，对齐后的同名 link 原点差最大约 **0.0144 mm**。
- 因而未发现显著的主杆长或传动销位重设计。
- 大腿/传动轴的 q、dq、力矩符号不能直接复制旧 V6：对齐后的前三个可转主动腿轴与旧 V6 轴向相反；右侧第四轴当前固定且异常。两个轮轴方向基本一致。

### 质量和轮径发生了实际变化

| 资产入口 | link / 树关节 | 总质量 kg | 说明 |
|---|---:|---:|---|
| `纯底盘` | 15 / 14 | 12.752 | 较早的连杆拆分，未显式建气簧两半/滑动副 |
| `纯底盘_v5` | 19 / 18 | 14.642 | 已处理的闭链/气簧资产 |
| `纯底盘_v5_232mm` | 19 / 18 | 14.642 | 232 mm 安装版本 |
| `纯底盘_v6_232mm` | 19 / 18 | 13.886 | 新源修正 RR_link2 质量等；仍是底盘 |
| `纯底盘_v6_232mm_stop105` | 19 / 18 | 13.886 | 膝上界 105° |
| `纯底盘_v7_flat_base107` | 19 / 18 | 13.886 | 膝上界 107° |
| `wheel_legv2.0.0` | 21 / 20 | **21.4418** | 加入 Yaw/Pitch，但仍是原始导出 |

更细的质量变化：

| 项目 | V6/V7 底盘 | 新全车 |
|---|---:|---:|
| base_link | 11.620 kg | 16.904 kg |
| 显式 Yaw + Pitch | 无 | 1.0263 + 1.2475 = 2.2738 kg |
| 剔除显式云台后的质量 | 13.886 kg | 19.168 kg |
| 轮子单件质量 | 0.200 kg | 0.200 kg |
| 轮子半径 | STL/碰撞均约 60 mm | STL 约 **52.5 mm** |

全车比 V6/V7 总重约 **54.4%**；不能仅理解为给旧底盘加了 2.2738 kg 云台，base_link 自身还增加了 5.284 kg，需要按实际 BOM/称重核对所属部件。

新轮 STL 最大径向距离两侧均约 0.052500 m，无额外 scale；旧轮 STL 约 0.060000 m，碰撞圆柱半径同为 0.06 m。新几何轮径为 105 mm，旧为 120 mm，减小 12.5%。这是确定的模型变化，不自动意味着新轮错误；需要与实车轮径核对。当前训练合同仍有 `wheel_radius_m=0.06`，未来使用全车时需要同步核对轮速换算、站高/接地放置及轮惯量。

导出零位 q=0 时，新全车对齐后的总 COM 约 `[-4.864, +4.023, -26.550] mm`，V6 为 `[-19.810, +0.258, -55.238] mm`。该姿态下 COM 高约 28.7 mm；这不是所有站高/云台角度下的固定偏移。

## 与 wheelbipeV14_2_1 的结构对照

参考 USD 实际含 37 个刚体、34 个旋转关节、2 个滑动关节、6 个树外球约束。16 个额外 guide link/guide 关节也计入这个数量，因此不能直接拿它的 link 数判断腿多了多少自由度。其总质量约 23.3016 kg。

| 新全车构件/功能 | 参考 V14 构件 | 新全车平面长度 mm | V14 平面长度 mm |
|---|---|---:|---:|
| `L/R_link1` 主大腿 | `left/right_rear1_link` | 210.000 | 210.000 |
| `L/R_link2` 膝到轮 | `left/right_rear2_link` | 250.000 | 250.005 |
| `LL/RR_link1` 输入短曲柄 | `left/right_front1_link` | 113.400 | 113.403 |
| `LL/RR_link2` 传动连杆 | `left/right_front2_link` | 135.000 | 135.003 |
| `LL/RR_link3` 摇杆两树关节间 | `left/right_front3_link` | 200.509 | 200.513 |
| `LL/RR_link4` 末杆至闭合销 | `left/right_front4_link` | 约 96.600，沿用旧测销点 | 96.600 |

V14 长度按实际关节坐标在 XZ 平面投影计算，忽略沿铰轴的装配错层；新车按其连杆局部平面计算。杆长差只有数微米级，支持非常接近的连杆机构。

V14 的 `loop_joints` 明确存在 `f3r1`、`f4r2`、`s2r2` 每侧各一个球约束，且均排除在 articulation 树外；最后一个连的是气簧末端与小腿。其气簧树从大腿向外延伸。旧 V6 的气簧树从小腿向外延伸、再闭回大腿，属于同一机械连接的不同树切分，不能将父子关系不同直接判成机构不同。

仍存在实际装配差异，例如新车气簧大腿侧安装点到髋轴的平面距离约 51 mm，而 V14 约 45 mm；因此其气簧力矩臂和力曲线不能按同杆长直接套用。

### 云台方向合理，但尺寸、质量和限位不同

新车 `base_link -> M_link1 -> M_link2` 对应参考的 `base_link -> gimbal_yaw_link -> gimbal_pitch_link`。

- `M_joint1` 的局部轴虽写成 `[1, -0.0029743, 0]`，经过 origin 旋转后，root 中的单位轴约为 `[0.00452034, 0.00002331, 0.99998978]`，接近竖直，偏差约 **0.259°**。
- `M_joint2` 的 root 轴约为 `[-0.99998978, 0.00000373, 0.00452036]`，是横向 Pitch 轴；对齐到 X 前/Y 左/Z 上后接近负 Y。两轴近乎正交。
- 因此没有证据表明 Yaw/Pitch 轴被交换；不能仅因局部轴不是常见的 Z/Y 就判错。
- 新车 Yaw 轴原点高度约 **81.497 mm**，Yaw 到 Pitch 原点距离 **97 mm**；参考分别约 **54.950 mm**，以及沿 Z 上移 **180 mm** 并带前后/左右偏置。不能把两者当作相同外参。
- 新云台总质量 **2.2738 kg**；参考约 **4.0231 kg**。两车上装惯性不同。

## 建议修正顺序

1. 回 CAD 修正 RR_joint1 的类型/轴线，并同步控制关节清单；保留原始导出作为来源。
2. 重新导出足够精度的惯量，逐件检查正定性和主惯量三角不等式。
3. 在独立的全车仿真资产中建立正确树、两个气簧滑动副和六个闭链约束；复核约束秩与腿部工作域。
4. 以实车尺寸核对 52.5 mm 轮半径、腿和 Pitch 止挡、质量归属与各轴符号；保留历史模型的原始定义。
5. 制作独立碰撞体、检查闭链全行程及云台运动的干涉，再进行有限物理验证。完成这些后，才具备用新全车开展训练的建模基础。

![几何对比](/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/full_robot_urdf_audit_20261029/geometry_comparison.png)

图为各资产导出零位下的左腿/云台投影，红圈标记旧资产及参考 USD 的实际闭环连接点。零位约定不同，不用该图直接比较站立性能。
