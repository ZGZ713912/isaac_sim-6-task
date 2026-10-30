# v2.0.2 腿电机动摩擦接入

默认整车入口 `FullRobotV202()` 与 `WheelLeg-ProneMotion-v4` 现在选择 `model/wheel_leg_v2.0.2/friction_v1/` 下的父资产和训练派生资产。原 `isaac/`、`training/` 和迁移前合同保持原字节，供正在运行的训练和历史比较继续使用。`previous_plan.json`、`migration.json` 保存迁移来源及 SHA-256；新训练仍使用全新 actor、critic 和空 Adam。

四电机的库仑平台初值按报告完整精度保存：左/右 hip **2.019506006848738 N·m**，左/右 knee **2.047246309400147 N·m**。映射分别为 `L_joint1`、`R_joint1` 与 `LL_joint1`、`RR_joint1`，都是主动电机输出轴，未再乘减速比。右侧是用户授权的左侧近似，没有右侧独立测量。

来源是 RMCS 的 `reports/dm_leg_friction_20261031/parameters.json`。本仓库在 `model/dm_leg_friction_20261031/` 保存逐字节参数快照、原报告和来源哈希。参数已用同姿态正反向半差去掉剩余中点偏置；仿真不再叠加该偏置，也不修改气簧力曲线或实车控制器。约0.2 rad/s 的2.1784/2.3011 N·m 仍保留在实测数据中，本次未把它当作静摩擦峰值或另拟合低速模型。

本次只采用平台库仑初值，局部辨识范围约0.6–3.1 rad/s，历史夹角约84°、当前110°标定下约88.56°。全角度、低速和高速使用同一常数是显式建模先验，不是全工作域辨识结论；稀疏夹角表不作插值节点。物理黏性系数、静摩擦脱离阈值、Stribeck 特征速度均保持 `null`。

每个1ms物理步在电机限幅、延迟等执行之后，按**实际**四轴速度加独立被动力矩：

`tau_passive = -F_c * clamp(v_actual / 0.05 rad/s, -1, 1)`。

绝对速度≥0.05 rad/s 时为 `-F_c*sign(v_actual)`，零速为零。0.05 rad/s 只是明示的线性数值平滑带，未辨识为物理黏性或 Stribeck 特征速度，也没有静摩擦保持器。PhysX 原生要求静摩擦力矩≥动摩擦力矩，所以这里不以填入未知静摩擦阈值的方式启用原生干摩擦。依据：[NVIDIA PhysX articulation friction](https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/Articulations.html)。主动腿轴原生静/动/黏性摩擦均读回零，防止重复施加。

控制器仍按40 N·m 限制电机力矩，摩擦在其后叠加；仿真显式执行器的总力矩限幅扩为42.0472463 N·m，仅为被动力矩留空间。摩擦不受电机使能、力矩缩放、反馈噪声或延迟控制，电机断使能时仍耗散。电机电流力矩与被动损耗分别记录，摩擦没有进入控制前馈。

RSL-RL 原有 TensorBoard event 增加 `LegFriction/<motor>_passive_abs_mean_nm` 与 `LegFriction/dissipated_power_mean_w`。逐1ms检查记录实际输入速度、被动摩擦、电机力矩、净施加力矩和 PhysX 执行器读回。原生探针分别检查名义域和全强度随机化域，检查不训练策略，不读取 checkpoint，不构成静站、自起、运动或硬件验收。

构建脚本 `scripts/build_leg_friction_v202.py` 创建新目录并拒绝覆盖既有派生资产。它在完整校验新资产后更新默认v4合同。复现应在尚未生成这两个新目录的干净副本中运行；不要在使用中的冻结目录上覆盖重建。

验证结果和剩余边界见 `reports/leg_motor_friction_20261031/README.md`。
