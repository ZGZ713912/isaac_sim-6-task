# 云台限位补充：2026-10-30

用户确认 yaw 没有位置限位；pitch 相对零位上仰 35°、下俯 30°。v2.0.2 主 URDF、配套 CSV、可动云台 USD、机械限位元数据及生成器已同步。

- `M_joint1`：URDF `continuous`，USD `[-inf,+inf]`，无 ±180° 限制。
- `M_joint2`：正角上仰，URDF `[-0.52359877559829882,+0.6108652381980153] rad`，USD `[-30,+35]°`。
- 六电机底盘训练 USD 的两个云台关节仍然在零位固定。

当前 CAD pitch 轴约沿 −Y，按右手规则正转使前向轴抬高。测试独立求出两端点前向轴的仰角，确认上仰 35°、下俯 30° 没有写反；CAD 原轴线轻微偏斜及全部电机零位保持原值。

URDF 的有界 revolute 需要 effort/velocity 字段。可动参考无云台驱动，因此 pitch effort 设 0、velocity 沿用被动关节 100 rad/s 数值保护；这两项不是云台电机规格。自由云台 USD 仅设置位置限位，没有新增驱动器。

`tests.log` 为本次 37 项通过记录。`native_readback.json` 对可动云台 USD 做原生 PhysX 导入回读，确认 21 个刚体/20 个活动关节、yaw 无界、pitch 的弧度限位正确；不包含对止挡施力或闭环云台控制验证，没有 PPO 或实车动作。

固定云台训练模型的 `robot.usda`、`model_spec.json`、气簧几何和力曲线、轮子模型、控制及运行配置等原有 78 个 bundle 文件逐字节保持不变。上一轮 11500 步证据仍保留在原目录，仅作为这个未变的固定云台六电机物理模型的证据；并不转授可动云台止挡动态资格。

`before/` 保留此次修改涉及的原文件、旧 manifest 和旧验证回执。新版独立回执在版本根目录 `validation_receipt.json`，并由 `report.json` 保存同一份副本。原报告中的历史 manifest SHA 不被改写为新 manifest。

主 URDF：`model/wheel_leg_v2.0.2/轮腿整车URDFv2.0.2/urdf/轮腿整车URDFv2.0.2.urdf`。
可动云台 USD：`model/wheel_leg_v2.0.2/isaac/robot_gimbal_free.usda`。
限位合同：`model/wheel_leg_v2.0.2/isaac/gimbal_limits.json`。
